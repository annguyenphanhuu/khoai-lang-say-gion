#!/usr/bin/env bash
# ROADMAP ① — Verify chế độ CUDA Graph thực tế + conv có bị chạy EAGER không.
#
# Trả lời 2 câu quyết định toàn bộ chiến lược:
#  (1) vLLM chọn cudagraph_mode nào? Có narrow về PIECEWISE không? (grep log INFO)
#  (2) causal_conv1d / short_conv có nằm trong TOP CPU-leaf lúc decode không?
#      - CÓ  ⇒ conv chạy eager ⇒ graph bị narrow ⇒ hướng ① có +5→+10 điểm.
#      - KHÔNG ⇒ conv nằm TRONG graph ⇒ giả thuyết "custom ShortConv kernel" đóng dứt điểm,
#               ① = 0 điểm, dồn sức sang ② (--async-scheduling).
#
# Dùng:  bench/verify_cudagraph.sh <docker-image> [thêm cờ vllm...]
#   VD:  bench/verify_cudagraph.sh annguyenphanhuu/vllm-fastsse:fullfp8-20260721-r1
#        bench/verify_cudagraph.sh <img> '--compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
#
# LƯU Ý: chạy trên rig local (vd RTX 3050) chỉ phản ánh LOGIC chọn mode (mamba UNIFORM_BATCH ×
#  backend support) — phần lớn arch-independent. Backend cụ thể có thể khác H200, nên đọc dòng
#  "attention backend" trong log.
set -u
IMG="${1:?can dockerfile image}"; shift || true
NAME="cgverify-$$"
docker rm -f "$NAME" >/dev/null 2>&1

MSYS_NO_PATHCONV=1 docker run -d --name "$NAME" --gpus all --cpus 3 --shm-size 2g \
  --cap-add SYS_PTRACE \
  -e VLLM_LOGGING_LEVEL=INFO -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  -v //d/models/LFM2.5-1.2B-Instruct://model:ro \
  -v //d/khoai-lang-say-gion/bench://bench:ro \
  "$IMG" \
  --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
  --max-model-len=4096 --gpu-memory-utilization=0.79 --tensor-parallel-size=1 \
  --enable-prefix-caching --quantization=fp8 "$@" >/dev/null || { echo "START FAIL"; exit 1; }

# chờ health (hoặc chết)
for i in $(seq 1 60); do
  if docker exec "$NAME" curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200; then break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "ENGINE DIED:"; docker logs --tail 40 "$NAME" 2>&1 | tail -40; docker rm -f "$NAME" >/dev/null; exit 1
  fi
  sleep 5
done

echo "=========== (1) QUYẾT ĐỊNH CUDA GRAPH MODE (grep log) ==========="
docker logs "$NAME" 2>&1 | grep -iE "cudagraph|capturing cuda|splitting_ops|attention backend|using .*backend|piecewise|full_and_piecewise|full_decode|falling back|not supported|narrow|adjust.*mamba|mamba" | sed 's/^/  /' | tail -40
echo "  ↑ tìm: 'cudagraph_mode=...'; WARNING narrow→PIECEWISE; 'Capturing CUDA graphs'; backend đang dùng."

echo ""
echo "=========== (2) CONV CHẠY EAGER? (py-spy top leaf khi decode C=1) ==========="
docker exec "$NAME" pip install py-spy -q >/dev/null 2>&1
PID=$(docker exec "$NAME" bash -c "ps -eo pid,cmd | grep -i 'VLLM::EngineCore' | grep -v grep | awk '{print \$1}'" | tr -d '\r')
[ -z "$PID" ] && { echo "  no EngineCore pid"; docker rm -f "$NAME" >/dev/null; exit 1; }
( docker exec "$NAME" python3 //bench/loadgen_c1.py 25 >/tmp/cgld_$$.txt 2>&1 ) &
sleep 5
docker exec "$NAME" py-spy record --pid "$PID" --duration 15 --rate 250 \
  --nonblocking --format raw --output //tmp/cgprof.txt >/dev/null 2>&1
wait
echo "  Top CPU leaf (bỏ sync) — soi 'causal_conv1d', 'short_conv', 'torch.split':"
docker exec "$NAME" bash -c "
awk -F';' '\$0 !~ /synchronize/ {n=\$NF; c=n; sub(/ [0-9]+\$/,\"\",n); sub(/^.* /,\"\",c); s[n]+=c}
END{for(k in s) printf \"%6d %s\n\", s[k], k}' //tmp/cgprof.txt | sort -rn | head -15 | sed 's/^/    /'
"
echo ""
echo "  ⇒ causal_conv1d/short_conv KHÔNG ở top  ⇒ conv TRONG graph ⇒ ①=0, sang ②."
echo "  ⇒ causal_conv1d/short_conv Ở top         ⇒ conv EAGER ⇒ graph narrow ⇒ ① có +5→+10."
docker rm -f "$NAME" >/dev/null 2>&1
