#!/usr/bin/env bash
# PROFILING GPU-forward của DECODE (HANDOFF §8.3) — chạy TRÊN RunPod H100/H200.
#
# Mục tiêu: biết CHÍNH XÁC 3ms/token decode chia ra sao — op/layer nào nóng nhất
#   (conv? attention? mlp? norm? gap giữa kernel?) để fuse ĐÚNG chỗ.
#
# Dùng:  bench/nsys_decode.sh <docker-image> [thêm cờ vllm...]
#   VD:  bench/nsys_decode.sh annguyenphanhuu/vllm-w4a8:online-r1 --quantization=online_w4a8
#        bench/nsys_decode.sh <img> --quantization=fp8
#
# Ràng buộc portal (để số SÁT): --cpus 3, VRAM ~18GB (chỉnh gpu-mem-util theo card).
# Xuất: /workspace/out/decode.nsys-rep (mở bằng Nsight UI) + bảng top-kernel in ra stdout.
set -u
IMG="${1:?can docker image}"; shift || true
NAME="nsysdec-$$"
OUTDIR="${OUTDIR:-/workspace/out}"
MODEL_MNT="${MODEL_MNT:-/workspace/model}"   # nơi đã upload/clone model trên pod
mkdir -p "$OUTDIR"
docker rm -f "$NAME" >/dev/null 2>&1

# H100 80GB: 18/80 ≈ 0.225 để mô phỏng MiG slice 18GB. Card khác chỉnh lại.
GPU_UTIL="${GPU_UTIL:-0.225}"

echo ">> Khởi động vLLM (constrained portal-like: 3 core, VRAM~18GB, util=$GPU_UTIL)"
docker run -d --name "$NAME" --gpus all --cpus 3 --shm-size 2g \
  --cap-add SYS_PTRACE \
  -e VLLM_LOGGING_LEVEL=INFO -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$MODEL_MNT":/model:ro \
  -v "$OUTDIR":/out \
  "$IMG" \
  --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
  --max-model-len=6144 --gpu-memory-utilization="$GPU_UTIL" --tensor-parallel-size=1 \
  --enable-prefix-caching --disable-log-stats "$@" >/dev/null || { echo "START FAIL"; exit 1; }

# chờ health hoặc chết
for i in $(seq 1 60); do
  code=$(docker exec "$NAME" curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || true)
  [ "$code" = "200" ] && { echo ">> healthy"; break; }
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "ENGINE DIED:"; docker logs --tail 60 "$NAME" 2>&1 | tail -60; docker rm -f "$NAME" >/dev/null; exit 1
  fi
  sleep 5
done

# warm prefix cache 1 lần để đo ĐÚNG steady-state decode (không dính prefill)
docker exec "$NAME" python3 - <<'PY' >/dev/null 2>&1 || true
import json,urllib.request
b=json.dumps({"model":"LFM2.5-1.2B-Instruct","temperature":0,"max_tokens":20,
 "messages":[{"role":"system","content":"warm "*400},{"role":"user","content":"hi"}]}).encode()
urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions",
 data=b,headers={"Content-Type":"application/json"}),timeout=120).read()
PY

PID=$(docker exec "$NAME" bash -c "ps -eo pid,cmd | grep -i 'VLLM::EngineCore' | grep -v grep | awk '{print \$1}'" | tr -d '\r')
[ -z "$PID" ] && { echo "no EngineCore pid"; docker rm -f "$NAME" >/dev/null; exit 1; }
echo ">> EngineCore PID=$PID"

# nsys có sẵn trong image CUDA? nếu không, cài nhanh.
docker exec "$NAME" bash -c "command -v nsys" >/dev/null 2>&1 || \
  docker exec "$NAME" bash -c "pip install -q nsys 2>/dev/null; apt-get update -qq && apt-get install -y -qq nsight-systems-cli 2>/dev/null" || true

echo ">> Bơm C=1 decode + nsys profile 12s (chỉ EngineCore forward)"
( docker exec "$NAME" python3 /bench/loadgen_c1.py 30 >/out/loadgen_c1.txt 2>&1 ) &
sleep 4
docker exec "$NAME" bash -c \
  "nsys profile --pid $PID --duration 12 --trace cuda,nvtx,osrt \
     --output /out/decode --force-overwrite true 2>/dev/null" || {
  echo '!! nsys thất bại — fallback torch profiler qua py-spy (xem verify_cudagraph.sh)';
}
wait

echo ""
echo "=========== TOP CUDA KERNELS (decode steady-state) ==========="
docker exec "$NAME" bash -c \
  "nsys stats --report cuda_gpu_kern_sum --format table /out/decode.nsys-rep 2>/dev/null | head -40" \
  || echo "  (mở /workspace/out/decode.nsys-rep bằng Nsight Systems UI để xem timeline)"

echo ""
echo "=========== C=1 tbt (sanity) ==========="
docker exec "$NAME" cat /out/loadgen_c1.txt 2>/dev/null | tail -3

echo ""
echo ">> Artifact: $OUTDIR/decode.nsys-rep  (tải về mở bằng Nsight Systems)"
echo ">> Đọc: kernel nào chiếm % lớn nhất + tổng gap giữa kernel = launch overhead còn lại."
docker rm -f "$NAME" >/dev/null 2>&1
