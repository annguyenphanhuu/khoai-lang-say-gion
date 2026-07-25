#!/usr/bin/env bash
# Đo CHI PHÍ CPU MỖI DECODE STEP của EngineCore tại ĐÚNG điểm vận hành (C=1).
#
# Vì sao C=1: `tokens_per_sec` bất biến 1.4% qua 20+ lượt portal ⇒ benchmark do trace
# arrival quyết định, server không bão hoà ⇒ batch decode ≈ 1 (on-tap/07 §3).
# Profile ở C=24 (đã làm ở Wave6) cho bức tranh frontend-nặng KHÔNG đại diện.
#
# Metric: số mẫu py-spy KHÔNG phải `cuda.synchronize` / số token sinh ra.
#   - `cuda.synchronize` = busy-wait chờ GPU của rig 3050 (chậm hơn H200 nhiều)
#     ⇒ phải loại, nếu không mọi thứ chìm trong đó.
#   - Phần còn lại = công việc Python/torch-dispatch THẬT mỗi step. Đây là đại lượng
#     chuyển được sang portal (cùng code, cùng số op; chỉ tốc độ core khác).
#
# Dùng:  bench/cpu_step_probe.sh <docker-image> <nhãn>
set -u
IMG="$1"; TAG="${2:-run}"; DUR="${DUR:-45}"
NAME="cpuprobe-$$"

docker rm -f "$NAME" >/dev/null 2>&1
MSYS_NO_PATHCONV=1 docker run -d --name "$NAME" --gpus all --cpus 3 --shm-size 2g \
  --cap-add SYS_PTRACE \
  -v //d/models/LFM2.5-1.2B-Instruct://model:ro \
  -v //d/khoai-lang-say-gion/bench://bench:ro \
  "$IMG" \
  --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
  --max-model-len=4096 --gpu-memory-utilization=0.79 --tensor-parallel-size=1 \
  --enable-prefix-caching --max-num-seqs=32 >/dev/null || { echo "START FAIL"; exit 1; }

for i in $(seq 1 50); do
  if docker exec "$NAME" curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200; then break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "[$TAG] ENGINE DIED"; docker logs --tail 25 "$NAME" 2>&1 | tail -20; docker rm -f "$NAME" >/dev/null; exit 1
  fi
  sleep 10
done

docker exec "$NAME" pip install py-spy -q >/dev/null 2>&1
PID=$(docker exec "$NAME" bash -c "ps -eo pid,cmd | grep -i 'VLLM::EngineCore' | grep -v grep | awk '{print \$1}'" | tr -d '\r')
[ -z "$PID" ] && { echo "[$TAG] no EngineCore pid"; docker rm -f "$NAME" >/dev/null; exit 1; }

( docker exec "$NAME" python3 //bench/loadgen_c1.py "$DUR" > /tmp/lg_$TAG.txt 2>&1 ) &
sleep 6
docker exec "$NAME" py-spy record --pid "$PID" --duration $((DUR-12)) --rate 250 \
  --nonblocking --format raw --output //tmp/prof.txt >/dev/null 2>&1
wait

TOKENS=$(grep -o 'tokens=[0-9]*' /tmp/lg_$TAG.txt | head -1 | cut -d= -f2)
echo "=== [$TAG] $(cat /tmp/lg_$TAG.txt | tail -1)"
docker exec "$NAME" bash -c "
awk -F';' '{c=\$NF; sub(/^.* /,\"\",c); if (\$0 ~ /cuda\/streams.py|synchronize/) sync+=c; else work+=c}
END{printf \"SYNC=%d WORK=%d\n\", sync, work}' //tmp/prof.txt
"
echo "--- [$TAG] top leaf (không tính sync) ---"
docker exec "$NAME" bash -c "
awk -F';' '\$0 !~ /synchronize/ {n=\$NF; c=n; sub(/ [0-9]+\$/,\"\",n); sub(/^.* /,\"\",c); s[n]+=c}
END{for(k in s) printf \"%6d %s\n\", s[k], k}' //tmp/prof.txt | sort -rn | head -12
"
docker rm -f "$NAME" >/dev/null 2>&1
echo "[$TAG] TOKENS=$TOKENS  -> chia WORK cho TOKENS để ra 'mẫu CPU / step'"
