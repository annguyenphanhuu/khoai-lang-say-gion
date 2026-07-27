#!/bin/bash
set -u
cd /workspace
rm -rf /workspace/spyauto /workspace/spysnap; mkdir -p /workspace/spysnap
pkill -f 'vllm.entrypoints.openai.api_server'; sleep 6
nohup taskset -c 0-2 env SPY_ALL=1 SPY_DIR=/workspace/spyauto SPY_EVERY=4 SPY_HZ=100 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_LOGGING_LEVEL=WARNING \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model --served-model-name LFM2.5-1.2B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.225 --enable-prefix-caching \
  --quantization fp8 --disable-uvicorn-access-log --renderer-num-workers=2 \
  > /workspace/spysnap/server.log 2>&1 &
for i in $(seq 1 60); do c=$(curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8000/health); [ "$c" = "200" ] && break; sleep 4; done
echo health=$c
# warm rep
BASE=http://localhost:8000 NCONV=70 TURNS=6 OUT=300 RATE=8 SEED=42 taskset -c 8-40 python3 repo/bench/ers_harness.py > /workspace/spysnap/warm.out 2>&1
sleep 3
BASE=http://localhost:8000 NCONV=70 TURNS=6 OUT=300 RATE=8 SEED=42 taskset -c 8-40 python3 repo/bench/ers_harness.py > /workspace/spysnap/run.out 2>&1 &
HP=$!
sleep 8; cp /workspace/spyauto/*.txt /workspace/spysnap/ 2>/dev/null
wait $HP
grep -E 'ERS_HIST|s_ttft' /workspace/spysnap/run.out
touch /workspace/spyall.done
