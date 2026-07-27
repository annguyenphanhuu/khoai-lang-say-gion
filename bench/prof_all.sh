#!/usr/bin/env bash
# Track A end-to-end: boot vLLM (K3 config, MPS-limited SMs) UNDER nsys, then drive a
# steady B~27 decode load so the capture window lands in steady-state decode.
#
# nsys 2026.1.3 dropped --pid attach, so the profiler has to own the launch. --delay skips
# boot+cudagraph capture; --kill=none leaves the server up afterwards for follow-up probes.
# --cuda-graph-trace=node is mandatory: decode runs a FULL cudagraph, which otherwise
# collapses into one "graph launch" row with no kernels inside.
set -u
O=/workspace/prof27
MPS_PCT="${MPS_PCT:-14}"
GPU_UTIL="${GPU_UTIL:-0.2025}"
DELAY="${DELAY:-240}"
DUR="${DUR:-20}"
CONC="${CONC:-34}"
TAG="${TAG:-step}"

rm -f $O/prof.done $O/$TAG.nsys-rep $O/$TAG.sqlite $O/load.log $O/server.log
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null
pkill -f 'nsys' 2>/dev/null
sleep 5

MPSENV=""
if [ "$MPS_PCT" != "0" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
  sleep 2
  pgrep -f nvidia-cuda-mps-control >/dev/null && echo "MPS alive (pct=$MPS_PCT)" || echo "MPS DOWN"
  MPSENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT"
fi

cd /workspace
nohup nsys profile \
    --delay="$DELAY" --duration="$DUR" --kill=none \
    -t cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
    -o $O/$TAG --force-overwrite true \
  taskset -c 0-2 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_LOGGING_LEVEL=INFO $MPSENV \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/model --served-model-name LFM2.5-1.2B-Instruct \
    --host 0.0.0.0 --port 8000 --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_UTIL" --tensor-parallel-size 1 \
    --enable-prefix-caching --quantization online_w4a8 ${EXTRA_ARGS:-} \
  > $O/server.log 2>&1 &

T0=$(date +%s)
for i in $(seq 1 100); do
  c=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
  [ "$c" = "200" ] && break
  sleep 3
done
BOOT=$(( $(date +%s) - T0 ))
echo "healthy after ${BOOT}s (capture fires at t=${DELAY}s)"

# In SUSTAIN mode each worker issues ONE very long generation, so the batch is pinned at
# exactly CONC with zero prefill during the capture. Start it late enough that prefill of
# all CONC prompts is finished, but early enough that they are still decoding at t=DELAY.
LEAD=${LEAD:-45}
SLEEP=$(( DELAY - BOOT - LEAD ))
[ $SLEEP -gt 0 ] && sleep $SLEEP
echo "starting load at t=$(( $(date +%s) - T0 ))s"
LOADDUR=$(( LEAD + DUR + 30 ))
DUR=$LOADDUR CONC=$CONC SUSTAIN=${SUSTAIN:-1} OUT=${OUT:-8000} \
  BASE=http://localhost:8000 python3 $O/load27.py > $O/load.log 2>&1
echo "--- load done ---"
tail -3 $O/load.log
ls -la $O/$TAG.nsys-rep 2>/dev/null || { echo "NO REPORT"; tail -20 $O/server.log; }
touch $O/prof.done
