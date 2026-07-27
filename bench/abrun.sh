#!/bin/bash
# usage: TAG=x EXTRA="--flag" REPS=2 MPS_PCT=0 bash abrun.sh
set -u
cd /workspace
TAG=${TAG:-t}; EXTRA=${EXTRA:-}; REPS=${REPS:-2}; SPY=${SPY:-0}
CORES=${CORES:-0-2}; RATE=${RATE:-8}; MPS_PCT=${MPS_PCT:-0}
O=/workspace/ab/$TAG
mkdir -p $O; rm -f $O/*.txt $O/*.out $O/done
pkill -f 'vllm.entrypoints.openai.api_server'; pkill -f spyserve; sleep 6
MPSENV=""
if [ "$MPS_PCT" != "0" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d
  sleep 2
  pgrep -f nvidia-cuda-mps-control >/dev/null && echo "MPS alive pct=$MPS_PCT" || echo "MPS DOWN"
  MPSENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT"
fi
LAUNCH="python3 -m vllm.entrypoints.openai.api_server"
[ "$SPY" = "1" ] && LAUNCH="python3 /workspace/bench/spyserve.py vllm.entrypoints.openai.api_server"
nohup taskset -c $CORES env SPY_OUT=$O/spy.txt SPY_HZ=100 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VLLM_LOGGING_LEVEL=${LOGLVL:-WARNING} $MPSENV ${ENVX:-} $LAUNCH \
  --model /workspace/model --served-model-name LFM2.5-1.2B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.225 --enable-prefix-caching \
  --quantization fp8 $EXTRA > $O/server.log 2>&1 &
for i in $(seq 1 60); do c=$(curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8000/health); [ "$c" = "200" ] && break; sleep 4; done
echo "TAG=$TAG EXTRA='$EXTRA' MPS=$MPS_PCT health=$c"
PID=$(pgrep -f 'api_server --model|spyserve.py vllm' | head -1)
for r in $(seq 1 $REPS); do
  sleep 3
  curl -s localhost:8000/metrics > $O/m0_$r.txt
  [ "$SPY" = "1" ] && kill -USR2 $PID
  BASE=http://localhost:8000 NCONV=70 TURNS=6 OUT=300 RATE=$RATE SEED=42 \
    taskset -c 8-40 python3 repo/bench/ers_harness.py > $O/run_$r.out 2>&1
  curl -s localhost:8000/metrics > $O/m1_$r.txt
  [ "$SPY" = "1" ] && kill -USR1 $PID
  echo "--- rep $r ---"; grep -E 'ERS_HIST|s_ttft|BATCH' $O/run_$r.out
  python3 /workspace/bench/ttft_split.py $O/m0_$r.txt $O/m1_$r.txt | tail -5
done
touch $O/done
