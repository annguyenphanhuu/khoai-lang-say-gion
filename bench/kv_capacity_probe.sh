set -u
LABEL="$1"; shift
NAME="c2-$$"
docker rm -f "$NAME" >/dev/null 2>&1
MSYS_NO_PATHCONV=1 docker run -d --name "$NAME" --gpus all --cpus 3 --shm-size 2g \
  -v //d/models/LFM2.5-1.2B-Instruct://model:ro \
  -v "//c/Users/anngu/AppData/Local/Temp/claude/D--khoai-lang-say-gion/b71136b8-90b0-4981-96dc-0830cacb8c20/scratchpad://probe" \
  annguyenphanhuu/vllm-w4a8:online-r2 \
  --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
  --max-model-len=8192 --gpu-memory-utilization=0.80 --tensor-parallel-size=1 \
  --quantization=fp8 --enable-prefix-caching "$@" >/dev/null 2>&1
for i in $(seq 1 40); do
  c=$(docker exec "$NAME" curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
  [ "$c" = "200" ] && break
  docker ps -q -f name="$NAME" | grep -q . || { echo "[$LABEL] DIED"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
  sleep 8
done
echo "##### $LABEL"
docker logs "$NAME" 2>&1 | grep -oE "block_size=[0-9]+|mamba_block_size=[0-9]+|Padding mamba page size by [0-9.]+%|Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | sort -u | tr '\n' ' '; echo
docker rm -f "$NAME" >/dev/null 2>&1
