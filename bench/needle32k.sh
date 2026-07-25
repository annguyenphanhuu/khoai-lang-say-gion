#!/usr/bin/env bash
# Long-context probe TẠI NHÀ: context ~14k token chứa 1 needle, hỏi needle ở
# cuối. So baseline vs patched (greedy) — mamba2a đụng block indices nên đây
# là bài test sát thủ nhất. Chạy 32k ctx + max-num-seqs 8 (vừa VRAM 4GB).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/streams"
mkdir -p "$OUT"

python - <<'EOF'
import json
filler = ("Doan van ke chuyen lich su thanh pho, mua mang, thoi tiet va cac le hoi truyen thong. " * 900)
ctx = filler[:52000] + "\nMA BI MAT LA: KHOAI-7391-LANG.\n" + filler[52000:104000]
body = {
  "model": "LFM2.5-1.2B-Instruct", "stream": True, "temperature": 0,
  "max_tokens": 60, "seed": 42,
  "messages": [
    {"role": "system", "content": "Ban la tro ly doc hieu chinh xac."},
    {"role": "user", "content": ctx + "\n\nCau hoi: MA BI MAT trong van ban la gi? Tra loi ngan gon."},
  ],
}
open(r"D:\khoai-lang-say-gion\image-fastsse\streams\body-needle.json", "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
print("needle body written")
EOF

start_server() {
  MSYS_NO_PATHCONV=1 docker run -d --rm --gpus all --name sse-needle \
    -p 8199:8000 -v "D:\\models\\LFM2.5-1.2B-Instruct:/model" --shm-size 2g \
    --entrypoint python3 "$1" -m vllm.entrypoints.openai.api_server \
    --model=/model --served-model-name=LFM2.5-1.2B-Instruct \
    --host=0.0.0.0 --port=8000 \
    --max-model-len=32768 --gpu-memory-utilization=0.75 --quantization=fp8 \
    --enable-prefix-caching --max-num-seqs=8 >/dev/null
  for i in $(seq 1 200); do
    curl -sf http://localhost:8199/health >/dev/null 2>&1 && return 0
    sleep 3
    docker ps -q -f name=sse-needle | grep -q . || { echo "CRASH khi boot $1"; return 1; }
  done
  return 1
}

for pair in "vllm/vllm-openai:v0.22.1-cu129 base" "khoai-fastsse:cu129-test2 patch"; do
  set -- $pair
  echo "== $2 =="
  start_server "$1" || exit 1
  curl -sN --max-time 300 http://localhost:8199/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary "@$OUT/body-needle.json" > "$OUT/$2-needle.sse"
  docker stop sse-needle >/dev/null
done

norm() { sed -E 's/"id":"chatcmpl-[^"]*"/"id":"ID"/g; s/"created":[0-9]+/"created":0/g' "$1"; }
echo "== needle content =="
for p in base patch; do
  printf "%s: " "$p"
  grep -o '"content":"[^"]*"' "$OUT/$p-needle.sse" | sed 's/"content":"//;s/"$//' | tr -d '\n' | head -c 200
  echo
done
if diff <(norm "$OUT/base-needle.sse") <(norm "$OUT/patch-needle.sse") >/dev/null; then
  echo "NEEDLE-32K: PASS(byte-identical)"
else
  python "$DIR/sem_compare.py" "$OUT/base-needle.sse" "$OUT/patch-needle.sse" && echo "NEEDLE-32K: PASS(sem)" || echo "NEEDLE-32K: FAIL"
fi
grep -q "KHOAI-7391-LANG" <(grep -o '"content":"[^"]*"' "$OUT/patch-needle.sse") && echo "NEEDLE FOUND in patched answer" || echo "needle not found (kiem tra chat luong model local)"
