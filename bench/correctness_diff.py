"""ROADMAP gate — CORRECTNESS DIFF (temp=0 phải GIỐNG HỆT bản gốc).

Cổng BẮT BUỘC cho ② (--async-scheduling) và ④ (speculative decoding): một tối ưu chỉ HỢP LỆ
nếu output ở temp=0 giống hệt bản best. Lệch 1 token ⇒ tối ưu đó ĐỔI kết quả ⇒ trượt Accuracy
Gate + có thể failed request ⇒ KHÔNG submit. (async/full-graph/spec-decode ĐÚNG phải lossless.)

Chạy 1 GPU: dùng 2 pha (capture rồi diff), không cần 2 endpoint cùng lúc.

  # Pha 1 — chụp reference từ image BEST (64.67) đang chạy ở :8000
  MODE=capture BASE=http://localhost:8000 OUT=bench/golden.json python3 bench/correctness_diff.py
  # ... tắt image best, bật image ứng viên (cùng port) ...
  # Pha 2 — so ứng viên với golden
  MODE=diff BASE=http://localhost:8000 REF=bench/golden.json python3 bench/correctness_diff.py

Hoặc 2 endpoint cùng lúc (2 GPU / 2 port): MODE=live BASE=cand REF_URL=ref.
Exit code 0 = PASS (an toàn submit); 1 = FAIL (KHÔNG submit).
"""
import json, os, sys, urllib.request

MODE = os.environ.get("MODE", "diff")            # capture | diff | live
BASE = os.environ.get("BASE", "http://localhost:8000")
REF_URL = os.environ.get("REF_URL", "http://localhost:8001")
REF_FILE = os.environ.get("REF", "bench/golden.json")
OUT = os.environ.get("OUT", "bench/golden.json")
MODEL = "LFM2.5-1.2B-Instruct"
MAXTOK = int(os.environ.get("MAXTOK", "300"))    # đúng output_tokens_per_turn_pinned
N = int(os.environ.get("N", "12"))

SYS = ("You are a meticulous technical assistant for a large distributed database platform. "
       "Answer using only the reference material supplied. Cite section numbers precisely. ") * 12
PROMPTS = [
    "Summarise the write path step by step, citing every parameter you mention.",
    "Restate the reference document as a numbered operational checklist.",
    "Compare read amplification and compaction scheduling; which bounds tail latency and why?",
    "Explain quorum-based replication with replication factor five in full detail.",
    "Walk through backpressure when the memtable queue is exceeded, referencing all sections.",
    "Describe the failure modes section by section, thoroughly and exhaustively.",
]


def gen(base, prompt):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": MAXTOK, "stream": False,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def first_diff(a, b):
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    return k, repr(a[max(0, k - 20):k + 20]), repr(b[max(0, k - 20):k + 20])


if MODE == "capture":
    ref = [gen(BASE, PROMPTS[i % len(PROMPTS)]) for i in range(N)]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False)
    print(f"captured {N} reference outputs -> {OUT}")
    sys.exit(0)

# load reference
if MODE == "live":
    ref = None  # sinh trực tiếp từ REF_URL bên dưới
else:
    with open(REF_FILE, encoding="utf-8") as f:
        ref = json.load(f)

fails = 0
for i in range(N):
    p = PROMPTS[i % len(PROMPTS)]
    a = ref[i] if ref is not None else gen(REF_URL, p)
    b = gen(BASE, p)
    if a == b:
        print(f"[{i:2}] OK   ({len(a)} ký tự giống hệt)")
    else:
        fails += 1
        k, ra, rb = first_diff(a, b)
        print(f"[{i:2}] DIFF @char {k}\n      ref ={ra}\n      cand={rb}")

print(f"\n{'PASS — an toàn submit' if fails == 0 else f'FAIL — {fails}/{N} lệch ⇒ KHÔNG submit (đổi output)'}")
sys.exit(1 if fails else 0)
