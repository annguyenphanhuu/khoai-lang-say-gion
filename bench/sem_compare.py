# So sánh NGỮ NGHĨA 2 file SSE khi byte-diff lệch do chunk coalescing
# (số token/chunk phụ thuộc timing, không tất định): nối toàn bộ delta content
# theo index, gom token logprobs, finish_reason cuối, usage — phải giống hệt.
import json
import sys


def parse(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            for ch in d.get("choices", []):
                i = ch["index"]
                s = out.setdefault(i, {"content": "", "finish": None, "lp_tokens": [], "role": None})
                delta = ch.get("delta") or {}
                if delta.get("role"):
                    s["role"] = delta["role"]
                if delta.get("content"):
                    s["content"] += delta["content"]
                if ch.get("finish_reason"):
                    s["finish"] = ch["finish_reason"]
                lp = ch.get("logprobs")
                if lp and lp.get("content"):
                    for t in lp["content"]:
                        s["lp_tokens"].append((t["token"], round(t["logprob"], 6)))
            if "usage" in d and d["usage"]:
                out.setdefault("usage", d["usage"])
    return out


a, b = parse(sys.argv[1]), parse(sys.argv[2])
if a == b:
    print("SEM-PASS")
    sys.exit(0)
print("SEM-FAIL")
for k in sorted(set(a) | set(b), key=str):
    if a.get(k) != b.get(k):
        print(" index", k)
        print("  A:", json.dumps(a.get(k), ensure_ascii=False)[:400])
        print("  B:", json.dumps(b.get(k), ensure_ascii=False)[:400])
sys.exit(1)
