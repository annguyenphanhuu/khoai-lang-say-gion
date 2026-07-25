"""Isolate the HOST-SIDE portion of TTFT.

A request whose prompt is 100% prefix-cached does ~zero prefill compute, so its
TTFT == HTTP + pydantic + jinja + tokenize + IPC + block-hash + schedule + 1 step
+ detok + SSE.  That is exactly the bucket the portal is stuck in (portal TTFT
54 ms while its GPU prefill is only ~2.5 ms).

Reports the host floor at several context lengths, so we can also read off the
per-1000-token slope (tokenize + block hashing).
"""
import json, os, random, statistics, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
REPS = int(os.environ.get("REPS", "25"))
rnd = random.Random(11)
W = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
     "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee").split()


def txt(ntok):
    return " ".join(rnd.choice(W) for _ in range(max(1, int(ntok / 1.35))))


def one(prompt):
    body = json.dumps({"model": "LFM2.5-1.2B-Instruct",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 2, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, ptok = None, 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                ptok = d["usage"]["prompt_tokens"]
            c = d.get("choices") or []
            if first is None and c and c[0].get("delta", {}).get("content"):
                first = time.perf_counter()
    return ((first - t0) * 1e3 if first else float("nan")), ptok


for _ in range(5):
    one("warm up the server please")

print(f"{'ctx_tok':>8} {'host_ttft_p50':>14} {'p10':>7} {'p90':>7}")
pts = []
for n in (200, 1000, 2000, 3000, 4500):
    p = txt(n)
    one(p); one(p)                      # populate the prefix cache
    xs = sorted(one(p)[0] for _ in range(REPS))
    _, ptok = one(p)
    p50 = statistics.median(xs)
    print(f"{ptok:>8} {p50:>14.2f} {xs[len(xs)//10]:>7.2f} {xs[-1-len(xs)//10]:>7.2f}")
    pts.append((ptok, p50))

n = len(pts)
mx = sum(x for x, _ in pts) / n
my = sum(y for _, y in pts) / n
b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
print(f"\nHOST floor  = {my - b*mx:.2f} ms   (context-independent part)")
print(f"HOST slope  = {b*1000:.2f} ms per 1000 context tokens  (tokenize + block hash)")
