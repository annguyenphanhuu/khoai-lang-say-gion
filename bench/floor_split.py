"""Split the context-INDEPENDENT TTFT floor into transport / frontend / engine."""
import json, os, statistics, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
R = int(os.environ.get("R", "40"))


def med(f, n=R):
    xs = []
    for _ in range(n):
        a = time.perf_counter()
        f()
        xs.append((time.perf_counter() - a) * 1e3)
    return statistics.median(sorted(xs))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        r.read()


def stream(path, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                d = json.loads(line[6:])
                c = d.get("choices") or []
                if first is None and c and (c[0].get("delta", {}).get("content")
                                            or c[0].get("text")):
                    first = time.perf_counter()
    return (first - t0) * 1e3


P = "hello"
print(f"GET /health                    {med(lambda: get('/health')):8.2f} ms   (HTTP transport floor)")
print(f"GET /v1/models                 {med(lambda: get('/v1/models')):8.2f} ms   (+ routing/serialize)")

xs = [stream("/v1/completions", {"model": "LFM2.5-1.2B-Instruct", "prompt": P,
                                 "max_tokens": 2, "temperature": 0, "stream": True})
      for _ in range(R)]
print(f"TTFT /v1/completions  (5 tok)  {statistics.median(sorted(xs)):8.2f} ms   (no chat template)")

xs = [stream("/v1/chat/completions", {"model": "LFM2.5-1.2B-Instruct",
                                      "messages": [{"role": "user", "content": P}],
                                      "max_tokens": 2, "temperature": 0, "stream": True})
      for _ in range(R)]
print(f"TTFT /v1/chat/completions      {statistics.median(sorted(xs)):8.2f} ms   (full frontend)")

xs = [stream("/v1/chat/completions", {"model": "LFM2.5-1.2B-Instruct",
                                      "messages": [{"role": "user", "content": P}],
                                      "max_tokens": 2, "temperature": 0, "stream": True,
                                      "logprobs": False, "n": 1})
      for _ in range(R)]
print(f"  same, explicit n/logprobs    {statistics.median(sorted(xs)):8.2f} ms")
