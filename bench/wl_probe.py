"""Replay the EXACT grading workload shape against a local vLLM and decompose TTFT.

spec: shared_system_prefix 1000 | per_conversation_prefix 1000 | 6 turns of
      (user 150 new tokens -> 300 output tokens), context append-only.

Prints per-turn TTFT / TBT and the engine's own prefix-cache counters, so we can
see whether turns 2..6 actually skip prefill.
"""
import json, os, random, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
NCONV = int(os.environ.get("NCONV", "3"))
OUT = int(os.environ.get("OUT", "60"))          # output tokens per turn (shorten for slow rig)
TURNS = int(os.environ.get("TURNS", "6"))

rnd = random.Random(42)
WORDS = ("the quick brown fox jumps over a lazy dog while parsing tokens and serving "
         "requests through an inference engine on limited hardware resources with "
         "careful attention to latency budgets and throughput characteristics observed "
         "during production workloads in a datacenter environment near the edge").split()


def words_for(ntok):
    # ~1.35 tokens/word for this tokenizer -> good enough; exactness not needed
    return " ".join(rnd.choice(WORDS) for _ in range(int(ntok / 1.35)))


def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            txt = r.read().decode()
    except Exception as e:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                  "vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
            if line.startswith(k):
                try:
                    out[k] = float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return out


def chat(messages, max_tokens):
    body = json.dumps({"model": "LFM2.5-1.2B-Instruct", "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                stamps.append(time.perf_counter())
                text.append(ch[0]["delta"]["content"])
    ttft = (stamps[0] - t0) * 1e3 if stamps else float("nan")
    gaps = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    gaps_sorted = sorted(gaps)
    tbt = gaps_sorted[len(gaps_sorted) // 2] if gaps else float("nan")
    return ttft, tbt, len(stamps), "".join(text), usage


SYS = words_for(1000)
print(f"# nconv={NCONV} turns={TURNS} out={OUT}", flush=True)
print(f"{'conv':>4} {'turn':>4} {'ttft_ms':>9} {'tbt_ms':>8} {'ntok':>5} "
      f"{'prompt_tok':>10} {'cached':>7} {'pfx_q':>8} {'pfx_hit':>8} {'hit%':>6}", flush=True)

rows = []
for c in range(NCONV):
    convpfx = words_for(1000)
    msgs = [{"role": "system", "content": SYS}]
    for t in range(1, TURNS + 1):
        user = words_for(150)
        if t == 1:
            user = convpfx + "\n\n" + user
        msgs.append({"role": "user", "content": user})
        m0 = metrics()
        ttft, tbt, n, txt, usage = chat(msgs, OUT)
        m1 = metrics()
        dq = m1.get("vllm:prefix_cache_queries_total", 0) - m0.get("vllm:prefix_cache_queries_total", 0)
        dh = m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0)
        ptok = (usage or {}).get("prompt_tokens", 0)
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        hitpct = (100.0 * dh / dq) if dq else float("nan")
        print(f"{c:>4} {t:>4} {ttft:>9.1f} {tbt:>8.2f} {n:>5} {ptok:>10} {cached:>7} "
              f"{dq:>8.0f} {dh:>8.0f} {hitpct:>6.1f}", flush=True)
        rows.append((c, t, ttft, tbt, ptok, dq, dh))
        msgs.append({"role": "assistant", "content": txt})

# summary
t1 = [r[2] for r in rows if r[1] == 1]
tn = [r[2] for r in rows if r[1] > 1]
print(f"\nTTFT turn1  mean={sum(t1)/len(t1):.1f} ms   (n={len(t1)})")
print(f"TTFT turn2+ mean={sum(tn)/len(tn):.1f} ms   (n={len(tn)})")
print(f"ratio turn2+/turn1 = {(sum(tn)/len(tn))/(sum(t1)/len(t1)):.3f}")
