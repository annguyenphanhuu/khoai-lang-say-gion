"""Faithful CONCURRENT ERS harness — reproduce the grading workload shape and
score it with the official ERS formula, so we optimize the REAL objective
(70 concurrent Poisson conversations) instead of the C=1 proxies that every
other bench script here measures.

Workload (grading-workload-spec.json):
  num_conversations=70, user_turns_per_conversation=6, total=420,
  shared_system_prefix=1000 tok (identical across ALL convs -> prefix-cache),
  per_conversation_prefix=1000 tok (identical across that conv's 6 turns),
  new_user_tokens_per_turn=150, output_tokens_per_turn_pinned=300,
  arrival = Poisson (seed 42).

Score (KB §2): s(x) = clamp((C-x)/(C-F), 0, 1)**2
  TTFT: F=10ms,  C=400ms
  TPOT: F=1ms,   C=10ms
  ERS  = mean_over_requests( 100 * (0.5*s_ttft + 0.5*s_tpot) )

The BTC Poisson *rate* is unknown, so ARRIVAL_RATE (conversations/sec) is a knob.
Sweep it to bracket the portal's concurrency; RELATIVE ranking of server configs
under the same load is what transfers to the portal. A /metrics sampler records
the real running-request (decode batch) histogram — the number this repo has
been guessing at ("batch 3-15") for weeks.

Usage:
  BASE=http://localhost:8000 NCONV=70 RATE=6 OUT=300 python3 bench/ers_harness.py
Env:
  NCONV(70) TURNS(6) OUT(300) RATE(conv/sec, default 6.0) SEED(42)
  SYS_TOK(1000) CONVPFX_TOK(1000) USER_TOK(150) BASE MODEL
"""
import json, os, random, threading, time, urllib.request

BASE   = os.environ.get("BASE", "http://localhost:8000")
MODEL  = os.environ.get("MODEL", "LFM2.5-1.2B-Instruct")
NCONV  = int(os.environ.get("NCONV", "70"))
TURNS  = int(os.environ.get("TURNS", "6"))
OUT    = int(os.environ.get("OUT", "300"))
RATE   = float(os.environ.get("RATE", "6.0"))      # conversations arrive/sec (Poisson)
SEED   = int(os.environ.get("SEED", "42"))
SYS_TOK     = int(os.environ.get("SYS_TOK", "1000"))
CONVPFX_TOK = int(os.environ.get("CONVPFX_TOK", "1000"))
USER_TOK    = int(os.environ.get("USER_TOK", "150"))

# ---- score ----
def sfun(x, F, C):
    v = (C - x) / (C - F)
    v = 0.0 if v < 0 else (1.0 if v > 1 else v)
    return v * v

def score_req(ttft_ms, tpot_ms):
    s_ttft = sfun(ttft_ms, 10.0, 400.0)
    s_tpot = sfun(tpot_ms, 1.0, 10.0)
    return 100.0 * (0.5 * s_ttft + 0.5 * s_tpot), s_ttft, s_tpot

# ---- token-ish text (same heuristic as wl_probe: ~1.35 tok/word) ----
WORDS = ("the quick brown fox jumps over a lazy dog while parsing tokens and serving "
         "requests through an inference engine on limited hardware resources with "
         "careful attention to latency budgets and throughput characteristics observed "
         "during production workloads in a datacenter environment near the edge").split()
def words_for(ntok, rnd):
    return " ".join(rnd.choice(WORDS) for _ in range(int(ntok / 1.35)))

SYS = words_for(SYS_TOK, random.Random(1000))   # SHARED across all conversations

def chat(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); stamps = []; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                stamps.append(time.perf_counter())
                text.append(ch[0]["delta"]["content"])
    ttft = (stamps[0] - t0) * 1e3 if stamps else float("nan")
    # TPOT = mean inter-token rate over the decode phase. Median-of-client-gaps is
    # WRONG under load: SSE/network buffering collapses most gaps to ~0 and inflates
    # s_tpot. Mean-rate = (last-first)/(n-1) recovers the true effective per-token time.
    tpot = ((stamps[-1] - stamps[0]) * 1e3 / (len(stamps) - 1)) if len(stamps) > 1 else float("nan")
    return ttft, tpot, len(stamps), "".join(text)

results = []; rlock = threading.Lock()
def run_conv(cid, start_delay):
    time.sleep(start_delay)
    rnd = random.Random(10_000 + cid)
    convpfx = words_for(CONVPFX_TOK, rnd)
    msgs = [{"role": "system", "content": SYS}]
    for t in range(1, TURNS + 1):
        user = words_for(USER_TOK, rnd)
        if t == 1:
            user = convpfx + "\n\n" + user
        msgs.append({"role": "user", "content": user})
        try:
            ttft, tpot, n, txt = chat(msgs, OUT)
            ok = n > 0
        except Exception as e:
            ttft, tpot, n, txt, ok = float("nan"), float("nan"), 0, "", False
        with rlock:
            results.append((cid, t, ttft, tpot, n, ok))
        msgs.append({"role": "assistant", "content": txt})

# ---- /metrics sampler: real running-batch histogram ----
stop_flag = threading.Event(); batch_samples = []
def sampler():
    while not stop_flag.is_set():
        try:
            with urllib.request.urlopen(BASE + "/metrics", timeout=2) as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("vllm:num_requests_running"):
                        batch_samples.append(float(line.rsplit(" ", 1)[1])); break
        except Exception:
            pass
        time.sleep(0.05)

def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); return xs[min(len(xs)-1, int(p/100*len(xs)))]

import re
def fetch_metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            return r.read().decode()
    except Exception:
        return ""

def snap(txt, metric):
    """Snapshot a histogram: {le: cumulative_count}, plus _sum, _count."""
    buckets = {}; total = 0.0; msum = 0.0
    for line in txt.splitlines():
        if line.startswith(metric + "_bucket"):
            m = re.search(r'le="([^"]+)"', line)
            le = float("inf") if m.group(1) == "+Inf" else float(m.group(1))
            buckets[le] = float(line.rsplit(" ", 1)[1])
        elif line.startswith(metric + "_count"):
            total = float(line.rsplit(" ", 1)[1])
        elif line.startswith(metric + "_sum"):
            msum = float(line.rsplit(" ", 1)[1])
    return buckets, total, msum

def hist_expect_s_diff(txt0, txt1, metric, F_ms, C_ms):
    """E[s] over the DELTA histogram (txt1 - txt0) so we never need to restart the
    server between configs. Portal-faithful, no client artifacts. Bucket midpoints."""
    b0, c0, s0 = snap(txt0, metric)
    b1, c1, s1 = snap(txt1, metric)
    les = sorted(b1.keys())
    prev_c = 0.0; prev_le = 0.0; num = 0.0; den = 0.0
    for le in les:
        cum = b1.get(le, 0.0) - b0.get(le, 0.0)
        cnt = cum - prev_c
        if cnt > 0:
            mid = prev_le if le == float("inf") else (prev_le + le) / 2.0
            num += cnt * sfun(mid * 1e3, F_ms, C_ms); den += cnt
        prev_c = cum
        if le != float("inf"): prev_le = le
    n = c1 - c0
    mean_ms = ((s1 - s0) / n * 1e3) if n else float("nan")
    return (num / den if den else float("nan")), mean_ms, n

def hist_ers(txt0, txt1):
    # TTFT histogram is fine-grained -> E[s] over buckets is accurate.
    es_ttft, m_ttft, n1 = hist_expect_s_diff(txt0, txt1, "vllm:time_to_first_token_seconds", 10.0, 400.0)
    # TPOT histogram's first bucket is le=10ms -> ALL sub-10ms requests collapse into it,
    # making E[s] a useless constant (~0.31). The _sum/_count MEAN is accurate, and since
    # per-token latency is tightly clustered under steady batching (verified: TTFT E[s]≈s(mean)),
    # score TPOT via s(mean). F=1ms, C=10ms.
    _, m_tpot, n2 = hist_expect_s_diff(txt0, txt1, "vllm:request_time_per_output_token_seconds", 1.0, 10.0)
    es_tpot = sfun(m_tpot, 1.0, 10.0)
    ers = 100.0 * (0.5 * es_ttft + 0.5 * es_tpot)
    return ers, es_ttft, es_tpot, m_ttft, m_tpot

def main():
    rnd = random.Random(SEED)
    # Poisson arrival: exponential inter-arrival, mean 1/RATE
    delays = []; t = 0.0
    for _ in range(NCONV):
        t += rnd.expovariate(RATE); delays.append(t)
    txt0 = fetch_metrics()   # histogram snapshot BEFORE load (for delta scoring)
    smp = threading.Thread(target=sampler, daemon=True); smp.start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=run_conv, args=(c, delays[c])) for c in range(NCONV)]
    for th in threads: th.start()
    for th in threads: th.join()
    wall = time.perf_counter() - t0
    stop_flag.set(); time.sleep(0.1)

    ok = [r for r in results if r[5]]
    failed = len(results) - len(ok)
    ers_parts = []
    for (_, _, ttft, tpot, _, _) in ok:
        e, _, _ = score_req(ttft, tpot); ers_parts.append(e)
    for _ in range(failed):
        ers_parts.append(0.0)   # failed request -> 0
    ers = sum(ers_parts) / len(ers_parts) if ers_parts else float("nan")

    ttfts = [r[2] for r in ok]; tpots = [r[3] for r in ok]
    t1 = [r[2] for r in ok if r[1] == 1]; tn = [r[2] for r in ok if r[1] > 1]
    print("="*64)
    print(f"CONFIG: nconv={NCONV} turns={TURNS} out={OUT} rate={RATE}/s wall={wall:.1f}s")
    print(f"ERS = {ers:.2f}   (requests={len(results)} failed={failed})")
    print(f"TTFT  p50={pct(ttfts,50):.1f}  p90={pct(ttfts,90):.1f}  ms   "
          f"(turn1 p50={pct(t1,50):.1f}  turn2+ p50={pct(tn,50):.1f})")
    print(f"TPOT  p50={pct(tpots,50):.2f}  p90={pct(tpots,90):.2f}  ms  (client mean-rate)")
    txt1 = fetch_metrics()
    h_ers, es_ttft, es_tpot, m_ttft, m_tpot = hist_ers(txt0, txt1)
    print(f">>> ERS_HIST = {h_ers:.2f}  <<<  PRIMARY (portal-faithful, server-side histograms)")
    print(f"    s_ttft={es_ttft:.3f} (mean TTFT {m_ttft:.1f}ms) | s_tpot={es_tpot:.3f} (mean TPOT {m_tpot:.2f}ms)")
    if batch_samples:
        print(f"DECODE BATCH (num_requests_running): "
              f"max={max(batch_samples):.0f}  p50={pct(batch_samples,50):.0f}  "
              f"p90={pct(batch_samples,90):.0f}  mean={sum(batch_samples)/len(batch_samples):.1f}")
    print("="*64)

if __name__ == "__main__":
    main()
