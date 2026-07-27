"""Steady B~27 decode load for kernel profiling (Track A).

27 concurrent streams, shared ~3300-token prefix (mirrors portal: 1000 shared sys +
1000 conv prefix + turn text, prefix-cached => prefill cheap => profile is decode-dominant),
max_tokens=300, temperature=0. Runs until DURATION seconds elapse, then stops.

Usage: BASE=http://localhost:8000 DUR=90 CONC=27 python3 load27.py
"""
import json, os, random, threading, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "LFM2.5-1.2B-Instruct")
CONC = int(os.environ.get("CONC", "27"))
DUR = float(os.environ.get("DUR", "90"))
OUT = int(os.environ.get("OUT", "300"))
PFX_TOK = int(os.environ.get("PFX_TOK", "3200"))
SUSTAIN = os.environ.get("SUSTAIN", "0") == "1"

WORDS = ("the quick brown fox jumps over a lazy dog while parsing tokens and serving "
         "requests through an inference engine on limited hardware resources with "
         "careful attention to latency budgets and throughput characteristics observed "
         "during production workloads in a datacenter environment near the edge").split()


def words_for(ntok, rnd):
    return " ".join(rnd.choice(WORDS) for _ in range(int(ntok / 1.35)))


SHARED = words_for(PFX_TOK, random.Random(1000))

stop = threading.Event()
lock = threading.Lock()
stats = {"req": 0, "tok": 0, "err": 0}


def one(tail):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SHARED},
                     {"role": "user", "content": tail}],
        "max_tokens": OUT, "temperature": 0, "stream": True,
        # portal pins output_tokens_per_turn=300; without this the model emits EOS after
        # ~2 tokens on synthetic prompts and the window becomes prefill-dominated.
        "min_tokens": OUT, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            # ignore_eos makes the model emit past EOS, where the detokenized content is
            # often "" -- count SSE chunks, not non-empty text, or every request reads as 0.
            d = json.loads(p)
            if d.get("choices"):
                n += 1
    return n


def worker(wid):
    rnd = random.Random(5000 + wid)
    i = 0
    while not stop.is_set():
        i += 1
        try:
            n = one("Question %d-%d: %s" % (wid, i, words_for(120, rnd)))
            with lock:
                stats["req"] += 1
                stats["tok"] += n
        except Exception:
            with lock:
                stats["err"] += 1
            time.sleep(0.5)
        # SUSTAIN: one very long generation per worker, never restarted. The capture window
        # then contains ONLY decode steps at exactly CONC batch -- no prefill contamination
        # and no batch-size churn from arrivals/departures.
        if SUSTAIN:
            return


def sampler(samples):
    while not stop.is_set():
        try:
            with urllib.request.urlopen(BASE + "/metrics", timeout=2) as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("vllm:num_requests_running"):
                        samples.append(float(line.rsplit(" ", 1)[1]))
                        break
        except Exception:
            pass
        time.sleep(0.1)


def main():
    samples = []
    ths = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(CONC)]
    sm = threading.Thread(target=sampler, args=(samples,), daemon=True)
    for t in ths:
        t.start()
    sm.start()
    t0 = time.time()
    while time.time() - t0 < DUR:
        time.sleep(2)
        with lock:
            print("t=%5.1fs req=%d tok=%d err=%d batch=%s" % (
                time.time() - t0, stats["req"], stats["tok"], stats["err"],
                ("%.0f" % samples[-1]) if samples else "?"), flush=True)
    stop.set()
    time.sleep(1)
    if samples:
        s = sorted(samples)
        print("BATCH p50=%.0f p90=%.0f mean=%.1f max=%.0f n=%d" % (
            s[len(s) // 2], s[int(0.9 * len(s))], sum(s) / len(s), s[-1], len(s)))
    print("TOTAL req=%d tok=%d err=%d" % (stats["req"], stats["tok"], stats["err"]))


if __name__ == "__main__":
    main()
