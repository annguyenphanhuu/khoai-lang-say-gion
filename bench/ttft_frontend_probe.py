"""Decompose the per-request CPU cost that sits IN FRONT of the engine (TTFT path).

Runs entirely on CPU with the real LFM2.5 tokenizer + real chat template.
Reproduces the grading workload shape:
  system prefix 1000 tok (shared) + conv prefix 1000 tok + N*(user 150 + assistant 300)
"""
import json, os, time, random, statistics

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer
import jinja2

MODEL = "/model"
tok = AutoTokenizer.from_pretrained(MODEL)

rnd = random.Random(42)
# Build realistic english-ish text of a target token count.
WORDS = ("the quick brown fox jumps over a lazy dog while parsing tokens and "
         "serving requests through an inference engine on limited hardware "
         "resources with careful attention to latency budgets and throughput "
         "characteristics observed during production workloads").split()

def make_text(target_tokens: int) -> str:
    out = []
    while True:
        out.append(rnd.choice(WORDS))
        if len(out) % 64 == 0:
            n = len(tok(" ".join(out), add_special_tokens=False)["input_ids"])
            if n >= target_tokens:
                return " ".join(out)

SYS = make_text(1000)
CONVPFX = make_text(1000)
USER = [make_text(150) for _ in range(6)]
ASSIST = [make_text(300) for _ in range(6)]

template_src = open(os.path.join(MODEL, "chat_template.jinja"), encoding="utf-8").read()
env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
tmpl = env.from_string(template_src)

def messages_for_turn(t: int):
    # t is 1-based
    msgs = [{"role": "system", "content": SYS}]
    first_user = CONVPFX + "\n\n" + USER[0]
    msgs.append({"role": "user", "content": first_user})
    for i in range(1, t):
        msgs.append({"role": "assistant", "content": ASSIST[i - 1]})
        msgs.append({"role": "user", "content": USER[i]})
    return msgs

REPS = 20
print(f"{'turn':>4} {'ntok':>6} {'render_ms':>10} {'encode_ms':>10} {'json_ms':>8} {'total_ms':>9}")
rows = []
for t in range(1, 7):
    msgs = messages_for_turn(t)
    body = json.dumps({"model": "LFM2.5-1.2B-Instruct", "messages": msgs,
                       "max_tokens": 300, "stream": True})
    # 1) jinja render
    ts = []
    for _ in range(REPS):
        a = time.perf_counter()
        text = tmpl.render(messages=msgs, add_generation_prompt=True,
                           bos_token="<|startoftext|>", tools=None)
        ts.append((time.perf_counter() - a) * 1e3)
    render = statistics.median(ts)
    # 2) tokenize
    ts = []
    for _ in range(REPS):
        a = time.perf_counter()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ts.append((time.perf_counter() - a) * 1e3)
    encode = statistics.median(ts)
    # 3) json parse of the HTTP body
    ts = []
    for _ in range(REPS):
        a = time.perf_counter()
        json.loads(body)
        ts.append((time.perf_counter() - a) * 1e3)
    js = statistics.median(ts)
    rows.append((t, len(ids), render, encode, js))
    print(f"{t:>4} {len(ids):>6} {render:>10.2f} {encode:>10.2f} {js:>8.2f} "
          f"{render+encode+js:>9.2f}")

avg = sum(r[2] + r[3] + r[4] for r in rows) / len(rows)
print(f"\nMEAN frontend CPU per request (render+encode+json) = {avg:.2f} ms")
print(f"MEAN over turns 2-6 only = {sum(r[2]+r[3]+r[4] for r in rows[1:])/5:.2f} ms")
