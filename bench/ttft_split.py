"""Decompose TTFT from vLLM /metrics histogram deltas.

usage: python3 ttft_split.py m0.txt m1.txt
Prints mean ms per phase over the requests that completed between the snapshots.
"""
import sys

M = ["vllm:time_to_first_token_seconds",
     "vllm:request_queue_time_seconds",
     "vllm:request_prefill_time_seconds",
     "vllm:request_inference_time_seconds",
     "vllm:request_decode_time_seconds",
     "vllm:e2e_request_latency_seconds",
     "vllm:request_time_per_output_token_seconds",
     "vllm:inter_token_latency_seconds",
     "vllm:request_prompt_tokens",
     "vllm:request_prefill_kv_computed_tokens",
     "vllm:request_generation_tokens",
     "vllm:iteration_tokens_total"]


def snap(path):
    out = {}
    for line in open(path):
        if not line.startswith("vllm:"):
            continue
        name, _, val = line.rpartition(" ")
        name = name.split("{", 1)[0]
        for m in M:
            if name == m + "_sum":
                out[(m, "sum")] = float(val)
            elif name == m + "_count":
                out[(m, "cnt")] = float(val)
    return out


a, b = snap(sys.argv[1]), snap(sys.argv[2])
print(f"{'metric':<46}{'n':>8}{'mean':>12}")
res = {}
for m in M:
    ds = b.get((m, "sum"), 0) - a.get((m, "sum"), 0)
    dc = b.get((m, "cnt"), 0) - a.get((m, "cnt"), 0)
    if dc <= 0:
        continue
    mean = ds / dc
    res[m] = mean
    tok = m.endswith("_tokens") or m.endswith("tokens_total")
    unit = f"{mean:10.1f} tok" if tok else f"{mean*1e3:10.2f} ms"
    print(f"{m:<46}{dc:8.0f}{unit:>12}")

t = res.get("vllm:time_to_first_token_seconds")
q = res.get("vllm:request_queue_time_seconds")
p = res.get("vllm:request_prefill_time_seconds")
if t and q is not None and p is not None:
    print("\n--- TTFT split (mean ms) ---")
    print(f"queue     {q*1e3:8.2f}  ({q/t*100:5.1f}%)")
    print(f"prefill   {p*1e3:8.2f}  ({p/t*100:5.1f}%)")
    rest = t - q - p
    print(f"rest      {rest*1e3:8.2f}  ({rest/t*100:5.1f}%)   <- frontend/tokenizer/detok/sched")
    print(f"TTFT      {t*1e3:8.2f}")
