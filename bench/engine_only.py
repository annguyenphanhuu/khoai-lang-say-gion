"""H2 diagnostic: engine-only decode TPOT (offline LLM, no HTTP/SSE frontend).
Run inside vLLM container with /model mounted. __main__ guard required (V1 spawn)."""
import os, time, statistics
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    mode = os.environ.get("MODE", "mp")
    llm = LLM(model="/model", max_model_len=2048, enforce_eager=True,
              gpu_memory_utilization=0.70, max_num_seqs=1, dtype="bfloat16",
              disable_log_stats=True)
    p = "The quick brown fox jumps over the lazy dog and then runs across the field."

    def gen(n):
        sp = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0)
        t = time.perf_counter()
        llm.generate([p], sp, use_tqdm=False)
        return time.perf_counter() - t

    gen(8); gen(8)  # warmup
    t1 = statistics.median([gen(1) for _ in range(5)])
    tn = statistics.median([gen(129) for _ in range(5)])
    print(f">>> ENGINE-ONLY[{mode}]: T1={t1*1000:.1f}ms T129={tn*1000:.1f}ms "
          f"TPOT={(tn-t1)/128*1000:.2f}ms")
