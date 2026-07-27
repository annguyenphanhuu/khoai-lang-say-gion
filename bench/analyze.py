"""Track A analysis: what does one decode step at B=27 actually consist of?

Answers the four questions in PLAN-2026-07-27.md from an nsys sqlite:
  1. kernel launches per decode step
  2. top kernels by total GPU time
  3. GEMM / attention / elementwise-norm-act split
  4. total inter-kernel gap (launch-bound vs execution-bound)

Step segmentation: lm_head runs exactly once per decode step, and it is the only GEMM whose
N dimension is the 65536-row vocab, so its launch count == step count. We cross-check that
against the per-layer kv-cache write kernel (once per layer per step).

Usage: python3 analyze.py /workspace/prof27/step27.sqlite
"""
import re
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "/workspace/prof27/step27.sqlite"
con = sqlite3.connect(DB)
cur = con.cursor()

rows = cur.execute("""
    SELECT k.start, k.end, s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    ORDER BY k.start
""").fetchall()
print("kernel launches in report: %d" % len(rows))
if not rows:
    raise SystemExit("no kernels")


def short(n):
    """Collapse the giant cutlass template names to something readable."""
    if "FlashAttnFwdSm90" in n:
        m = re.search(r"cute::C<\(int\)(\d+)>, cute::C<\(int\)(\d+)>, cute::C<\(int\)(\d+)>>, \(int\)64", n)
        return "flash_attn_fwd_sm90[tile %s]" % (",".join(m.groups()) if m else "?")
    if "FlashAttnFwdCombine" in n:
        return "flash_attn_combine"
    if "cutlass_3x_gemm_sm90_fp8" in n:
        m = re.search(r"cute::C<\(int\)(\d+)>, cute::C<\(int\)(\d+)>, cute::C<\(int\)(\d+)>>", n)
        return "gemm_fp8_sm90[tile %s]" % (",".join(m.groups()) if m else "?")
    if "MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput" in n:
        m = re.search(r"KernelTmaWarpSpecializedCooperative>, cute::tuple<cute::C<\(int\)(\d+)>, "
                      r"cute::C<\(int\)(\d+)>, cute::C<\(int\)(\d+)>>", n)
        return "gemm_w4a8_mixed[tile %s]" % (",".join(m.groups()) if m else "?")
    if n.startswith("nvjet_"):
        return "gemm_cublas_" + n
    if "elementwise_kernel" in n:
        m = re.search(r"native::(\w+Functor|\w+_kernel_cuda)", n)
        return "aten_elementwise[%s]" % (m.group(1) if m else "?")
    if len(n) > 70:
        return n[:70]
    return n


def category(n):
    if "FlashAttnFwd" in n or "flash" in n.lower() or "attn" in n.lower():
        return "attention"
    if ("gemm" in n.lower() or "Gemm" in n or n.startswith("nvjet_")
            or "cutlass_3x" in n or "MixedInput" in n):
        return "gemm"
    if "causal_conv1d" in n or "short_conv" in n:
        return "conv"
    if "reshape_and_cache" in n or "cache" in n.lower():
        return "kv_write"
    if ("elementwise" in n or "triton_" in n or "norm" in n.lower()
            or "silu" in n.lower() or "copy" in n.lower() or "vectorized" in n):
        return "elementwise/norm/act"
    return "other"


# --- step segmentation ---------------------------------------------------
lmhead = [r for r in rows if "cutlass_3x_gemm_sm90_fp8" in r[2]]
kvw = [r for r in rows if "reshape_and_cache" in r[2]]
print("lm_head fp8 gemm launches : %d" % len(lmhead))
print("reshape_and_cache launches: %d" % len(kvw))

steps = len(lmhead)
if steps < 5:
    raise SystemExit("too few steps detected (%d) - is the window pure decode?" % steps)

t0, t1 = lmhead[0][0], lmhead[-1][0]
span_ms = (t1 - t0) / 1e6
nstep = steps - 1
print("\nwindow between first/last lm_head: %.1f ms over %d steps" % (span_ms, nstep))
print("wall step time            : %.3f ms   (rig regime, NOT portal - KB S5)" % (span_ms / nstep))
print("kv-write per step         : %.1f  (== layer count with kv cache)" % (len(kvw) / steps))

# kernels strictly inside the segmented steps
inwin = [r for r in rows if t0 <= r[0] <= t1]
print("kernels per decode step   : %.1f" % (len(inwin) / nstep))

# --- busy vs gap ---------------------------------------------------------
busy = 0
gap = 0
prev_end = None
for s, e, _ in inwin:
    if prev_end is not None and s > prev_end:
        gap += s - prev_end
    busy += e - s
    prev_end = max(prev_end, e) if prev_end is not None else e
total = t1 - t0
print("\nGPU busy   : %7.3f ms/step (%.1f%%)" % (busy / 1e6 / nstep, 100.0 * busy / total))
print("GPU gap    : %7.3f ms/step (%.1f%%)   <-- >20%% means launch-bound" % (
    gap / 1e6 / nstep, 100.0 * gap / total))

# --- categories ----------------------------------------------------------
cat_t = defaultdict(float)
cat_n = defaultdict(int)
for s, e, n in inwin:
    c = category(n)
    cat_t[c] += e - s
    cat_n[c] += 1
print("\n%-22s %8s %10s %12s" % ("category", "%busy", "count/step", "ms/step"))
for c in sorted(cat_t, key=lambda x: -cat_t[x]):
    print("%-22s %7.1f%% %10.1f %12.4f" % (
        c, 100.0 * cat_t[c] / busy, cat_n[c] / nstep, cat_t[c] / 1e6 / nstep))

# --- top kernels ---------------------------------------------------------
k_t = defaultdict(float)
k_n = defaultdict(int)
for s, e, n in inwin:
    k_t[short(n)] += e - s
    k_n[short(n)] += 1
print("\n%-46s %7s %10s %10s" % ("top kernels", "%busy", "n/step", "us/call"))
for k in sorted(k_t, key=lambda x: -k_t[x])[:15]:
    print("%-46s %6.1f%% %10.1f %10.1f" % (
        k[:46], 100.0 * k_t[k] / busy, k_n[k] / nstep, k_t[k] / 1e3 / k_n[k]))
