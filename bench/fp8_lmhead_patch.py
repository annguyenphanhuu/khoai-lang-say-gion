"""Add the ParallelLMHead dispatch that the competition base image has (patch_full_fp8.py)
but stock vllm==0.25.1 lacks. Without it K3's online_w4a8 asserts at boot.

Inserts a branch into Fp8Config.get_quant_method returning the same online FP8 linear
method used for LinearBase (is_checkpoint_fp8_serialized=False path).
"""
import re
from pathlib import Path

import vllm

P = Path(vllm.__file__).parent / "model_executor/layers/quantization/fp8.py"
MARKER = "# [perf-patch fp8-lmhead]"
src = P.read_text(encoding="utf-8")
if MARKER in src:
    print("already patched")
    raise SystemExit(0)

ANCHOR = "        elif isinstance(layer, RoutedExperts):"
assert src.count(ANCHOR) == 1, "anchor not unique: %d" % src.count(ANCHOR)

BRANCH = (
    "        " + MARKER + " lm_head -> FP8 (mirrors competition base image)\n"
    "        elif type(layer).__name__ == \"ParallelLMHead\":\n"
    "            from vllm.model_executor.layers.quantization.online.fp8 import (\n"
    "                Fp8PerTensorOnlineLinearMethod,\n"
    "            )\n"
    "\n"
    "            if self.is_checkpoint_fp8_serialized:\n"
    "                return Fp8LinearMethod(self)\n"
    "            m = Fp8PerTensorOnlineLinearMethod()\n"
    "            m.marlin_input_dtype = get_marlin_input_dtype(prefix)\n"
    "            return m\n"
)

src = src.replace(ANCHOR, BRANCH + ANCHOR, 1)
P.write_text(src, encoding="utf-8")

import importlib
import vllm.model_executor.layers.quantization.fp8 as f
importlib.reload(f)
import inspect
assert "ParallelLMHead" in inspect.getsource(f.Fp8Config.get_quant_method)
print("fp8 lm_head dispatch patched OK")
