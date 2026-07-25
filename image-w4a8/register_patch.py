"""Append registration of online_w4a8 to vLLM quantization __init__.py (chạy lúc BUILD)."""
from pathlib import Path

INIT = Path("/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/layers/quantization/__init__.py")
MARKER = "# [perf-patch online-w4a8]"

src = INIT.read_text(encoding="utf-8")
assert MARKER not in src, "already patched"
src += (
    "\n\n" + MARKER + " register custom online W4A8\n"
    "from vllm.model_executor.layers.quantization.online_w4a8 import "
    "OnlineW4A8Config as _OW4  # noqa: E402\n"
    'register_quantization_config("online_w4a8")(_OW4)\n'
)
INIT.write_text(src, encoding="utf-8")
assert MARKER in INIT.read_text(encoding="utf-8")
print("registration appended OK")
