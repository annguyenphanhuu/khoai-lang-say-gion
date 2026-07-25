"""Validate lúc BUILD (không cần GPU): syntax + import + đăng ký online_w4a8.

Tolerate máy build không có CUDA (import vllm có thể fail ở libcuda) — chỉ hard-fail nếu lỗi
là CODE của ta (NameError/ImportError symbol/AttributeError), không phải thiếu driver.
"""
import ast
import sys
import traceback

BASE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization"
for f in (f"{BASE}/online_w4a8.py", f"{BASE}/__init__.py"):
    ast.parse(open(f, encoding="utf-8").read())
print("syntax OK")

try:
    import vllm.model_executor.layers.quantization as q
    assert "online_w4a8" in q.QUANTIZATION_METHODS, q.QUANTIZATION_METHODS
    cfg = q.get_quantization_config("online_w4a8")
    assert cfg.__name__ == "OnlineW4A8Config", cfg
    print("import+register OK:", cfg.__name__)
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    msg = str(e).lower()
    if any(k in msg for k in ("libcuda", "no cuda", "nvml", "device type", "driver", "cuda")):
        print("WARN: CUDA absent at build -> skip runtime import check (OK for CPU build machine)")
    else:
        print("FATAL: online_w4a8 code error at import")
        sys.exit(1)
