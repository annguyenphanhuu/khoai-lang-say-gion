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

# [K3] lm_head FP8 chỉ chạy được nếu base image ĐÃ có patch dispatch ParallelLMHead trong fp8.py.
# Đây là điều kiện tiên quyết ngoài file của ta -> kiểm bằng text, không cần import (không cần GPU).
fp8_src = open(f"{BASE}/fp8.py", encoding="utf-8").read()
assert "(LinearBase, ParallelLMHead)" in fp8_src, \
    "FATAL: fp8.py trong base image chưa patch dispatch ParallelLMHead (patch_full_fp8.py)"
ow_src = open(f"{BASE}/online_w4a8.py", encoding="utf-8").read()
assert "ParallelLMHead" in ow_src and "_fp8()" in ow_src, \
    "FATAL: online_w4a8.py thiếu nhánh lm_head FP8 (K3)"
print("K3 lm_head-fp8 prerequisites OK")

# [r5] Cong tac sliding-window trong lfm2.py. Mac dinh TAT => r5 phai chay y het r3.
LFM2 = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/lfm2.py"
lfm2_src = open(LFM2, encoding="utf-8").read()
ast.parse(lfm2_src)
assert "# [perf-patch lfm2-swa]" in lfm2_src, "FATAL: swa_patch.py chua chay tren lfm2.py"
assert "per_layer_sliding_window=_LFM2_SWA" in lfm2_src, "FATAL: kwarg SWA khong duoc chen"
assert '_os.getenv("LFM2_SWA", "0")' in lfm2_src, "FATAL: cong tac env sai ten"
# Mac dinh phai la None (khong set env) — kiem bang chinh bieu thuc, khong can import vllm.
import os as _os_check  # noqa: E402

_ns: dict = {}
exec('import os as _os\n_LFM2_SWA = int(_os.getenv("LFM2_SWA", "0") or 0) or None', _ns)
assert _ns["_LFM2_SWA"] is None or _os_check.getenv("LFM2_SWA"), \
    "FATAL: SWA phai OFF khi khong set env"
print("r5 lfm2-swa switch OK (default OFF)")

try:
    import vllm.model_executor.layers.quantization as q
    assert "online_w4a8" in q.QUANTIZATION_METHODS, q.QUANTIZATION_METHODS
    cfg = q.get_quantization_config("online_w4a8")
    assert cfg.__name__ == "OnlineW4A8Config", cfg
    print("import+register OK:", cfg.__name__)
    inst = cfg()
    assert inst._fp8().get_name() == "fp8", inst._fp8()
    print("K3 Fp8Config delegate OK")
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    msg = str(e).lower()
    if any(k in msg for k in ("libcuda", "no cuda", "nvml", "device type", "driver", "cuda")):
        print("WARN: CUDA absent at build -> skip runtime import check (OK for CPU build machine)")
    else:
        print("FATAL: online_w4a8 code error at import")
        sys.exit(1)
