"""Self-test CPU (không cần GPU) — validate phần TOÁN của online W4A8.

Chạy: docker run --rm -v $PWD/image-w4a8:/w --entrypoint python3 <vllm-img> /w/_selftest.py
Chỉ kiểm phần chạy được không-GPU: RTN + pack/unpack round-trip (nếu sai packing thì kernel
CUTLASS sẽ giải mã rác). Phần param-vLLM + kernel CUTLASS cần vLLM-config context + Hopper ⇒
chỉ verify được khi boot thật trên portal.
"""
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.quantization.online_w4a8 import _rtn_int4_g128  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types  # noqa: E402

torch.manual_seed(0)
for out_f, in_f in [(256, 512), (3072, 2048), (2048, 8192), (512, 2048)]:
    W = torch.randn(out_f, in_f, dtype=torch.bfloat16)
    u, scale = _rtn_int4_g128(W)
    assert u.shape == (out_f, in_f) and int(u.min()) >= 0 and int(u.max()) <= 15
    assert scale.shape == (out_f, in_f // 128)
    packed = pack_quantized_values_into_int32(u, scalar_types.uint4b8, packed_dim=1)
    assert packed.shape == (out_f, in_f // 8) and packed.dtype == torch.int32
    un = unpack_quantized_values_into_int32(packed, scalar_types.uint4b8, packed_dim=1)
    assert torch.equal(un.to(torch.int32), u), f"round-trip mismatch @ {(out_f, in_f)}"
    # sai số dequant mô phỏng (rel-fro)
    q = u.to(torch.float32).reshape(out_f, in_f // 128, 128) - 8
    s = scale.to(torch.float32).unsqueeze(2)
    deq = (q * s).reshape(out_f, in_f)
    rel = (deq - W.float()).norm() / W.float().norm()
    print(f"[OK] shape {(out_f, in_f)}  packed {tuple(packed.shape)}  rel-err {rel:.4f}")
print("SELFTEST (RTN+pack) PASSED")
