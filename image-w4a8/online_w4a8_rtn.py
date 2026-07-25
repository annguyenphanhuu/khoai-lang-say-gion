"""RTN int4 group-128 (symmetric) — LÕI toán cho online W4A8.

Đây là phần TÔI KIỂM SOÁT (thuần torch, test được CPU). Kernel CUTLASS W4A8 của vLLM lo phần
còn lại (encode/reorder int4, pack scale fp8, gemm). Dequant của kernel là:
    w ≈ (uint4b8 - 8) * group_scale * channel_scale
nên ở đây ta chỉ cần: scale (bf16, per [out, in/128]) sao cho  W ≈ q_signed * scale,
với q_signed = clamp(round(W/scale), -8, 7).

Hàm này DÙNG CHUNG cho cả unit-test local lẫn patch tích hợp (patch import từ file này).
"""
import torch

GROUP_SIZE = 128
QMAX = 7  # int4 symmetric: levels [-8, 7], dùng 7 cho scale (‑8 vẫn hợp lệ khi clamp)


def rtn_int4_group_sym(W: torch.Tensor, group_size: int = GROUP_SIZE):
    """W: [out, in] (bf16/fp16/fp32) -> (q_signed[int8 in -8..7] [out,in], scale[out, in//gs]).

    Nhóm dọc theo INPUT dim (khớp group-128 của CutlassW4A8: scale {input_dim=0? -> per (out, group)}).
    """
    assert W.dim() == 2, W.shape
    out_f, in_f = W.shape
    assert in_f % group_size == 0, (in_f, group_size)
    ng = in_f // group_size
    Wf = W.to(torch.float32).reshape(out_f, ng, group_size)
    absmax = Wf.abs().amax(dim=2)                      # [out, ng]
    scale = (absmax / QMAX).clamp(min=1e-8)            # [out, ng]
    q = torch.round(Wf / scale.unsqueeze(2)).clamp(-8, 7).to(torch.int8)
    return q.reshape(out_f, in_f), scale               # scale kept fp32; patch sẽ .to(bf16)


def dequant_int4_group_sym(q_signed: torch.Tensor, scale: torch.Tensor,
                           group_size: int = GROUP_SIZE) -> torch.Tensor:
    out_f, in_f = q_signed.shape
    ng = in_f // group_size
    qf = q_signed.to(torch.float32).reshape(out_f, ng, group_size)
    # mô phỏng ĐÚNG kernel: scale bf16 (mất chính xác) rồi dequant
    s = scale.to(torch.bfloat16).to(torch.float32).unsqueeze(2)
    return (qf * s).reshape(out_f, in_f)


def quant_error(W: torch.Tensor):
    """Trả (rel_fro, cos) — sai số quantize→dequant int4-g128 mô phỏng path kernel."""
    q, s = rtn_int4_group_sym(W)
    Wq = dequant_int4_group_sym(q, s)
    Wf = W.to(torch.float32)
    rel_fro = (Wq - Wf).norm() / Wf.norm().clamp(min=1e-12)
    cos = torch.nn.functional.cosine_similarity(
        Wq.reshape(1, -1), Wf.reshape(1, -1)).item()
    return rel_fro.item(), cos
