"""Online W4A8 (int4 weight group-128 + FP8 activation) cho LFM2 — TUÂN THỦ 'online only'.

Cơ chế: load BF16 GỐC như bình thường -> RTN int4-g128 NGAY LÚC LOAD -> nạp vào kernel
CUTLASS W4A8 CÓ SẴN của vLLM (CutlassW4A8LinearKernel: act FP8 e4m3 per-token, weight int4,
CUDA-graph). KHÔNG viết CUDA, KHÔNG ship checkpoint pre-quant.

Chọn --quantization online_w4a8. Chỉ quantize LinearBase (bỏ embedding/lm_head tied -> BF16).
Fail-loud: mọi giả định sai đều assert để lộ nguyên nhân ở boot đầu tiên (không phí submit mù).
"""
from typing import Any

import torch

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

GROUP_SIZE = 128


def _rtn_int4_g128(W: torch.Tensor, group_size: int = GROUP_SIZE):
    """W[out,in] bf16 -> (u[out,in] int32 uint4b8 in [0,15], scale[out,in/gs] bf16).

    Symmetric: scale=absmax/7; q=clamp(round(W/scale),-8,7); u=q+8. Kernel dequant:
    (u-8)*group_scale*chan_scale ≈ W.
    """
    out_f, in_f = W.shape
    assert in_f % group_size == 0, f"in_features {in_f} not divisible by {group_size}"
    ng = in_f // group_size
    Wf = W.detach().to(torch.float32).reshape(out_f, ng, group_size)
    absmax = Wf.abs().amax(dim=2, keepdim=True)
    scale = (absmax / 7.0).clamp(min=1e-8)                 # [out,ng,1]
    q = torch.round(Wf / scale).clamp(-8, 7)               # [out,ng,gs] signed
    u = (q + 8).to(torch.int32).reshape(out_f, in_f)       # uint4b8 [0,15]
    scale = scale.reshape(out_f, ng).to(torch.bfloat16).contiguous()
    return u, scale


class OnlineW4A8LinearMethod(QuantizeMethodBase):
    def __init__(self, quant_config: "OnlineW4A8Config"):
        self.quant_config = quant_config

    # copy y hệt UnquantizedLinearMethod.create_weights -> BF16 load chuẩn (hỗ trợ fused QKV/gate_up).
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        from vllm.model_executor.parameter import ModelWeightParameter
        from vllm.model_executor.utils import set_weight_attrs
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition,
                             dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        from vllm.model_executor.kernels.linear.mixed_precision.cutlass import (
            CutlassW4A8LinearKernel)
        from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
            MPLinearLayerConfig)
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            pack_quantized_values_into_int32)
        from vllm.model_executor.parameter import (
            BasevLLMParameter, GroupQuantScaleParameter, PackedvLLMParameter)
        from vllm.scalar_type import scalar_types

        W = layer.weight.data
        assert W.dim() == 2, f"online_w4a8 expects 2D weight, got {tuple(W.shape)}"
        out_f, in_f = W.shape
        assert out_f % 128 == 0 and in_f % 128 == 0, \
            f"online_w4a8 needs out%128 & in%128, got {(out_f, in_f)}"

        u, scale_bf16 = _rtn_int4_g128(W, GROUP_SIZE)
        packed = pack_quantized_values_into_int32(u, scalar_types.uint4b8, packed_dim=1)

        def _noop(*a, **k):
            return None

        wp = PackedvLLMParameter(data=packed, input_dim=1, output_dim=0,
                                 packed_dim=1, packed_factor=8, weight_loader=_noop)
        ws = GroupQuantScaleParameter(data=scale_bf16, output_dim=0, input_dim=1,
                                      weight_loader=_noop)
        wsh = BasevLLMParameter(
            data=torch.tensor([out_f, in_f], dtype=torch.int64, device=W.device),
            weight_loader=_noop)

        delattr(layer, "weight")                       # free BF16
        layer.register_parameter("weight_packed", wp)
        layer.register_parameter("weight_scale", ws)
        layer.register_parameter("weight_shape", wsh)

        cfg = MPLinearLayerConfig(
            full_weight_shape=(in_f, out_f),
            partition_weight_shape=(in_f, out_f),
            weight_type=scalar_types.int4,
            act_type=torch.float8_e4m3fn,
            group_size=GROUP_SIZE,
            zero_points=False,
            has_g_idx=False,
            out_type=torch.bfloat16,
        )
        kernel = CutlassW4A8LinearKernel(
            cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale")
        kernel.process_weights_after_loading(layer)   # convert scales fp8 + cutlass encode/reorder
        layer.w4a8_kernel = kernel

    def apply(self, layer, x, bias=None):
        return layer.w4a8_kernel.apply_weights(layer, x, bias)


class OnlineW4A8Config(QuantizationConfig):
    def __init__(self):
        super().__init__()
        self._fp8_cfg = None

    def _fp8(self):
        """Fp8Config của CHÍNH image này (dispatch ParallelLMHead đã có nhờ patch_full_fp8.py).

        Dùng lại nguyên đường code đã chạy thật ở compose-cpu-lean.yml (64.67) thay vì tự
        viết method mới cho lm_head.
        """
        if self._fp8_cfg is None:
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            self._fp8_cfg = Fp8Config(is_checkpoint_fp8_serialized=False,
                                      activation_scheme="dynamic")
        return self._fp8_cfg

    @classmethod
    def get_name(cls):
        return "online_w4a8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90  # CutlassW4A8 = Hopper

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineW4A8Config":
        return cls()

    def get_quant_method(self, layer, prefix: str):
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        if isinstance(layer, LinearBase):
            return OnlineW4A8LinearMethod(self)
        # [K3] lm_head 65536x2048: BF16 = 268MB/step. FP8 = 134MB/step ⇒ -0.22ms TPOT.
        # KHÔNG int4 (lm_head là layer sai số cao nhất); FP8 đã đủ hạ một nửa byte.
        # embed_tokens là VocabParallelEmbedding thuần (không phải ParallelLMHead) ⇒ vẫn BF16.
        if isinstance(layer, ParallelLMHead):
            # [K4] LMHEAD_INT4=1 => int4 luôn: 134MB -> 67MB/step (-0.11ms). Rủi ro accuracy cao hơn
            # FP8 (lm_head là layer sai số lớn nhất) ⇒ NẾU thắng ERS thì BẮT BUỘC verify GPQA trên rig
            # trước khi chọn làm bài chốt, không tin `accuracy_drop` của portal.
            import os
            if os.environ.get("LMHEAD_INT4") == "1":
                return OnlineW4A8LinearMethod(self)
            method = self._fp8().get_quant_method(layer, prefix)
            assert method is not None, (
                "fp8 dispatch trả None cho ParallelLMHead -> base image thiếu patch_full_fp8")
            return method
        return None  # embedding / norms -> giữ nguyên (BF16)
