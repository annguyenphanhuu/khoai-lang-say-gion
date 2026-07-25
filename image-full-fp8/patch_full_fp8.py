"""Exact-match vLLM 0.22.1 patch extending online FP8 for LFM2.5."""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SHORT_CONV = ROOT / "model_executor/layers/mamba/short_conv.py"
LFM2 = ROOT / "model_executor/models/lfm2.py"
FP8 = ROOT / "model_executor/layers/quantization/fp8.py"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


short_conv = SHORT_CONV.read_text(encoding="utf-8")
short_conv = replace_once(
    short_conv,
    "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
    "from vllm.model_executor.layers.mamba.abstract import MambaBase\n"
    "from vllm.model_executor.layers.quantization import QuantizationConfig\n",
    "ShortConv quantization import",
)
short_conv = replace_once(
    short_conv,
    """        cache_config: CacheConfig | None = None,
        prefix: str = "",
""",
    """        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,  # [perf-patch full-fp8]
        prefix: str = "",
""",
    "ShortConv quant_config argument",
)
short_conv = replace_once(
    short_conv,
    """            bias=self.bias,
            prefix=f"{prefix}.in_proj",
        )
""",
    """            bias=self.bias,
            quant_config=quant_config,  # [perf-patch full-fp8]
            prefix=f"{prefix}.in_proj",
        )
""",
    "ShortConv in_proj",
)
short_conv = replace_once(
    short_conv,
    """            bias=self.bias,
            prefix=f"{prefix}.out_proj",
        )
""",
    """            bias=self.bias,
            quant_config=quant_config,  # [perf-patch full-fp8]
            prefix=f"{prefix}.out_proj",
        )
""",
    "ShortConv out_proj",
)
SHORT_CONV.write_text(short_conv, encoding="utf-8")

lfm2 = LFM2.read_text(encoding="utf-8")
lfm2 = replace_once(
    lfm2,
    """            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.conv",
""",
    """            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,  # [perf-patch full-fp8]
            prefix=f"{prefix}.conv",
""",
    "LFM2 ShortConv wiring",
)
LFM2.write_text(lfm2, encoding="utf-8")

fp8 = FP8.read_text(encoding="utf-8")
fp8 = replace_once(
    fp8,
    "from vllm.model_executor.layers.quantization import QuantizationMethods\n",
    "from vllm.model_executor.layers.quantization import QuantizationMethods\n"
    "from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead\n",
    "FP8 ParallelLMHead import",
)
fp8 = replace_once(
    fp8,
    """        if isinstance(layer, LinearBase):
            if is_layer_skipped(
""",
    """        if isinstance(layer, (LinearBase, ParallelLMHead)):  # [perf-patch full-fp8]
            if is_layer_skipped(
""",
    "FP8 ParallelLMHead dispatch",
)
FP8.write_text(fp8, encoding="utf-8")

print("full-fp8: exact-match patch applied")
