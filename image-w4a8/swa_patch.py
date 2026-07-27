"""[perf-patch lfm2-swa] Mo cong sliding-window cho 6 layer GQA cua LFM2 (chay luc BUILD).

VI SAO CAN PATCH (khong lam duoc bang CLI/hf-overrides — da doc source trong image):
  - arg_utils.py:1738 chi set CacheConfig.sliding_window khi `not is_interleaved(hf_text_config)`.
  - transformers_utils/config.py:568 is_interleaved = len(set(layer_types)) > 1. LFM2 co
    {"full_attention", "conv"} => True => moi --hf-overrides {"sliding_window": N} deu BI BO QUA.
  - lfm2.py goi Attention(...) KHONG truyen per_layer_sliding_window.
  => Duong duy nhat con lai la truyen per_layer_sliding_window o layer. Day la doi 1 kwarg,
     KHONG dung vao kernel: van FlashAttention 3, van full cudagraph, van fp8 .view() path.
     Khac han truc kv-cache-dtype (turboquant/fp8) — cho da tra thue kernel +3.0ms va thua 4/4.

CONG TAC: env LFM2_SWA. Khong set hoac =0 => per_layer_sliding_window=None => hanh vi Y HET r3.
"""

import re
from pathlib import Path

LFM2 = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/lfm2.py"
)
MARKER = "# [perf-patch lfm2-swa]"

src = LFM2.read_text(encoding="utf-8")
assert MARKER not in src, "already patched"

# 1) Header: doc env 1 lan luc import module.
anchor = "from vllm.model_executor.layers.attention import Attention"
assert src.count(anchor) == 1, f"anchor import khong tim thay/khong duy nhat: {anchor!r}"
src = src.replace(
    anchor,
    anchor
    + "\n"
    + MARKER
    + "\nimport os as _os\n"
    '_LFM2_SWA = int(_os.getenv("LFM2_SWA", "0") or 0) or None\n',
    1,
)

# 2) Them kwarg vao dung loi goi Attention( trong Lfm2Attention.
assert src.count("self.attn = Attention(") == 1, "so lan goi self.attn = Attention( != 1"
m = re.search(r"self\.attn = Attention\(.*?\n(?P<ind>[ \t]*)\)\n", src, re.S)
assert m is not None, "khong parse duoc block self.attn = Attention(...)"
inner_indent = m.group("ind") + "    "
injected = (
    m.group(0)[: -(len(m.group("ind")) + 2)]
    + f"{inner_indent}per_layer_sliding_window=_LFM2_SWA,  {MARKER}\n"
    + m.group("ind")
    + ")\n"
)
src = src[: m.start()] + injected + src[m.end() :]

LFM2.write_text(src, encoding="utf-8")

out = LFM2.read_text(encoding="utf-8")
assert "per_layer_sliding_window=_LFM2_SWA" in out
compile(out, str(LFM2), "exec")
print("lfm2 swa patch applied OK (default OFF: LFM2_SWA unset => None)")
