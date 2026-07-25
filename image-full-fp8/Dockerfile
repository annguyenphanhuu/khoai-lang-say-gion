# Extend vLLM's online FP8 coverage for LFM2.5 without changing checkpoint files:
#   1) ShortConv dense in/out projections
#   2) tied LM head (the input embedding remains BF16)
ARG BASE=vllm/vllm-openai:v0.22.1
FROM ${BASE}

LABEL org.opencontainers.image.source="https://github.com/annguyenphanhuu/khoai-lang-say-gion" \
      org.opencontainers.image.description="Auditable online FP8 coverage patch for LFM2.5 ShortConv and LM head"

COPY patch_full_fp8.py /opt/perf-patch/patch_full_fp8.py

RUN python3 /opt/perf-patch/patch_full_fp8.py \
 && python3 -c "import ast; from pathlib import Path; files=['/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/short_conv.py','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/lfm2.py','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py']; [ast.parse(Path(p).read_text()) for p in files]; assert sum(Path(p).read_text().count('[perf-patch full-fp8]') for p in files) == 5; print('full-fp8: syntax and markers OK')"
