# WAVE7 — CPU-lean: bài 63.38 + fast-path serialize SSE (fastsse).
# (ShortConv patch đã gỡ 2026-07-24: giả thuyết bị bác bỏ — conv đã trong Full CUDA Graph.)
ARG BASE=annguyenphanhuu/vllm-fastsse:fullfp8-20260721-r1
FROM ${BASE}

COPY _fastsse.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/_fastsse.py
COPY patch_fastsse.py /opt/perf-patch/patch_fastsse.py
COPY verify_fastsse.py /opt/perf-patch/verify_fastsse.py

RUN python3 /opt/perf-patch/patch_fastsse.py \
 && python3 /opt/perf-patch/verify_fastsse.py \
 && python3 -c "import ast; from pathlib import Path; p='/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py'; ast.parse(Path(p).read_text()); print('cpulean: serving.py syntax OK')"
