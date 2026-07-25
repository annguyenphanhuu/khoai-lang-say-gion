# Online W4A8 (int4-g128 + FP8 activation) cho LFM2 — reuse CutlassW4A8LinearKernel có sẵn.
# TUÂN THỦ 'online only': quantize BF16 gốc NGAY LÚC LOAD, KHÔNG ship checkpoint pre-quant.
#
# BUILD (máy CÓ mạng; KHÔNG cần GPU — validate chỉ syntax+register):
#   docker build -f image-w4a8/Dockerfile -t <YOU>/vllm-w4a8:online-r1 image-w4a8/
#   docker push <YOU>/vllm-w4a8:online-r1
#   # lấy digest @sha256 sau push -> dán vào compose-w4a8.yml
ARG BASE=annguyenphanhuu/vllm-fastsse:fullfp8-20260721-r1
FROM ${BASE}

COPY online_w4a8.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/online_w4a8.py
COPY register_patch.py /opt/w4a8/register_patch.py
COPY validate_build.py /opt/w4a8/validate_build.py

RUN python3 /opt/w4a8/register_patch.py \
 && python3 /opt/w4a8/validate_build.py
