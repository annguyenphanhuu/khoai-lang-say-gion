# Online W4A8 (int4-g128 + FP8 activation) cho LFM2 — reuse CutlassW4A8LinearKernel có sẵn.
# TUÂN THỦ 'online only': quantize BF16 gốc NGAY LÚC LOAD, KHÔNG ship checkpoint pre-quant.
#
# Dockerfile này tái lập đúng `online-r6` (= r3 + công tắc LMHEAD_INT4/LFM2_SWA + fastsse).
# Đã verify: fastsse_patch.py cho serving.py md5 = 1cb01e491e6145533877afd0da1d0aa7, trùng r6.
#
# BUILD (máy CÓ mạng; KHÔNG cần GPU — validate chỉ syntax+register):
#   docker build -f image-w4a8/Dockerfile -t <YOU>/vllm-w4a8:online-r7 image-w4a8/
#   docker push <YOU>/vllm-w4a8:online-r7
#   # lấy digest @sha256 sau push -> dán vào compose-*.yml
#
# ⚠️ LFM2_SWA: công tắc sliding-window, MẶC ĐỊNH OFF. Bật lên = `protocol aborted`, MẤT BÀI.
#    Xem STRATEGY.md §0.2. Không bao giờ set biến này trong compose.
ARG BASE=annguyenphanhuu/vllm-fastsse:fullfp8-20260721-r1
FROM ${BASE}

COPY online_w4a8.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/online_w4a8.py
COPY _fastsse.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/_fastsse.py
COPY register_patch.py /opt/w4a8/register_patch.py
COPY swa_patch.py /opt/w4a8/swa_patch.py
COPY fastsse_patch.py /opt/w4a8/fastsse_patch.py
COPY verify_fastsse.py /opt/w4a8/verify_fastsse.py
COPY validate_build.py /opt/w4a8/validate_build.py

# verify_fastsse.py là CỔNG CỨNG: build fail nếu frame fast-path lệch 1 byte so với
# model_dump_json(exclude_unset=True) trên 132 case hiểm. Đây là toàn bộ lập luận
# "output-preserving" của patch — đừng bỏ bước này.
RUN python3 /opt/w4a8/register_patch.py \
 && python3 /opt/w4a8/swa_patch.py \
 && python3 /opt/w4a8/fastsse_patch.py \
 && python3 /opt/w4a8/verify_fastsse.py \
 && python3 /opt/w4a8/validate_build.py
