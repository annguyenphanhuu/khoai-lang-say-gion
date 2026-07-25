# CANDIDATE P4 — Speculative decoding: Full-FP8 target 1.2B + LFM2.5-230M draft tại /draft.
#
# TUÂN THỦ "Online Quantization only": draft = weights BF16 GỐC, KHÔNG pre-quantize checkpoint.
# vLLM online-FP8 draft qua speculative-config quantization=fp8. Base Full-FP8 patch (ShortConv
# proj + LM head FP8) áp cho CẢ target lẫn draft vì patch nằm ở vLLM source.
#
# BUILD (máy CÓ MẠNG, không phải lúc chấm):
#   pip install -U "huggingface_hub[cli]"
#   hf download LiquidAI/LFM2.5-230M --local-dir image-draft-spec/draft-230m
#   docker build -f image-draft-spec/Dockerfile -t <YOU>/vllm-draft-spec:230m-r1 image-draft-spec/
#   docker push <YOU>/vllm-draft-spec:230m-r1
#   # lấy digest @sha256 sau push, dán vào compose-opt-draft-spec.yml
#
# Đổi 350M: hf download LiquidAI/LFM2.5-350M --local-dir image-draft-spec/draft-350m + sửa COPY.
ARG BASE=annguyenphanhuu/vllm-fastsse:fullfp8-20260721-r1
FROM ${BASE}

COPY draft-230m /draft

# Fail BUILD sớm nếu draft sai (vocab phải == target 65536 để qua verify_equal_vocab_size_if_draft_model).
RUN test -f /draft/config.json \
 && python3 -c "import json; c=json.load(open('/draft/config.json')); \
assert c.get('vocab_size')==65536, ('vocab mismatch', c.get('vocab_size')); \
print('draft OK: vocab', c['vocab_size'], '| layers', c.get('num_hidden_layers'), '| arch', c.get('architectures'))"
