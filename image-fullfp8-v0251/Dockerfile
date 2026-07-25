# WAVE10 — cùng patch Full-FP8, ĐỔI ENGINE sang v0.25.1. Biến DUY NHẤT = version vLLM.
#
# GIẢ THUYẾT (fit tốt nhất cho một đội 70-80): per-step overhead của EngineCore là thứ
# chặn TPOT ở 3.61ms. async scheduling che phần PREPARE (1.66ms) nhưng phần CRITICAL
# (post-GPU: sample→output→IPC, ~3ms) thì KHÔNG. Giữa v0.22.1 và v0.25.1, vLLM đã viết
# lại nhiều hot-path của V1 engine + async scheduler. Nếu v0.25.1 che được NHIỀU hơn hoặc
# loop nhẹ hơn ⇒ TPOT tiến về sàn GPU (~1ms) ⇒ s_tpot → ~1.0 ⇒ có thể +15..25đ.
# Đây là structural axis DUY NHẤT chưa test sạch (W1-B so stock-vs-patched, không phải
# same-patch-different-version).
FROM vllm/vllm-openai:v0.25.1-cu129
COPY patch_full_fp8.py /opt/perf-patch/patch_full_fp8.py
RUN python3 /opt/perf-patch/patch_full_fp8.py \
 && python3 -c "import ast; from pathlib import Path; files=['/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/short_conv.py','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/lfm2.py','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py']; [ast.parse(Path(p).read_text()) for p in files]; assert sum(Path(p).read_text().count('[perf-patch full-fp8]') for p in files) == 5; print('full-fp8 v0251: syntax + 5 markers OK')"
