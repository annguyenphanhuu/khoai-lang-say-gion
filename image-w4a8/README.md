# Online W4A8 (int4-g128 + FP8 act) — Workstream B

Đường DUY NHẤT vượt 64.67: hạ TPOT ~3.8→~2.7ms bằng weight 4-bit qua **kernel CUTLASS W4A8 có
sẵn của vLLM** (`CutlassW4A8LinearKernel`: act FP8-e4m3 per-token, CUDA-graph). KHÔNG viết CUDA.
Hợp lệ "online only": quantize BF16 gốc LÚC LOAD, không ship checkpoint pre-quant.

## File
- `online_w4a8.py` — quant method (RTN int4-g128 → nạp CutlassW4A8LinearKernel). Chỉ quantize
  LinearBase (attention/MLP/conv-proj); lm_head tied + embedding giữ BF16.
- `register_patch.py` — append đăng ký `online_w4a8` vào vLLM (chạy lúc build).
- `validate_build.py` — check syntax+import+register lúc build (không cần GPU).
- `_selftest.py` — test CPU phần toán RTN+pack (đã PASS local).
- `Dockerfile` — FROM image full-fp8, cài patch + validate.

## Đã verify LOCAL (CPU, không Hopper)
- ✅ syntax, import, đăng ký method (`get_quantization_config('online_w4a8')` OK).
- ✅ RTN + pack/unpack round-trip KHỚP định dạng kernel (mọi shape LFM2), rel-err ~0.114.
- ✅ chữ ký constructor param (PackedvLLMParameter/GroupQuantScaleParameter) đúng.
- ❌ CHƯA verify (cần Hopper cap90): kernel CUTLASS runtime + accuracy thực. → portal submit #1.

## Build & push (máy có mạng; KHÔNG cần GPU)
```bash
docker build -f image-w4a8/Dockerfile -t <YOU>/vllm-w4a8:online-r1 image-w4a8/
docker push <YOU>/vllm-w4a8:online-r1
docker inspect --format='{{index .RepoDigests 0}}' <YOU>/vllm-w4a8:online-r1   # lấy @sha256
```
Dán digest vào `compose-w4a8.yml` (thay `REPLACE_WITH_YOUR_W4A8_IMAGE@sha256:REPLACE_AFTER_PUSH`).

## Nộp (submit #1) — đọc kết quả
- `tbt_median ~2.5–3.0` + `accuracy_drop` nhỏ → **~74**. Sau đó stack TTFT tiến 78–80.
- boot fail → log portal (fail-loud assert) chỉ đúng dòng → sửa → **submit #2**.
- `tbt` vẫn cao → kernel không như kỳ vọng → xem lại (đổi Marlin-FP4 hoặc mixed-precision).

## Nếu accuracy_drop lớn (dự phòng)
Chuyển mixed-precision: trong `online_w4a8.py::get_quant_method`, trả `None` cho các layer
`self_attn` (q/k/v/o — lỗi int4 cao nhất) để giữ chúng ở method mặc định (FP8), chỉ int4 cho MLP.
