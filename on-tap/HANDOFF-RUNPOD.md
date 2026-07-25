# HANDOFF — LFM2 vLLM. Trạng thái sau session H100 (2026-07-25)

**Best: 65.06 (`compose-w4a8`) / 64.67 (fp8 cpu-lean). Cần 75 vào vòng; mục tiêu 80.**
**KẾT LUẬN: MỌI đòn config + kernel-quant ĐÃ CẠN. Đường 80+ duy nhất = speculative decoding → `on-tap/NEXT-80.md`.**

## Scoring (nhớ nhanh)
`ERS = 50·(s_ttft+s_tpot)`, `s(x)=clamp((C−x)/(C−F),0,1)²`. TTFT F10/C400. TPOT(=tbt) F1/C10.
Nhạy: **tbt ~8.6đ/ms (đắt)**, ttft ~0.23đ/ms. Hiện tbt~3, ttft~45, 6-7 fail → ~65.
Để 80 cần s_sum≈1.6. tbt kẹt 3ms ⇒ s_ttft phải ≈1.0 (ttft~10ms, bất khả) ⇒ **BẮT BUỘC hạ tbt** — chỉ spec decode làm được.

## ⛔ NGÕ CỤT — ĐÃ CHỨNG MINH, ĐỪNG THỬ LẠI
| Đòn | KQ | Bằng chứng (đo H100 hôm nay / lịch sử) |
|---|---|---|
| **int4 / W4A8 (mọi kernel)** | **CHẾT** | Machete (SOTA Hopper int4) = **0.70x fp8** @batch-1, đo CUDA-graph sạch (fp8 42µs vs int4 60µs / layer). Dequant > byte tiết kiệm ⇒ W4A8 hòa fp8. Không kernel int4 nào cứu được. |
| **CPU input-path** (sampling/detok/prepare) | vô ích | tbt 3-core = 8-core (1.55≈1.67ms) ⇒ GPU-bound, CPU đã bị che. Loại toàn bộ P3. |
| **fuse_norm_quant / fuse_act_quant** | đã bật sẵn | default-on vllm≥0.25.1 ⇒ baseline đã fuse. Chỉ còn `fuse_attn_quant` (kỳ vọng ~0). |
| FP8-KV / right-size / interactivity / BnB-NF4 / W4A16 | regress/chậm | KB §4.3, §5. |
| **draft_model** (draft LFM2 hybrid) | boot-crash | 2 kv_cache_group assert. Mọi draft LFM2 hybrid chết. |
| async explicit / flashinfer / capture-128 / n-gram (lần 1) | null/reject | KB §5. |

**tbt=3ms = op-floor GPU vật lý** (full-H100: linear 0.68ms + phi-linear 0.85ms = 1.53ms; MiG bóp BW ~2x → 3ms). Không hạ được bằng quant/fusion/CPU. Phi-linear (conv/attn/norm) LỚN HƠN linear.

## 🖥️ Rig RunPod — resume nhanh
- Pod **H100 SXM 80GB**, template *RunPod PyTorch 2.8*, **driver 580 / CUDA 13.0** (image cần CUDA13 ✓), cap 9.0.
- **Network volume `/workspace`** sống qua Stop: model `/workspace/model` (HF `LiquidAI/LFM2.5-1.2B-Instruct`), bench `/workspace/bench`.
- **vllm 0.25.1** pip (torch 2.11+cu130). ⚑ **Luật 3 cho dùng image vllm BẤT KỲ** ⇒ build từ 0.25.1, KHÔNG khóa vào fork fastsse cũ.
- SSH: `ssh <podid>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519` (proxy, paste đa-dòng OK). **ĐỪNG dùng web terminal** (rớt ký tự; paste dài phải chia base64 <300 char/dòng).
- Serve baseline: `OMP_NUM_THREADS=1 taskset -c 0-2 vllm serve /workspace/model --served-model-name LFM2.5-1.2B-Instruct --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.225 --enable-prefix-caching --quantization fp8 --disable-log-stats`
- Bench: `loadgen_c1.py` (tbt), `wl_probe.py` (ttft workload-shape), `graph_bench.py` (GPU-time kernel), `correctness_diff.py` (cổng temp=0 bắt buộc).
- ⚠️ Full-H100 BW dư → tbt tuyệt đối 1.53ms (portal 3ms bóp BW, **KHÔNG tái tạo được**). Rig ĐO ĐƯỢC: **correctness, acceptance spec, GPU-time tương đối** — đủ để phát triển spec decode. Chốt điểm cuối trên **portal**.
- 💸 Pod ~$3/giờ → **Stop khi nghỉ** (volume rẻ vẫn giữ data).

→ **Bước tiếp: `on-tap/NEXT-80.md`.** Rổ 5 bài an toàn ~63-65 giữ nguyên (STRATEGY §4).
