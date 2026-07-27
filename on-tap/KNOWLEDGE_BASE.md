# KNOWLEDGE BASE — LFM2.5 trên MiG H200 (SSOT)

> Chỉ ghi **số đo** và **source đã đọc**. Quyết định: `STRATEGY.md`. Cập nhật 2026-07-26, 20 submit.
> ⚠️ Đọc `STRATEGY.md` §1 và §2 trước khi dùng bảng §3: máy đo có **hai trạng thái cách 5.7 ERS**,
> và mô hình byte cũ ("KV = 48% traffic") đã **bị số đo bác bỏ**.

## 1. Phần cứng · model · workload

**Portal:** 1 slice **MiG H200 1g.18gb** — 18GB VRAM, **~0.6 TB/s HBM** (đo được 613 GB/s trên
đường weight), **~18 SM (1/7)**, cap 9.0, **3 CPU core**, 8GB RAM. vLLM trong image **0.22.1**.

**Model** `LFM2.5-1.2B-Instruct`: 16 layer = **10 short-conv + 6 GQA**. `hidden=2048`,
`intermediate=12288`, 32 head / 8 KV head, `head_dim=64`, `vocab=65536`, `conv_L_cache=3`.
Weight đọc/step: BF16 2.4GB · full-FP8 1.2GB · **K3 (int4-g128 body + FP8 lm_head) 0.734GB**.
`lm_head` 65536×2048 = 134M param ⇒ BF16 268MB · FP8 134MB · int4 67MB.

**Workload** (`grading-workload-spec.json`): 70 hội thoại × 6 turn = **420 request đều được chấm**.
Shared prefix 1000 tok + per-conv prefix 1000 tok + 150 user/turn; **output pin 300**; Poisson seed 42.
Context ~2150 → ~4400. **Turn 2–6 (83% request) chỉ prefill ~150 token mới** (prefix-cache ăn thật)
⇒ ~3–6 ms compute. Batch decode thật **B ≈ 27–31**, do tải quyết định không do knob.

## 2. Hàm điểm & cách bóc

`ERS = mean_req(50·(s_ttft + s_tpot))`, `s(x)=clamp((C−x)/(C−F),0,1)²`, request lỗi ⇒ `s=0`.
TTFT: F=10 C=400ms · TPOT: F=1 C=10ms. `Điểm cuối = 100·ERS·f(Δacc)`, `f=1` nếu Δ≤0.10.
Bóc TPOT: `s_ttft=((400−ttft_p50)/390)²` · `TPOT = 10 − 9·√(ERS_hc/50 − s_ttft)`
với `ERS_hc = ERS·420/(420−failed)`.
⚠️ `tbt_median` làm tròn số nguyên ⇒ vô dụng. ⚠️ Khi `failed > 10`, hiệu chỉnh `ERS_hc` **sai**
(lỗi không phải MCAR — chúng là đuôi chậm, vốn đã cho ~0 điểm): dùng raw ERS, bỏ TPOT bóc.

**Vật lý TPOT (hồi quy 4 điểm weight, R²=0.986):** `TPOT = 2.05 ms + 1.63 ms/GB × weight_GB`.
Cơ cấu K3: **sàn 2.05 (61%) · weight 1.20 (36%) · KV ≤0.44 (≤13%)**. ~1.6 ms chưa giải thích.
⚠️ **Đừng suy TPOT từ phép tính byte KV.** Phép tính cho 1.09 GB/step; số đo cho ≤0.44 ms.

## 3. Bảng kết quả portal

**Đợt B — 7 lượt liên tiếp, 26/07** (cùng ngày, WARNING, `--disable-log-stats`, gpu-mem 0.90):

| slot | config (1 biến vs K3) | ERS | ttft p50/p95 | failed | TPOT |
| --: | :-- | --: | --: | --: | --: |
| 1 | **K3 nguyên bản** | **59.74** | 65/92 | 4 | 3.84 |
| 2 | `--renderer-num-workers=2` | 65.12 | 55/72 | 6 | 3.39 |
| 3 | `--no-scheduler-reserve-full-isl` (**no-op**) | 65.46 | 55/81 | 6 | 3.35 |
| 4 | image r6 + kv fp8_e4m3 | 64.17 | 58/80 | 4 | 3.47 |
| 5 | kv fp8_e4m3 | 64.97 | 54/77 | 6 | 3.44 |
| 6 | **K3 nguyên bản** | **59.67** | 62/85 | **21** | (bỏ) |
| 7 | image r6 + kv fp8_e4m3 (lặp slot 4) | 65.42 | 59/83 | 6 | 3.25 |

**Đọc bảng này — ba kết luận đã kiểm định:**
- **Slot 1 và slot 3 là CÙNG MỘT CHƯƠNG TRÌNH** (§4.2) nhưng lệch 5.72 ERS ⇒ biến động trạng thái máy.
- 4 candidate (slot 2,3,4,5,7) là **một quần thể**: chi² = 1.56/3 dof, sd 0.52, Δ cặp lớn nhất 1.16
  so với ngưỡng phân giải 1v1 = 3.77 ⇒ **không cái nào hơn cái nào**.
- Ba run **nền** của 26/07 (2×K3 + K3+INFO) chiếm đúng hạng thấp nhất 1-2-3 của 11 run; 8 biến thể
  chiếm hạng 4–11 (p = 0.0061). Nhưng 5 knob **không liên quan về cơ chế** đều cho ~+5 và **không
  cộng dồn** ⇒ **không phải nhân quả**, là trạng thái máy tương quan với việc "chạy config nền".

**Đợt A — 4 lượt liên tiếp, 26/07:** K3+INFO 60.77 (ttft 63) · lm_head int4 63.32 (60) ·
image r6 fastsse 64.87 (58) · kv fp8_e4m3 66.94 (50).

**Lịch sử (ngày khác — KHÔNG so trực tiếp với đợt A/B):**

| Config | ERS | ttft | TPOT |
| :-- | --: | --: | --: |
| **K3 25/07 — chưa từng tái lập** | **68.57** | 47 | 3.19 |
| gpu-mem 0.95 | 67.03 | 48 | 3.47 |
| block-size 16 · 32 | 65.81 · 63.09 | 51 · 61 | 3.54 · 3.60 |
| K3 − `--disable-log-stats` | 65.25 | 55 | 3.38 |
| W4A8 lm_head BF16 | 65.06 | 50 | 3.66 |
| full FP8 | 64.67 · 62.50 | 45 · 48 | 3.86 |
| kv fp8_e4m3 + block16 | 64.13 | 57 | 3.58 |
| base v0.25.1 full-FP8 | 62.92 | — | — |
| performance-mode=interactivity | 62.31 | 61 | — |
| cascade (22 failed) | 61.37 | 55 | 3.56 |
| kv fp8_e5m2 | 60.98 | 63 | 3.81 |
| max-model-len right-size | 60.82 | — | — |
| max-num-batched-tokens 2048 | 60.44 | 63 | 3.88 |
| FlashInfer | 56.80 | — | — |
| kv turboquant_4bit_nc | 52.1 | 65 | 5.04 |
| BF16 anchor | 48.17 | 56 | ~6.0 |
| tắt CUDA graph | 33.93 | — | ~12 |
| BnB-NF4 · W4A16 | 30.8 · 22.51 | 84 · 205 | 15 · 6 |
| `LFM2_SWA=1024` | — | — | **`protocol aborted`** |

⚠️ **Portal không trả log**, chỉ trả chỉ số, chỉ khi chạy xong. `accuracy_drop=0` + `f_delta=1` ở
**mọi** run kể cả BnB-NF4 ⇒ không tin trường đó.
⚠️ `failed` nền 4–6 (1.3%, ăn ~0.85 ERS). `tokens_per_sec / tỉ lệ sống sót` = 0.0580 ± 0.0002
(cv 0.27%) ở 6/7 run — **trừ slot 7 thiếu 31%** ⇒ cờ đỏ chưa loại trừ trên image r6.

**Ba slope còn đứng vững** (Δ đủ lớn để vượt biến động trạng thái):
weight BF16→int4 = −2.8 ms TPOT (khớp 613 GB/s) · tắt cudagraph = tbt 3→12, **đừng đụng** ·
4-bit weight qua được accuracy gate.

## 4. Đã đóng — root cause

**4.1 KV quantization (ĐÓNG THẬT).** Đo `k = 1.006` (byte KV còn lại) thay vì 0.50 ⇒ đọc KV tốn
**≤0.44 ms ≤13% step**. Đã loại bằng source, không cần GPU:
- kernel `flash_fwd_hdim64_e4m3_paged_split_sm90` **CÓ** trong `.so` ⇒ giả thuyết "pad hdim 64→128" **SAI**
- cache cấp phát thật **1 byte/phần tử**; `block_size` không bị nhân đôi bởi hợp nhất page hybrid
  (mamba page LFM2 = 8192 B, dưới cả hai ngưỡng)
- `calculate_kv_scales` bị **ép tắt** cho model hybrid
- `flash_attn.py:753` `key_cache.view(fp8_dtype)` — `.view()`, không copy/dequant
- `flash_attn.py:295` `_cudagraph_support = ALWAYS if FA==3`, **không phụ thuộc kv dtype**;
  `fa_utils.py:78` chọn FA3 trên SM90; `flash_attn.py:186` không ràng buộc `head_size`
- turboquant_4bit_nc: byte giảm thật 3.66× nhưng backend riêng ép `flash_attn_version=2`,
  decode Triton, `UNIFORM_BATCH` cudagraph, prefill dequant khi q_len>128 ⇒ thuế kernel +3.0 ms
⇒ **KV không phải bottleneck.** Giả thuyết còn sống: decode attention trên slice 18 SM bị chặn bởi
**độ trễ / số giao dịch**, không phải bề rộng phần tử.

**4.2 `--no-scheduler-reserve-full-isl` = NO-OP.** `kv_cache_manager.py:346`
`full_num_tokens = min(request.num_tokens, max_model_len)` ⇒ reserve theo **ISL thật** (~4400 tok
≈ 54 MB), không theo 32768. Pool KV ~13 GB ≈ 1.07M token; đang dùng 27×4400 = 119K = **11%**
⇒ không bao giờ chặn. `apply_admission_cap` chỉ có hiệu lực với SWA/chunked-local.
**Đây là chốt logic của §1**: nó khiến slot 1 và slot 3 thành cùng một chương trình.

**4.3 Sliding-window — `protocol aborted`, mất bài.** Đóng vĩnh viễn mọi window. Kỹ thuật:
`arg_utils.py:1738` + `is_interleaved` ⇒ CLI/`hf-overrides` không set được; phải patch `lfm2.py`
(image r5, env `LFM2_SWA`, **mặc định OFF — không bao giờ bật**).

**4.4 Còn lại, ngắn gọn.** `--block-size` pin: chặn tự động unify page hybrid ⇒ prefix-cache tệ ⇒
ttft nổ · `gpu-mem 0.95`: pool to ⇒ B to · `max-num-batched-tokens 2048`: throttle prefill ⇒ B to ·
`max-model-len` right-size: pool do gpu-mem quyết định, **và hạ max-len rủi ro probe** ·
cascade: `gpu_model_runner.py:3777` `disable_full=use_cascade_attn` ⇒ mất FULL cudagraph, 22 failed ·
custom ShortConv / fuse / fused sampling ≈ 0 (đã trong cudagraph, fuse default-on) ·
mọi spec decode: draft LFM2 ⇒ 2 kv-group; n-gram ⇒ `short_conv.py` không rollback `conv_state` ·
`max-num-seqs` thấp / MSE-clip int4 / `--async-scheduling` explicit: vô nghĩa (B do tải;
async **đã default-on**, `config/vllm.py:975`) · FlashInfer: prefill chậm · SGLang/TRT-LLM: **phạm luật**.
`--api-server-count`: **no-op** — `api_server.py:665 run_server()` gọi `run_server_worker()` 1 lần,
không fork; cờ chỉ được `cli/serve.py` dùng mà portal ép entrypoint module.
Rust frontend: `VLLM_USE_RUST_FRONTEND` có trong `envs.py:545` nhưng binary `vllm-rs` **không có
trong image**. Đuôi TTFT turn-1: không có (p95 chỉ 67–92).

## 5. ⚠️ Rig H100 (RunPod) SAI regime băng thông

HBM **3.35 TB/s (5.6×)**, **132 SM (~7×)** so với MiG. Đọc weight FP8: rig 0.36 ms vs portal 2.0 ms.
⇒ Mọi phán quyết về **băng thông** đo trên rig đều vô giá trị.
**Rig dùng được cho:** accuracy/GPQA, correctness temp=0, **danh mục kernel + tỉ lệ giữa chúng**
(nsys/ncu — đây là việc cần làm, `STRATEGY.md` §5B). Vận hành: `RUNBOOK-RUNPOD.md`.

## 6. Đọc source trong image (không cần GPU)

`docker run --rm --entrypoint bash annguyenphanhuu/vllm-w4a8:online-r6 -c 'sed -n "1,80p" /usr/local/lib/python3.12/dist-packages/vllm/<path>'`

**Mặc định đáng nhớ (vLLM 0.22.1):** `renderer_num_workers=1` (ThreadPoolExecutor,
`renderers/base.py:88`) · `scheduler_reserve_full_isl=True` (`config/scheduler.py:140`) ·
`enable_log_requests=False` (⇒ INFO **không** in prompt mỗi request) · `async_scheduling` auto **True** ·
`prefix_caching_hash_algo="sha256"`, `xxhash` **chưa cài** · `stream_interval=1` (đừng đổi — §0).
`CacheDType` hợp lệ: `auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla,
turboquant_{k8v4,4bit_nc,k3v4_nc,3bit_nc}, int8_per_token_head, fp8_per_token_head, nvfp4`.

**`lm_head` + tie_weights:** `lfm2.py:498-504` tạo `ParallelLMHead` rồi `tie_weights(embed_tokens)`;
`vocab_parallel_embedding.py:272` gọi `quant_config.get_quant_method(self)` ⇒ FP8-hoá lm_head
**không phá** embedding lookup (`embed_tokens` vẫn BF16). Cơ sở của K3.

## 7. Image & build

- `image-w4a8/` → `online-r3` (**K3, dùng cho bài chốt**) · `online-r5` (+ env `LMHEAD_INT4`,
  `LFM2_SWA`) · `online-r6` (+ fastsse, **§3.3 có cờ đỏ**). Base `vllm-fastsse:fullfp8-20260721-r1`.
- Dockerfile tái lập được r6: đã chạy thử chuỗi build trên CPU, `serving.py` md5
  `1cb01e491e6145533877afd0da1d0aa7` trùng r6, `verify_fastsse OK: 132 cases byte-identical`.
  ⇒ Build r7 (ví dụ thêm patch fork uvicorn) làm được ngay, không cần GPU.
- `verify_fastsse.py` là **cổng cứng** lúc build (lệch 1 byte ⇒ build fail).
- `image-fullfp8-v0251/` = base v0.25.1 (lever dự phòng, 62.92).
- Đã xoá `image-cpulean/`, `image-full-fp8/` và 21 compose bài thua — lấy lại bằng
  `git show HEAD:<path>`; số đo giữ ở §3.
