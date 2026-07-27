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
Cơ cấu K3: **sàn 2.05 (61%) · weight 1.20 (36%) · KV ≤0.44 (≤13%)**.
⚠️ **Đừng suy TPOT từ phép tính byte KV.** Phép tính cho 1.09 GB/step; số đo cho ≤0.44 ms.
✅ **1.6 ms "chưa giải thích" đã được định danh = decode attention** — xem **§8** (nsys, 27/07).

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

## 8. 🔬 nsys kernel-level, 27/07 — SÀN LÀ ATTENTION

**Cách đo** (`bench/prof_all.sh` + `bench/load27.py` + `bench/analyze.py`, rig H100, MPS
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14` ≈ 18/132 SM, `gpu-mem 0.2025` ≈ slice 18GB, `taskset 0-2`,
vLLM 0.25.1 + patch `online_w4a8` + `bench/fp8_lmhead_patch.py`, `--cuda-graph-trace=node`).
Cửa sổ **decode thuần**: 27 request `ignore_eos` sinh dài, **không** request mới ⇒ B=27 chính xác,
0% prefill. Phân đoạn step bằng lm_head GEMM (1 lần/step); kiểm chéo: **6 kv-write + 6 attention
+ 10 conv mỗi step** khớp đúng `layer_types` (6 full_attention + 10 conv). 3691 step đo được.

| Hạng mục | /step | ms/step | % busy |
| :-- | --: | --: | --: |
| **attention** (`flash_attn_fwd_sm90` hdim64 + combine) | 19 | **2.75** | **54%** |
| gemm (w4a8 mixed ×44, lm_head fp8 ×1, cublas ×20) | 65 | 1.874 | 37% |
| elementwise / norm / act | 124 | 0.375 | 7.4% |
| conv (`_causal_conv1d_update`) · other | 15.5 | 0.057 | 1.1% |
| **tổng** | **223.5** | 5.06 busy + 0.36 gap | |

**Bốn câu hỏi của Track A — trả lời hết:**
1. **223 kernel/step**, không phải 80–100. Nhưng con số này **không quan trọng** (xem 4).
2. Top kernel: **một kernel duy nhất chiếm 54%** — `flash_attn_fwd_sm90` hdim 64, 6 call/step.
   GEMM lớn nhất (`w4a8 mixed` tile 128×32×128, 44 call) chỉ 28.6%.
3. **elementwise/norm/act = 7.4%** dù chiếm **124/223 = 56% số launch** ⇒ fusion gỡ được hơn nửa
   số kernel nhưng **tối đa 7% thời gian**. Dưới ngưỡng plan (25%) ⇒ **không phải lever**.
4. **Gap = 4–7%** ⇒ **KHÔNG launch-bound**, cudagraph đang làm đúng việc. Boot log xác nhận
   `cudagraph_mode=FULL_AND_PIECEWISE`, decode chạy nhánh **FULL** (fact độc lập regime).

**Attention tuyến tính theo context** (2 điểm, mọi hạng mục khác **bất biến** — gemm 1.8746 vs
1.8749, elementwise 0.375 vs 0.375, đây là chứng cứ nội tại rằng phép đo sạch):

> **t_attn/layer-call = 56 µs + 0.100 µs × context_token**   (ctx 3953 → 452 µs; ctx 7954 → 853 µs)

**Quy về portal:** ở ctx ~3300 (workload thật) ⇒ attention rig = 6×387 µs = **2.32 ms**, phần
không-attention = **2.31 ms**. Tỉ lệ rig→portal lấy từ nhánh weight (portal 1.20 / rig 1.874 = 0.64)
⇒ attention portal ≈ **1.5 ms**, tức **~73% của sàn 2.05 ms** và **~46% của TPOT 3.25**.
⇒ **1.6 ms "chưa giải thích" chính là decode attention.** Không còn ẩn số nào đắt hơn.

**KV fp8 trên rig: chỉ −19% attention, không phải −50%** (385 µs @ctx 4210 so với 478 µs dự đoán
từ đường bf16; KV pool 1.17M → 2.34M token xác nhận cờ ăn thật, kernel đổi tile 64×192→64×128).
⇒ **Xác nhận trực tiếp giả thuyết còn sống ở §4.1**: attention bị chặn bởi **độ trễ / số giao dịch
trên slice ~18 SM**, không phải bề rộng phần tử. Giải thích trọn vẹn vì sao portal đo `k = 1.006`
và mọi biến thể KV-quant đều không mua được ERS. **Trục KV đóng vĩnh viễn, có cơ chế.**

**Hệ quả — mọi lever "sàn" đã cạn trong phạm vi cờ CLI:**
- fusion / `enable_noop` / `-O3` ⇒ chạm tối đa 7.4% (§8.3) — **không đáng lượt nộp**.
- `cudagraph-capture-sizes` dày quanh 24–32 ⇒ vô nghĩa, không launch-bound (§8.4).
- KV quant mọi dtype ⇒ đã có cơ chế bác bỏ.
- cascade attention (chia sẻ prefix 1000 tok cho cả batch — ~30% attention) **vẫn đóng**: nguồn
  0.25.1 cảnh báo *"No piecewise cudagraph for executing cascade attention… fall back to eager"*,
  trùng khớp `gpu_model_runner.py:3777` ở §4.4 và 22 failed đã đo. Mất cudagraph tốn hơn phần thắng.
⇒ **80 đòi một attention kernel khác cho shape (B=27, hdim 64, ~18 SM)** — việc build image, không
phải việc của cờ CLI. Trần cờ-CLI vẫn là **77** (STRATEGY §4), đường đi thực tế là TTFT.

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
