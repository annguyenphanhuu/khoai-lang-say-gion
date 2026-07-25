# KNOWLEDGE BASE — LFM2.5 trên MiG H200 (SSOT)

> Cập nhật 2026-07-25 (đã dọn). Luật thi: `README.md` — **chỉ vLLM**, **chỉ online quantization**.
> Nguyên tắc: mỗi dòng dưới đây là **số đo** hoặc **source code đã đọc**. Không giữ giả thuyết.
> Best portal: **65.06** (`compose-w4a8.yml`) / **64.67** (`compose-cpu-lean.yml`).

---

## 1. Phần cứng & model

**Portal:** 1 slice **MiG H200 1g.18gb** — 18GB VRAM, **~0.6 TB/s HBM**, ~1/7 SM, cap 9.0,
**3 CPU core**, 8GB RAM. Ubuntu 24.04, driver 590.x / CUDA 13.x.

**Model** `LFM2.5-1.2B-Instruct` (theo `config.json`): 16 layer = **10 short-conv + 6 GQA**.
`hidden=2048`, `intermediate=12288`, 32 head / 8 KV head, `head_dim=64`, `vocab=65536`,
`conv_L_cache=3`, `conv_dim=2048`. Mọi in/out feature đều chia hết 128.

Byte weight (1.2B param): BF16 2.4GB · FP8 1.2GB · int4-g128 ~0.6GB.
`lm_head` = 65536×2048 = **134M param** ⇒ BF16 268MB/step, FP8 134MB/step.

---

## 2. Hàm điểm (mọi quyết định dựa vào đây)

`s(x) = clamp((C−x)/(C−F), 0, 1)²` · `ERS = mean_req(50·(s_ttft + s_tpot))`
TTFT: F=10ms C=400ms · TPOT(=tbt): F=1ms C=10ms · S=0 nếu request lỗi.

**Độ nhạy tại điểm hiện tại:** TPOT **~7.65 điểm/ms** · TTFT **~0.23 điểm/ms** ⇒ TPOT đắt **~33×**.

| Điểm vận hành | s_ttft | s_tpot | ERS |
| :-- | --: | --: | --: |
| Hiện tại (TTFT 48ms, TPOT 3.77ms) | 0.815 | 0.479 | **64.7** |
| TTFT → 10ms (bất khả), TPOT giữ | 1.000 | 0.479 | 74.0 |
| TTFT 25 + TPOT 3.17 | 0.925 | 0.575 | 75.0 |
| TTFT 20 + TPOT 2.66 | 0.935 | 0.665 | **80.0** |

⇒ **TTFT một mình không thể tới 80 (trần 74).** 80 đòi TTFT ~20ms **VÀ** TPOT ~2.7ms cùng lúc.

---

## 3. Workload & điểm vận hành (đo, không đoán)

`grading-workload-spec.json`: 70 hội thoại × 6 turn = **420 request đều được chấm**;
shared system prefix 1000 tok + per-conv prefix 1000 tok + 150 user/turn; **output pin 300**; Poisson seed 42.

- Context ~2150 (turn 1) → ~4400 (turn 6). Prefix-cache ăn thật từ turn 2.
- **Turn 1 phải prefill ~1150 token MỚI** (conv-prefix 1000 + user 150; chỉ system-prefix 1000 được cache
  chung). Turn 2–6 chỉ ~150 token. ⇒ 1/6 số request có prefill nặng gấp ~8×.
- **Batch decode thật = 27–31** (đo `vllm:num_requests_running`). Con số "3–15" trong tài liệu cũ là SAI.
- **Batch do TẢI quyết định, không phải knob:** throughput output ≈ 8.1 tok/ms ⇒ `B ≈ 8.1 · TPOT(ms)`.
  Fit `TPOT ≈ 1.585 + 0.073·B` giải đồng thời → TPOT 3.87 / B 31 = khớp số đo.
  **Hệ quả 1:** ΔTPOT tự khuếch đại **~2.4×** (giảm phần batch-invariant thì batch tụt theo).
  **Hệ quả 2:** hạ `--max-num-seqs` là **vô nghĩa** — chặn admission chỉ dồn queue, TTFT nổ.

---

## 4. Kết quả portal (oracle duy nhất hợp lệ)

Portal làm tròn `tbt` về số nguyên ⇒ **đọc ERS, không đọc tbt** khi so hai bài gần nhau.
6–7 failed/420 xuất hiện ở MỌI run trong ngày = variance portal (connection/HTTP), không phải lỗi config.

| Config | ERS | tbt | ttft p50 | failed | acc_drop |
| :-- | --: | --: | --: | --: | --: |
| **`compose-w4a8.yml`** | **65.06** | 3 | 50 | 6 | **0** |
| **`compose-cpu-lean.yml`** | **64.67** | 3.8 | 45 | 0 | — |
| `compose-fastsse-lean.yml` | 63.40 | — | — | — | — |
| `compose-fullfp8-v0251.yml` | 62.92 | — | — | — | — |
| `compose-full-fp8.yml` | 62.50 | 3.8 | 48 | 0 | — |
| w4a8 + interactivity (đã xoá) | 62.31 | 3 | 61 | — | — |
| kv-cache-dtype fp8 trên nền fp8 (đã xoá) | 59.62 | 4 | 62 | 6 | 0 |
| max-model-len 6144 "rightsize" (đã xoá) | 60.82 | 4 | 62 | 7 | 0 |
| BF16 anchor (đã xoá) | 48.17 | **6** | 56 | 7 | 0 |
| diag-eager, tắt CUDA graph (đã xoá) | 33.93 | **12** | — | — | — |
| bitsandbytes NF4 (đã xoá) | 30.8 | 15 | 84 | 9 | **0** |
| online W4A16 / tinygemm (đã xoá) | 22.51 | 6 | 205 | — | — |

**Hai slope quan trọng rút từ bảng này:**
1. **BF16 → FP8 = tbt 6 → 3.8 (−2.2ms).** ⇒ trên MiG, decode **BỊ CHẶN BỞI BĂNG THÔNG ĐỌC WEIGHT**.
2. **Tắt CUDA graph = tbt 3 → 12.** ⇒ graph đang che ~9ms launch overhead; đừng đụng vào graph.
3. **4-bit QUA ĐƯỢC Accuracy Gate** (BnB-NF4 và W4A8 đều `acc_drop=0`) ⇒ accuracy không phải rào cản của int4;
   chỉ kernel là rào cản.

---

## 5. ⚠️ Rig H100 (RunPod) đo SAI regime — không dùng làm oracle băng thông

Rig chạy `--gpu-memory-utilization 0.225` + `taskset -c 0-2` để "giả lập MiG". Nhưng flag đó
**chỉ giới hạn dung lượng**, KHÔNG giới hạn SM hay băng thông:

| | Portal MiG H200 1g | Rig H100 SXM |
| :-- | --: | --: |
| HBM bandwidth | ~0.6 TB/s | **3.35 TB/s (5.6×)** |
| SM | ~1/7 GPU | **132 (toàn bộ, ~7×)** |
| Đọc weight FP8 1.2GB | **2.0 ms** | 0.36 ms |

⇒ Trên rig, weight-read chỉ ~10% TPOT; trên portal >50%. **Mọi phán quyết về đòn BĂNG THÔNG đo trên
rig đều vô giá trị** (int4, fp8-KV, kinh tế của spec decode). Việc rig cho `ERS_HIST 66 ≈ portal 64.67`
là **trùng số của hai bottleneck khác nhau**, không phải bằng chứng rig faithful — đây chính là cái bẫy.

**Rig CHỈ dùng được cho:** correctness (temp=0 diff), acceptance rate của spec, tỉ lệ GPU-time *tương đối*
giữa các kernel cùng loại. **Chốt điểm: chỉ portal.** Portal không giới hạn số lần submit online
(chỉ *chọn* 5 bài ở cuối) ⇒ portal là oracle rẻ nhất, đừng thuê GPU để đoán MiG.

---

## 6. ĐÃ ĐÓNG — đừng thử lại (kèm root cause)

| Đòn | Trạng thái | Root cause |
| :-- | :-- | :-- |
| `--async-scheduling` explicit | null | v0.22.1+ đã bật mặc định khi compatible |
| FlashInfer backend | 56.80 | prefill chậm; giữ FlashAttention (default Hopper) |
| `--max-cudagraph-capture-size=128` | failed req | assertion mamba-cache (PR #34571) |
| Custom ShortConv kernel | ~0 gain | 10 conv layer đã nằm TRONG Full CUDA Graph (`mamba_attn.py`, `UNIFORM_BATCH`) |
| `fuse_norm_quant` / `fuse_act_quant` | đã default-on | vllm ≥ 0.25.1 log config xác nhận |
| Tối ưu CPU input-path (P3) | vô ích | tbt 3-core ≈ 8-core ⇒ GPU-bound, async đã che CPU |
| `--max-model-len` right-size | 60.82 | KV pool do `gpu-memory-utilization` quyết định, không do max-len |
| `--performance-mode=interactivity` | 62.31 | ttft phình lên 61 |
| Online W4A16 / tinygemm | 22.51 | overhead dequant nổ ở prefill |
| bitsandbytes NF4 | 30.8 | kernel BnB eager, ngoài CUDA graph |
| `draft_model` spec (mọi draft LFM2) | boot crash | `validate_same_kv_cache_group` có `assert len(groups)==1`; draft LFM2 hybrid ⇒ 2 kv-group. Không flag nào sửa |
| n-gram / prompt-lookup spec | acceptance ≈ 0 **+** correctness FAIL 12/12 | (a) workload là chat tự do, không có gì để copy từ context; (b) `short_conv.py` update `conv_state` qua `state_indices_tensor_p`, **không có nhánh rollback theo num_accepted** ⇒ token-duplication. `--mamba-cache-mode all` không cứu (bị loại trừ với `--enable-prefix-caching`, và khi bật thật thì chỉ rollback SSM-state chứ không rollback conv-state) |
| Hạ `--max-num-seqs` | vô nghĩa | batch do tải quyết định (§3); chặn admission ⇒ queue nổ |
| **SGLang / TensorRT-LLM** | **PHẠM LUẬT** | `README.md` §3: chỉ được dùng vLLM |
| MSE-clip cho int4 | vô ích | cùng số bit ⇒ cùng byte traffic; accuracy đã pass sẵn |

**KHÔNG đóng (tài liệu cũ đóng sai, xem §5):** int4/W4A8 hạ tbt · `--kv-cache-dtype fp8` ·
"tbt=3ms là op-floor vật lý". Ba kết luận này đều suy từ rig H100 sai regime.

---

## 7. Lever còn mở

> **LUẬT VÀNG (từ mô hình byte/step ở `STRATEGY.md` §1):** decode gần như 100% bandwidth-bound
> (2.29 GB/step ÷ 0.6 TB/s = 3.82ms ≈ đo 3.8ms). ⇒ **Tối ưu chỉ đáng làm nếu nó DI CHUYỂN ÍT BYTE HƠN.**
> Giảm FLOP / giảm số launch / fuse op = **0 điểm**, vì GPU đang đứng chờ HBM.


1. **`lm_head` đang là BF16 trong `compose-w4a8.yml`** — [`online_w4a8.py`](../image-w4a8/online_w4a8.py)
   `get_quant_method` trả `None` cho mọi thứ không phải `LinearBase`, còn patch FP8 ở
   [`patch_full_fp8.py`](../image-full-fp8/patch_full_fp8.py) chỉ sửa dispatch của *config fp8*
   (`isinstance(layer, (LinearBase, ParallelLMHead))`). ⇒ W4A8 **thêm lại 134MB/step** so với
   cpu-lean FP8 ≈ 0.22ms thuần lãng phí. Fix = 1 dòng.
2. **KV bandwidth ở điểm vận hành thật.** Batch 27 × ctx ~3300 × 6 attn layer × 2 × 8 kvhead × 64 ×
   2 byte ≈ **1.0 GB/step ≈ 1.7ms** trên 0.6 TB/s — cùng cỡ weight-read, **không phải "0.2–0.4ms"**
   như roofline cũ (roofline đó tính cho batch 3–15 = điểm vận hành sai).
3. **Đuôi TTFT turn-1.** mean 48ms > p50 45ms + cấu trúc §3 ⇒ nếu turn2+ ≈ 15ms thì turn-1 đang ở
   **~300–350ms, s_ttft ≈ 0 cho 1/6 request**. `bench/ers_harness.py` đã in sẵn `turn1 p50 / turn2+ p50`
   — **chưa ai ghi lại con số này**. Đây là phép đo rẻ nhất còn lại.
4. Custom CUDA/Triton kernel fuse op hybrid — **hợp lệ** theo luật, nhưng theo LUẬT VÀNG ở trên thì
   **vô ích** (không giảm byte). Kernel chỉ đáng viết trên trục KV.

---

## 8. ĐỌC SOURCE trong image thi đấu `annguyenphanhuu/vllm-w4a8:online-r1` (vllm 0.22.1)

> Docker local đã có đúng image (`afa562d714e1`, khớp digest pin trong `compose-w4a8.yml`) ⇒ đọc source
> trực tiếp, không cần đoán, không cần thuê GPU:
> `docker run --rm --entrypoint bash <image> -c 'grep ... /usr/local/lib/python3.12/dist-packages/vllm/...'`

### 8.1 ✅ FA3 tiêu thụ fp8-KV NATIVE — không có dequant pass

`v1/attention/backends/flash_attn.py:753-756` khi `is_quantized_kv_cache()`:
`key_cache = key_cache.view(fp8_dtype)` — là **`.view()`, KHÔNG copy/dequant** — rồi truyền
`q_descale/k_descale/v_descale` vào kernel FA3. `supports_kv_cache_dtype` (L183-193) cho `fp8/fp8_e4m3`
**chỉ khi** `get_flash_attn_version()==3` **và** `is_device_capability_family(90)` ⇒ **đúng trên MiG H200**.

⇒ **Giả thuyết "backend dequant fp8→BF16 nên không tiết kiệm byte" là SAI.** Cơ chế đọc-nửa-byte CÓ SẴN,
không cần viết kernel Triton. Nguyên nhân regression 59.62 nằm ở chỗ khác — xem 8.3.

`cache_config.calculate_kv_scales` default **False** + đã deprecated ⇒ k/v scale = 1.0 (lấy từ checkpoint
nếu có). Không phải vấn đề hiệu năng, và khớp với `acc_drop=0` đã đo.

### 8.2 ❌ Cascade / shared-prefix attention TỰ TẮT FULL CUDA GRAPH ⇒ ĐÓNG

`v1/worker/gpu_model_runner.py:3777`:
```python
cudagraph_mode, batch_descriptor = dispatch_cudagraph(
    num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output)
#   -> invalid_modes={CUDAGraphMode.FULL} if disable_full else None
```
Điều kiện gate (`flash_attn.py:1054`) thì workload này **thoả hết**: `common_prefix_len ≥ 256` (ta có ~992),
`num_reqs ≥ 8` (ta có 27), không alibi/sliding-window; sau đó là một perf model thô so với FlashDecoding.

**Nhưng bật cascade = mất FULL cudagraph.** Đo portal: tắt graph ⇒ tbt 3 → **12ms**. Mà cascade chỉ tiết kiệm
~0.5ms KV traffic. ⇒ **lỗ nặng. K1 ĐÓNG ở stock vLLM 0.22.1.** Muốn mở phải làm cascade capture được
full-graph (2 kernel + metadata động) = đúng nghĩa research, không làm trong deadline.

### 8.3 🔍 Nghi phạm mới cho regression fp8-KV (tbt 4, ttft 62, 6 failed)

Model hybrid ⇒ vLLM phải **unify page size** giữa 6 layer GQA và 10 layer ShortConv:
`v1/core/kv_cache_utils.py:1012` `unify_kv_cache_spec_page_size` — *"unify the page size ... raise
NotImplementedError if failed"*, L1042 *"Cannot unify by adjusting block_size"*, L1551 *"Unify page size by
padding layers' page_size to the nearest larger page_size"* (`page_size_padded`).

Đổi KV sang fp8 **làm page của layer attention giảm một nửa** (16×8×64×2×2B = 32KB → 16KB) trong khi page
conv-state không đổi ⇒ **vLLM có thể đổi `block_size` hoặc pad để unify**. Block size to hơn ⇒ **granularity
prefix-cache thô hơn** ⇒ turn 2–6 match kém ⇒ **đúng triệu chứng ttft 45→62 mà tbt không giảm**.

**Phép đo quyết định (miễn phí, đọc log boot):** so `block_size` và dòng **`GPU KV cache size: N tokens`**
giữa run fp8-KV và run baseline. Nếu N **không tăng ~2×** ⇒ unification đã ăn hết phần tiết kiệm.
Đòn thử: pin `--block-size` tường minh (16 / 32) cùng `--kv-cache-dtype fp8`.
