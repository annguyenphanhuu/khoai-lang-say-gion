# CHIẾN LƯỢC — KHOAI LANG SẤY GIÒN

> Số đo: `on-tap/KNOWLEDGE_BASE.md`. Luật: `README.md` (**chỉ vLLM**, **chỉ online quant**).
> Điểm cao nhất từng thấy **68.57** (K3) — **KHÔNG tái lập được**, xem §1. Hạn 30/07/2026.
> Cập nhật 2026-07-26 sau 20 submit. Bản này **thay thế** mọi kết luận cũ về nhiễu và về trục KV.

---

## 0. LUẬT OUTPUT-PRESERVING (không bao giờ vi phạm)

Protocol có **probe long-context riêng**, ngoài phần chấm ERS. `LFM2_SWA=1024` bị
`protocol aborted: long-context probe failed (0%)` ⇒ **mất cả bài**, không phải trừ điểm.
⇒ Mọi đòn làm model **nhìn thấy ít hơn / trả lời khác đi** đều bị chặn: sliding-window, cắt/nén
context, KV eviction, dual-path, `--stream-interval>1` (đổi luồng bị bấm giờ).
Hợp lệ: quantization, kernel, scheduling, frontend. Trước mỗi submit hỏi: *model có nhìn thấy ít hơn?*

## 1. 🔴 MÁY ĐO CÓ HAI TRẠNG THÁI CÁCH NHAU ~5.7 ERS

**Chứng minh cứng** (không cần giả thuyết): `--no-scheduler-reserve-full-isl` là **no-op đã verify
trong source** (§3.1) ⇒ slot 1 và slot 3 ngày 26/07 là **CÙNG MỘT CHƯƠNG TRÌNH**, cùng workload.
Chúng ra **59.74** và **65.46**.

- Trong cùng một trạng thái, máy đo **rất chính xác**: 5 run candidate có **sd 0.52, spread 1.29**;
  hai run K3 khớp nhau **0.07**. σ_trong-trạng-thái ≈ 1.0.
- Giữa các trạng thái: nhảy **~5.7 ERS**. Không do thứ tự nộp (K3 ở slot 6 cuối ngày = 59.67,
  hiệu ứng thứ tự **−0.07**) ⇒ giả thuyết cold-start **bị loại ở 3.7σ**.
- **68.57 cách trung bình 59.7 tới 7.6σ.** Cùng file (md5 giống hệt), cùng image digest, ba kết quả.
  ⇒ Hạ cấp 68.57 xuống "một lần rút may ở trạng thái tốt". **Đừng nộp K3 mà kỳ vọng 68.57.**

**Hệ quả vận hành:**
1. Ngưỡng phân giải 1-run-vs-1-run là **~4 ERS**. Δ < 2.5 không thể mua bằng ngân sách còn lại.
2. Điểm bảng = **best-of** ⇒ phương sai là bạn. Nộp lặp cấu hình tốt nhất **là** chiến lược điểm.
3. Bỏ hẳn thiết kế "cắm mốc xen kẽ khử drift" — không có drift, chỉ có nhảy trạng thái.
4. Khi `failed > 10`, dùng **raw ERS** (đó mới là điểm chấm) và **bỏ TPOT bóc ra**: 21 lỗi của
   slot 6 là phần đuôi chậm, hiệu chỉnh `×420/(420−failed)` thổi lên +2.5 ERS sai.

## 2. 🔴 VẬT LÝ ĐÚNG CỦA TPOT — mô hình cũ SAI

Hồi quy trên 4 điểm khác nhau về weight (BF16 / full-FP8 / W4A8-lmheadBF16 / K3):

> **TPOT = 2.05 ms + 1.63 ms/GB × weight_GB**  ·  R² = 0.986  ·  slope = **613 GB/s ≈ băng thông
> danh nghĩa 600 GB/s** (weight đọc ở ~100% hiệu suất)

Với K3 (weight 0.734 GB): weight **1.20 ms** + **sàn cố định 2.05 ms** = 3.25 ✓ (đo 3.25–3.47).

**Đo trực tiếp KV:** `--kv-cache-dtype=fp8_e4m3` cho hệ số byte KV còn lại **k = 1.006** thay vì 0.50
⇒ **toàn bộ việc đọc KV tốn ≤ 0.44 ms ≤ 13% của step**, không phải 48–60% như tài liệu cũ tính.
Con số 1.09 GB KV/step là **phép tính**, không phải phép đo — và nó sai.

⇒ **Cơ cấu thật: sàn 2.05 (61%) · weight 1.20 (36%) · KV ≤0.44 (≤13%).**

### ✅ 2b. SÀN ĐÃ ĐƯỢC ĐỊNH DANH (nsys 27/07, KB §8) — **sàn = decode attention**

Giả thuyết cũ ("latency × 80–100 kernel") **SAI ở phần cơ chế**: có tới **223 kernel/step**, nhưng
**gap chỉ 4–7%** ⇒ **không launch-bound**. Đo decode thuần B=27 trên rig (MPS 14% ≈ 18 SM):

| | /step | ms/step | % |
| :-- | --: | --: | --: |
| **attention** (6 call, `flash_attn_fwd_sm90` hdim64) | 19 | **2.75** | **54%** |
| gemm | 65 | 1.87 | 37% |
| elementwise/norm/act | 124 | 0.375 | **7.4%** |

`t_attn/call = 56 µs + 0.100 µs × ctx`. Quy về portal (hệ số 0.64 lấy từ nhánh weight) ở ctx 3300
⇒ attention ≈ **1.5 ms ≈ 73% của sàn 2.05**. **1.6 ms ẩn số = attention.** Hết ẩn số.

**KV fp8 trên rig chỉ cắt attention 19%** (không phải 50%) ⇒ attention bị chặn bởi **độ trễ / số
giao dịch trên ~18 SM**, không phải byte. Đây là **cơ chế** giải thích `k=1.006` của §3.1.

⇒ Hệ quả cứng: fusion (≤7.4%), cudagraph-sizes (không launch-bound), KV quant (có cơ chế bác bỏ),
cascade (mất cudagraph ⇒ eager, §3.2) — **tất cả đều không phải lever**. 80 đòi **kernel attention
khác cho shape B=27/hdim64/18 SM** = phải build image, không phải cờ CLI.

## 3. TRỤC ĐÃ ĐÓNG

**3.1 KV quantization — ĐÓNG THẬT.** fp8_e4m3 (4 lần), e5m2, turboquant 4bit, int8, nvfp4.
Đo được k=1.006. Đã loại bằng source: kernel `flash_fwd_hdim64_e4m3_paged_split_sm90` **có tồn tại**
trong `.so` (giả thuyết pad hdim 64→128 là **SAI**) · cache cấp phát thật 1 byte/phần tử ·
block_size không bị nhân đôi bởi hợp nhất page hybrid (mamba page LFM2 = 8192 B) ·
`calculate_kv_scales` bị ép tắt cho hybrid · `.view()` không copy · `_cudagraph_support=ALWAYS`
không phụ thuộc kv dtype. ⇒ KV đơn giản **không phải bottleneck**. Rút mọi ngân sách khỏi trục này.

**3.2 Các cờ cho ttft/scheduling.** `--renderer-num-workers=2` và `--no-scheduler-reserve-full-isl`:
cùng ra ttft 55, và cái sau là no-op đã chứng minh (`kv_cache_manager.py:346` reserve theo ISL thật
~54MB; pool KV ~13 GB dùng 11% ⇒ không bao giờ chặn). `block-size` pin · `gpu-mem 0.95` ·
`max-num-batched-tokens 2048` · `max-model-len` · `performance-mode` · cascade (22 failed) ·
FlashInfer · sliding-window (**abort**) · mọi spec decode · custom ShortConv/fuse.
`--api-server-count` **no-op** trên entrypoint bị ép; Rust frontend **không có** trong image.

**3.3 Cẩn trọng với image fastsse (r6).** Slot 7 có `tokens_per_sec` 0.0395 thay vì ~0.057
(thiếu 31%, cv của trường này ở 6 run kia chỉ 0.27%) ⇒ **cờ đỏ output-preserving chưa loại trừ**.
Đừng dùng r6 làm nền cho bài chốt tới khi hiểu con số đó.

## 4. TRẦN VÀ ĐƯỜNG TỚI 80

| Cần đạt | ttft 10 (sàn) | ttft 20 | ttft 30 | ttft 40 |
| :-- | --: | --: | --: | --: |
| **TPOT ≤ để có 80** | 2.90 | 2.61 | 2.35 | 2.10 |
| **TPOT ≤ để có 75** | 3.50 | 3.19 | 2.90 | — |

TPOT hiện tại **3.25**. ⇒ **Trần tuyệt đối của kiến trúc hiện nay là 77.0** (ttft về sàn 10 ms).

- **75 khả thi**: chỉ cần ttft 54 → ~20 ms, TPOT giữ nguyên.
- **80 KHÔNG khả thi** nếu không phá sàn 2.05 ms. Weight đã cạn (int4 body + FP8 lm_head;
  lm_head int4 chỉ thêm −0.11 ms = +0.85 ERS, dưới ngưỡng phân giải). KV chỉ còn ≤0.44 ms.
  ⇒ **80 = crack sàn, và sàn = decode attention (§2b).** Không có đường khác — và đường đó
  **không mở được bằng cờ CLI**, phải thay kernel attention (build image). Trong phạm vi cờ,
  **77 là trần cứng**; mục tiêu vận hành đúng là **75–77 qua TTFT**.

## 5. KẾ HOẠCH → `PLAN-2026-07-27.md` — **rig để quyết định, portal chỉ để chốt bài**

Track A đã xong (§2b). Kế hoạch 27/07 viết lại: **0 lượt nộp**, toàn bộ là rig.

**Vì sao không nộp.** Mô hình điểm đã kiểm chứng (TPOT 3.25 + ttft 47 ⇒ ERS 68.4 tính vs 68.57 đo).
Kỷ lục 68.57 **đã nằm trong két** (best-of). Một lượt ở trạng thái tốt (65.5, σ=1.0) muốn vượt nó
phải đi +3.07σ ⇒ p≈0.0011 ⇒ **15 lượt chỉ có ~1.1% cơ hội phá kỷ lục**. Cộng mọi Δ dưới 4 ERS đều
không đọc được ⇒ **portal không còn dùng để A/B được nữa**. Đừng nộp để thăm dò.

**Ngân sách điểm — 80 cần CẢ HAI nửa, không nửa nào đủ một mình:**
cắt nửa attention (TPOT 3.25→2.50) = **+6.5 ERS** · kéo ttft 50→20 = **+7.2 ERS** · cả hai = **82**.

**TTFT là trục có dụng cụ tốt nhất.** ttft ~50 ms trong khi prefill turn 2–6 chỉ ~150 token
(~3–6 ms compute) ⇒ **~45 ms là hàng đợi/scheduler/frontend, bị chặn bởi 3 CPU core** — mà
`taskset -c 0-2` trên rig tái lập **đúng** ràng buộc đó (khác băng thông, sai 5.6×, KB §5).
⇒ Bóc TTFT bằng histogram `queue_time / prefill_time / ttft` là việc ưu tiên 1, không cần profiler.

**Ứng viên còn sống duy nhất ở tầng env var:** `VLLM_FLASH_ATTN_VERSION=2` và
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` — attention chiếm 54% và chưa ai đo FA2/Triton trên shape
(B=27, hdim 64, 18 SM). Đo bằng `bench/prof_all.sh` (tỉ lệ cùng máy ⇒ độc lập regime).

**Cổng nộp:** chỉ tiêu lượt khi rig cho ≥15% trên thành phần trội **và** quy đổi ra Δ ≥ 4 ERS
**và** đòn đó output-preserving + không mất FULL cudagraph. Đủ ⇒ nộp 2 lượt cùng config.

## 6. QUY TẮC

1. Δ < 4 ERS với 1 run mỗi bên = **không kết luận được**. Đừng đóng/mở trục dựa vào nó.
2. Giữ `VLLM_LOGGING_LEVEL=WARNING`, `--disable-log-stats`, `--max-model-len=32768`
   (hạ max-len có nguy cơ làm probe long-context fail = mất bài).
3. `failed` 4–6 là nền (~1.3%, ăn ~0.85 ERS). `>10` ⇒ dùng raw ERS, bỏ TPOT bóc.
4. Portal **không trả log**. Chỉ rig đọc được log, và chỉ tin rig cho fact độc lập regime.
5. `accuracy_drop=0` + `f_delta=1` ở **mọi** run ⇒ không tin. GPQA trên rig chỉ cần khi một đòn
   quantization đã thắng ERS vượt ngưỡng 4.
6. Pin digest image lúc nộp; luôn tag mới.
