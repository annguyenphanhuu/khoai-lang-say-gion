# CHIẾN LƯỢC — KHOAI LANG SẤY GIÒN

> Số đo: `on-tap/KNOWLEDGE_BASE.md`. Luật: `README.md` (**chỉ vLLM**, **chỉ online quant**).
> Điểm cao nhất từng thấy **68.57** (K3) — **KHÔNG tái lập được**, xem §1. Hạn 30/07/2026.
> Cập nhật **2026-07-27 chiều** (sau ngày rig). Bản này **thay thế** mọi kết luận cũ về nhiễu,
> về trục KV, và về "sàn = attention" (§2b đã sửa). Số đo rig: `FINDINGS-2026-07-27-RIG.md`.

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

### 🔴 2b. SÀN = **CPU của EngineCore**, KHÔNG phải attention (sửa 27/07 chiều)

Bản trước kết luận "sàn 2.05 ms = decode attention" từ nsys. **Số nsys đúng, suy luận sai**: nó đo
tỉ lệ trong *thời gian GPU busy*, không đo GPU chiếm bao nhiêu phần của một step.
Đo lại bằng ERS harness trên rig, **client pin ra khỏi core 0-2** (mọi số rig cũ nhiễm lỗi này ⇒
TTFT bị thổi 94 vs 44 ms) — chi tiết `FINDINGS-2026-07-27-RIG.md`:

| Thí nghiệm | Số | Suy ra |
| :-- | :-- | :-- |
| `--no-async-scheduling` | TPOT 3.33 → 4.75 | async = `max(CPU,GPU)`, tắt = `CPU+GPU` ⇒ **{3.33, 1.42}** |
| **MPS 50% (cắt nửa SM)** | TPOT **không đổi** | GPU dư ≥2× ⇒ **3.33 là CPU**, 1.42 là GPU |
| quét RATE (B = 7.8/15.8/27) | `TPOT ≈ 1.60 + 0.064×B` | intercept = GPU ✓ · slope = **64 µs Python/req/step** |
| spy `VLLM::EngineCore` | `zero_block_ids` 18% · `copy_to_gpu` 17% · `mamba_get_block_table_tensor` 5% · `collect_mamba_copy_meta` 3% | **43% CPU/step = bookkeeping hybrid LFM2. 0% attention.** |

⇒ Mục tiêu đúng của trục TPOT là **cắt việc Python mỗi request mỗi step**, không phải thay kernel
attention. Cắt đôi attention chỉ đáng ~0.2 ms (không phải 0.75).

**Số nsys cũ vẫn giữ giá trị mô tả** (decode thuần B=27, MPS 14% ≈ 18 SM): attention 2.75 ms/step
= **54% GPU busy** (6 call `flash_attn_fwd_sm90` hdim64) · gemm 37% · elementwise **7.4%** ·
gap 4–7% ⇒ **không launch-bound**. `t_attn/call = 56 µs + 0.100 µs × ctx`.
KV fp8 chỉ cắt attention 19% ⇒ attention bị chặn bởi độ trễ/số giao dịch, không phải byte —
đây vẫn là **cơ chế** giải thích `k=1.006` của §3.1.

⇒ Hệ quả cứng **không đổi**: fusion (≤7.4%), cudagraph-sizes (không launch-bound), KV quant,
cascade (mất cudagraph ⇒ eager) — **không cái nào là lever**. Cái đổi là **đích ngắm**:
`zero_block_ids` / `copy_to_gpu` / block-table hybrid, chứ không phải kernel attention.

## 3. TRỤC ĐÃ ĐÓNG

**3.1 KV quantization — ĐÓNG THẬT.** fp8_e4m3 (4 lần), e5m2, turboquant 4bit, int8, nvfp4.
Đo được k=1.006. Đã loại bằng source: kernel `flash_fwd_hdim64_e4m3_paged_split_sm90` **có tồn tại**
trong `.so` (giả thuyết pad hdim 64→128 là **SAI**) · cache cấp phát thật 1 byte/phần tử ·
block_size không bị nhân đôi bởi hợp nhất page hybrid (mamba page LFM2 = 8192 B) ·
`calculate_kv_scales` bị ép tắt cho hybrid · `.view()` không copy · `_cudagraph_support=ALWAYS`
không phụ thuộc kv dtype. ⇒ KV đơn giản **không phải bottleneck**. Rút mọi ngân sách khỏi trục này.

**3.2 Các cờ cho ttft/scheduling.** `--no-scheduler-reserve-full-isl` là no-op đã chứng minh
(`kv_cache_manager.py:346` reserve theo ISL thật ~54MB; pool KV ~13 GB dùng 11% ⇒ không bao giờ chặn).
⚠️ **`--renderer-num-workers=2` KHÔNG đóng** — nó cùng ra ttft 55 với cái no-op nên hồi 26/07 bị xếp
nhầm vào đây; rig 27/07 cho thấy nó **thật** (mặc định là 1, Δrest −3.5 ms, xem §2b/§5).
Đóng thật: `block-size` pin · `gpu-mem 0.95` ·
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
- **80 KHÔNG khả thi** nếu không phá sàn. Weight đã cạn (int4 body + FP8 lm_head; lm_head int4
  chỉ thêm −0.11 ms = +0.85 ERS). KV chỉ còn ≤0.44 ms.
  ⇒ **80 = crack sàn, và sàn = CPU của EngineCore (§2b)**, cụ thể 64 µs Python/request/step ở
  `zero_block_ids` / `copy_to_gpu` / block-table hybrid. Vẫn phải build image — nhưng là **patch
  Python đo được trên rig**, không phải viết kernel attention. Trong phạm vi cờ CLI, **77 là trần
  cứng** và cờ đã cạn (còn +0.94 ERS, xem `compose-k3-frontend.yml`).

## 5. KẾ HOẠCH → `PLAN-2026-07-27.md` — **rig để A/B, portal chỉ để ăn điểm**

**Rig giờ nhạy hơn portal ~10×.** Với client pin ra khỏi core 0-2 và bỏ rep lạnh sau mỗi boot,
rig phân giải **Δ ≈ 0.3 ERS**; portal cần **≥4 ERS**. ⇒ **Không bao giờ A/B trên portal nữa.**

**Cổng nộp — tách làm hai, đừng gộp:**
- **Cổng A (tiêu lượt để HỌC): đóng vĩnh viễn.** Không đọc được Δ < 4 ERS.
- **Cổng B (tiêu lượt để ĂN ĐIỂM, best-of):** nộp khi ① rig cho Δ > 0 với hai bên **không chồng
  lấn** (≥2 boot, chỉ rep ấm), ② output-preserving + giữ FULL cudagraph, ③ flag/patch đã verify
  **trong chính image** bằng docker offline. **Không** đòi Δ ≥ 4 — 4 ERS là ngưỡng *đọc được*,
  không phải ngưỡng *đáng nộp*; best-of chỉ cần dịch cả phân phối lên.

**Quota theo NGÀY, không dồn** ⇒ lượt không dùng là lượt mất trắng ⇒ khi đã có config nghiêm ngặt
tốt hơn thì **nộp hết quota vào đúng nó**, không nộp đối chứng.
`compose-k3-frontend.yml` (K3 + `--disable-uvicorn-access-log` + `--renderer-num-workers=2`,
**+0.94 ERS** đo trên rig) đủ cổng B ⇒ P(phá 68.57) đi từ ~1.1% lên ~16% cho một ngày quota.

**Ngân sách điểm — 80 vẫn cần CẢ HAI nửa:** TTFT 43 = queue 0.0 + prefill 19 + **frontend 19**
(= tokenize lại ~2900 token mỗi turn) · TPOT 3.33 = GPU 1.42 + **CPU 64 µs/req/step**.
Hai đòn còn lại, cả hai là **patch Python đo được trên rig**: cache tokenization theo prefix
(**+4 ERS**) và cắt input-prep hybrid (**+5 ERS**).

⚠️ **`vllm.__version__` trong image portal trả `0.22.1`** dù source khớp 0.25.1 gần như từng dòng.
Mọi lập luận "nguồn 0.25.1" từ nay phải **grep trong chính image**, không grep bản pip trên rig.

## 6. QUY TẮC

1. Δ < 4 ERS với 1 run **portal** mỗi bên = **không kết luận được**. Đừng đóng/mở trục dựa vào nó.
   Trên **rig** thì ngược lại: 2 boot × 2 rep ấm mỗi bên đọc được tới **0.3 ERS** — mọi A/B về đây.
   ⚠️ Client harness **phải** `taskset -c 8-40` (server `0-2`) và **bỏ rep đầu sau mỗi boot**.
2. Giữ `VLLM_LOGGING_LEVEL=WARNING`, `--disable-log-stats`, `--max-model-len=32768`
   (hạ max-len có nguy cơ làm probe long-context fail = mất bài).
3. `failed` 4–6 là nền (~1.3%, ăn ~0.85 ERS). `>10` ⇒ dùng raw ERS, bỏ TPOT bóc.
4. Portal **không trả log**. Chỉ rig đọc được log, và chỉ tin rig cho fact độc lập regime.
5. `accuracy_drop=0` + `f_delta=1` ở **mọi** run ⇒ không tin. GPQA trên rig chỉ cần khi một đòn
   quantization đã thắng ERS vượt ngưỡng 4.
6. Pin digest image lúc nộp; luôn tag mới.
