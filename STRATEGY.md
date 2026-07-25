# CHIẾN LƯỢC TỔNG THỂ — KHOAI LANG SẤY GIÒN

> **Single Source of Truth về Tri thức chi tiết:** `on-tap/KNOWLEDGE_BASE.md`
> **Mục tiêu:** Quản lý rổ 5 bài nộp an toàn tuyệt đối trước 30/07/2026.
> **Cập nhật 2026-07-24:** audit lại evidence và thay roadmap 80+; không coi giả thuyết
> floor/ceiling là kết quả đo.
>
> ### 🚩 CẬP NHẬT 2026-07-25 (session H100) — ĐỌC TRƯỚC
> Đã đo trực tiếp trên H100 (đầy đủ ở KB §5.10-12, `HANDOFF-RUNPOD.md`):
> - **tbt=3ms = op-floor GPU vật lý.** MỌI đòn hạ tbt cấp config/kernel-quant ĐÃ CẠN, ĐỪNG thử lại:
>   int4/W4A8 mọi kernel (Machete=0.70x fp8, đo graph sạch) · CPU input-path (3-core=8-core, GPU-bound)
>   · fuse_norm/act_quant (đã default-on). ⇒ **§3 bên dưới (P2/P3/P5) coi như ĐÓNG**, chỉ giữ làm lịch sử.
> - **Đường 80+ DUY NHẤT = speculative decoding** (hạ effective-tbt = forward/accepted). Kế hoạch thực thi
>   chi tiết: **`on-tap/NEXT-80.md`** (n-gram/prompt-lookup → EAGLE-3, gate correctness_diff + acceptance).
> - Rổ 5 bài an toàn (§4) GIỮ NGUYÊN. Được build image từ **vllm 0.25.1** (luật 3), không khóa fork cũ.
>
> ### 🚩 CẬP NHẬT 2026-07-25 (session H100 #2, đo trực tiếp trên pod) — SPEC NGRAM ĐÓNG
> Đã chạy trọn Bước 0 correctness-gate của NEXT-80 trên pod H100, isolate sạch từng biến:
> - **ngram spec KHÔNG lossless trên LFM2 hybrid ở vllm 0.25.1** — G0 FAIL 12/12 (temp=0), triệu chứng
>   token-duplication ("as as", "modes modes"). Đây là **bug rollback conv-state của ShortConv**, KHÔNG phải nhiễu:
>   determinism-check cùng config back-to-back = PASS ⇒ cổng hợp lệ; spec vs golden cùng-base (chỉ khác spec) = FAIL.
> - **`--mamba-cache-mode all` KHÔNG cứu được:** (a) nó bị **loại trừ với `--enable-prefix-caching`** (vLLM tự tụt về
>   `align` ⇒ không cấp buffer rollback); (b) khi tắt prefix-cache để `all` bật thật, mamba SSM-state được rollback
>   (`preprocess_mamba_all_specdec`/`postprocess_mamba_align_gpu`) **nhưng conv-state của ShortConv thì KHÔNG** →
>   vẫn corrupt. Ngoài ra base no-prefix-cache tự lệch output vs baseline prefix-cache (prefix-cache hybrid = experimental).
> - **Hệ quả:** cả ngram LẪN EAGLE (đều verify k-token qua target-forward, dính chung ShortConv conv-state) đóng ở
>   tầng config. Muốn mở 80+ spec ⇒ **phải patch source vLLM** snapshot/restore ShortConv conv_state theo num_accepted
>   (kernel + rebuild image, rủi ro cao, + no-prefix-cache đánh đổi TTFT). ⇒ Khuyến nghị: **chốt rổ an toàn ~65 (§4)**.

> ### 🚩 CẬP NHẬT 2026-07-25 (session H100 #3 — ĐO CONCURRENCY THẬT LẦN ĐẦU) — ĐỌC TRƯỚC
> Mọi bench cũ (`loadgen_c1.py`, `wl_probe.py`) đều **concurrency=1**; portal chấm **70 hội thoại
> đồng thời Poisson**. Đã viết `bench/ers_harness.py` (harness đồng thời trung thực, chấm ERS bằng
> đúng công thức từ **histogram server-side vLLM** — miễn artifact client) và đo trên pod H100:
> - **Calibrate:** baseline fp8+prefix-cache ở RATE=8 (batch decode mean **27**) ⇒ **ERS_HIST 66** ≈
>   portal 64.67. ⇒ điểm vận hành portal ≈ **batch 27**, KHÔNG phải "3–15" như KB đoán.
> - **Sweep scheduler — KHÔNG knob nào thắng default:** max-num-batched-tokens {1024..8192}, max-num-seqs 128,
>   long-prefill-threshold, kv-cache-dtype fp8 → tất cả ≤ baseline (đa số làm TTFT phình 80–88ms).
>   vLLM v1 default đã tối ưu. ⇒ **giả thuyết "tune scheduling = điểm lossless" BÁC BỎ bằng số đo.**
> - **PHÁT HIỆN LỚN — TPOT TĂNG theo batch:** batch 8→2.17ms, 16→2.67ms, 27→3.6ms. ⇒ ở batch cao
>   (điểm portal), decode **KHÔNG thuần weight-bound** mà pha compute + KV-read. **Hệ quả phá 2 hướng "80+":**
>   (a) **4-bit weight vô dụng** (weight không còn là hạng duy nhất); (b) **speculative decoding vô dụng/hại**
>   — spec THÊM compute (verify k×batch) vào step đã tải ⇒ TPOT tệ hơn. Cả `NEXT-80.md` lẫn P5 W4A8 dựa trên
>   giả định batch-thấp/weight-bound ⇒ **sai tiền đề dưới concurrency thật.**
> - **⇒ Trần config-lossless ~65–66 XÁC NHẬN bằng bằng chứng mạnh.** Đường 75+ KHÔNG nằm ở flag/quant/spec;
>   chỉ còn = **hạ per-token compute của hybrid ở batch cao** (fuse kernel hoặc đổi stack TensorRT-LLM/SGLang)
>   — dự án lớn, không phải cấu hình. Bài nộp tốt nhất vẫn là cpu-lean 64.67 / w4a8 65.06.

---

## 1. Nhật Ký Kết Quả Thực Nghiệm (Cập nhật 24/07/2026)

| Cấu hình | ERS Score | Failed | TTFT p50 | TBT Median | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`compose-cpu-lean.yml`** | **64.67** | **0** | **45 ms** | **3.8 ms** | **BEST KNOWN SCORE** |
| `compose-full-fp8.yml` | 62.50 | 0 | 48 ms | 3.8 ms | Control Baseline |
| `compose-fastsse-lean.yml` | 63.40 | 0 | — | — | + fastsse SSE + renderer=3 |
| `compose-fullfp8-v0251.yml` | 62.92 | — | — | — | Version fallback v0.25.1 |
| BF16 submission lịch sử | ~50.0 | 0 | — | — | Score đã đo; artifact BF16 hiện cần khôi phục/xác minh |
| ~~opt80-cudagraph / flashinfer / pure-cudagraph / shortconv~~ | 56–64 | 6–7 | — | — | ❌ ĐÃ XÓA — giả thuyết sai (xem KB §5) |
| `compose-opt-draft-spec.yml` | **N/A** | boot fail | — | — | ❌ CLOSED — EngineCore init crash (draft hybrid, assert kv-group; KB §5.9) |
| ngram spec (fp8 + `--speculative-config ngram`) | **N/A** | G0 FAIL | — | — | ❌ CLOSED — boot OK nhưng correctness FAIL 12/12 (ShortConv conv-state rollback bug); `mamba-cache-mode all` không cứu. Session H100 #2 |

---

## 2. Bottleneck Còn Lại & Điều Kiện Để Đạt 80+

* **ShortConv launch overhead không còn là ứng viên breakthrough.** Model có 10 short-conv
  layer; source vLLM hỗ trợ Full CUDA Graph cho decode-only. Runtime của đúng image best vẫn
  phải được xác minh từ log, không suy từ source thành fact vận hành.
* **Measured:** best hiện tại TPOT median 3.8ms. **Supported:** decode chịu áp lực weight
  bandwidth. **Hypothesis:** phân rã `~2.0–2.3ms GPU + ~1.5–1.8ms CPU`; chưa có trace portal
  chứng minh hai phần này cộng tuần tự hoặc có thể bóp về 2.2ms.
* **Hàm điểm:** quanh điểm hiện tại, TPOT đáng ~7.65 điểm/ms (≈33× TTFT/ms). Với TTFT điển
  hình 45ms, muốn chạm 80 cần TPOT điển hình khoảng **≤2.1ms** và không có tail/fail đáng kể.
  Vì ERS tính theo từng request, đây chỉ là target định hướng, không phải score prediction.
* **Kết luận:** config/frontend-only có thể kiếm vài điểm nhưng chưa có evidence đủ để bảo
  đảm 80. Để 80+ cần một đường TPOT mới được chứng minh: giảm padding/critical-path thật,
  speculative decoding có acceptance cao và đúng output, hoặc kernel W4A8 online giữ được
  tốc độ prefill FP8.

---

## 3. Future Plan Đúng Để Hướng Tới 80+ (ranked)

### P0 — Khóa lại evidence và baseline

1. Đối chiếu `score ↔ compose ↔ image digest`; khôi phục đúng BF16 anchor. File
   `compose-anchor-control.yml` hiện dùng Full-FP8 image và `--quantization=fp8`, nên KHÔNG
   phải BF16 anchor.
2. Xem 64.67 là **best observed**, chưa gán toàn bộ +2.17 cho NoLog/thread=1 nếu chưa có
   paired/median evidence. Mọi candidate mới phải isolate một biến và so cùng thời điểm.

### P1 — Xác minh runtime, không coi verify là một optimization

1. Đọc log/metrics của đúng image best: `cudagraph_mode`, capture sizes, attention backend,
   async scheduler, quantized-module coverage và decode batch histogram.
2. Chỉ dùng `compose-opt-fullgraph.yml` nếu log chứng minh baseline bị narrow. Không mặc định
   ép `[1,2,4,8]`: default vLLM đã có capture size nhỏ; giới hạn tới 8 có thể fallback khi
   batch thực lớn hơn. `128` vẫn bị loại vì đã gây assertion Mamba cache.
3. **PHÉP ĐO QUYẾT ĐỊNH — GPU-bound (A) vs core-saturated (B):** async đã bật mặc định
   (`vllm.py:975`) + detok/SSE chạy ở process API-server tách rời ⇒ mô hình "3.8ms = GPU+CPU
   cộng tuần tự" SAI. Critical path EngineCore ≈ GPU_forward + sample + build-input (metadata),
   KHÔNG gồm detok/SSE. Vì TPOT vẫn 3.8ms nên hoặc **(A)** GPU-bound cao hơn ước lượng (sustained
   MiG BW ~0.4 TB/s ⇒ ~2.9–3.4ms, CPU đã bị che) hoặc **(B)** 3 core bão hòa (CPU rò lên critical
   path). Đo `GPU-only time` + `CPU core utilization` khi decode để chọn nhánh. **Kết quả này
   quyết định P2/P3 loại trừ nhau:** (A) ⇒ P2b/P2c/P3 ≈ null, chỉ còn `interactivity` + phá
   byte-weight; (B) ⇒ P3 cắt input-path (metadata+sampling, KHÔNG phải output-path) mới là mũi chính.
   Bằng chứng yếu nghiêng (A): `fastsse-lean` (63.40) < `cpu-lean` (64.67) — cắt output-path không giúp.

### P2 — Low-risk latency path có cơ sở trực tiếp

1. **`--performance-mode=interactivity`**: ứng viên config số 1 cho workload latency,
   batch thấp; vLLM dùng CUDA Graph fine-grained ở small batch để giảm padding.
2. **Fastokens** (`VLLM_USE_FASTOKENS=1`): ⚠️ KHÔNG phải flag drop-in — verify `import fastokens`
   → **ModuleNotFoundError trong image thi đấu**. Đây là **task BUILD IMAGE MỚI** (pip
   `fastokens>=0.2`, digest mới, re-validate detok byte-identical). Chỉ đáng làm NẾU P1 ra
   nhánh (B) core-saturated; nếu (A) thì detok đã bị che → gain ≈ 0.
3. Sau đó mới isolate riêng renderer workers, FastSSE, OMP/thread binding và logging.
   Không gộp nhiều biến. Expected gain của nhóm này là **chưa biết, nhiều khả năng nhỏ**;
   mục tiêu là lấy gain thật, không gán trước +2→+8.

### P3 — Tìm và cắt critical path TPOT thật

1. Xác định phần còn tuần tự sau async mặc định: scheduler metadata, sampling, Mamba state
   post-processing, detokenize, ZMQ/IPC hay SSE.
2. Chỉ patch engine khi trace chứng minh hot path nằm trên critical path của portal-equivalent
   workload. Chênh engine-only/server local không được coi là toàn bộ overhead có thể xóa.
3. Gate tiếp tục: TPOT phải giảm sub-ms ổn định, output/correctness giữ nguyên, không đổi
   Accuracy Gate hay tạo failed request.

### P4 — Đường algorithmic để thật sự vượt vùng 70

1. **N-gram hiện tại: ĐÓNG** — đã đo acceptance≈0 và chậm hơn control; workload spec không
   công bố prompt là “restate document”.
2. Chỉ mở suffix/draft/spec proposer khác khi có cả ba bằng chứng:
   acceptance đủ cao trên workload đại diện, TPOT end-to-end thắng control, và greedy output
   khớp trên LFM hybrid + prefix caching. Issue rollback state upstream khiến correctness là
   cổng bắt buộc, không phải chi tiết triển khai.
   ⚠️ Async auto-OFF cho spec generic KHI để mặc định, NHƯNG **`draft_model` + explicit
   `--async-scheduling` ĐƯỢC hỗ trợ** (verify `vllm.py`: raise chỉ khi method ∉ {eagle,
   ngram-GPU, draft_model}). Vẫn phải đo net effect.
3. Không ghi gain +15→+25 trước khi có acceptance và TPOT thật.
4. ❌ **`compose-opt-draft-spec.yml` — CLOSED (fail G0 boot, root cause ở tầng source).**
   EngineCore crash lúc init: `RuntimeError: Engine core initialization failed … Failed core
   proc(s): {}`. Đã debug bằng cách đọc SOURCE vLLM 0.22.1 trong ĐÚNG image thi đấu:
   `SpecDecodeBaseProposer.validate_same_kv_cache_group` (`vllm/v1/spec_decode/llm_base_proposer.py`)
   có `assert len(groups)==1`. Draft LFM2 là hybrid (`ShortConv(MambaBase→AttentionLayerBase)` +
   `Lfm2Attention`) ⇒ 2 kv_cache_group ⇒ assert nổ. **Không flag nào sửa; mọi draft họ LFM2 đều
   hybrid ⇒ đóng vĩnh viễn `draft_model`.** Chi tiết: KB §5.9. EAGLE/MTP lách được boot nhưng vẫn
   dính rollback conv-state trên target + phải train head ⇒ không khuyến nghị.
   ⇒ **Hướng thuật toán 80+ còn lại chuyển sang P5 (W4A8 online Hopper)**, gate bằng phép đo P1
   byte-weight slope: `compose-p1-fp8.yml` vs `compose-p1-bf16.yml` (xem header các file đó).

### P-LEGAL — Nhóm hợp lệ, chi phí thấp, CHƯA TỪNG SUBMIT (verify grep repo). Gom ~72–75.

> Roofline (KB §4.3): TPOT bị chặn bởi **đọc weight** (~60%). Nhóm này tấn công các số hạng
> KHÁC (KV bandwidth, TTFT, padding), KHÔNG bị gate bởi P1 ⇒ làm ngay bất kể P1.

1. **`--kv-cache-dtype fp8`** — online, kernel CUTLASS sẵn; giảm KV bandwidth (chỉ ăn ở ctx dài).
   ⚠️ **BẮT BUỘC** chạy `bench/correctness_diff.py` — FP8-KV có thể trượt Accuracy Gate. ΔTPOT −0.1→−0.4ms.
2. **Khai thác nửa TTFT** — `--max-num-batched-tokens`/`--scheduling-policy`/chunked-prefill/warmup
   prefix. Cơ chế điểm: TTFT 45→22ms ⇒ để đạt 80 chỉ cần TPOT **~2.7ms** thay vì 2.1. Isolate 1 biến/lần.
3. **`--performance-mode=interactivity`** (`compose-opt-interactivity.yml`, đã build, chưa đo).

### P5 → MOONSHOT — Custom online W4A8 (đường 80 vật lý duy nhất nếu P1=nhánh A)

1. **W4A16/tinygemm: ĐÓNG** (portal 22.51; prefill nổ vì overhead dequant).
2. ⚠️ **Rule-lock (đọc source):** framework `online` của image CHỈ có weight-scheme 8-bit
   (`fp8_per_tensor/fp8_per_block/mxfp8`) — **KHÔNG có 4-bit online**. ⇒ W4A8 hợp lệ đòi **tự viết**
   một `OnlineQuantizationConfig` scheme 4-bit + kernel Hopper fused (Machete/CUTLASS preshuffled,
   dequant trong GEMV) để KHÔNG lặp overhead tinygemm. Image mới, digest mới.
3. Điều kiện thắng: prefill giữ FP8-fast, accuracy đạt gate, kernel-overhead < ~1.1ms tiết kiệm băng thông.
4. **CHỈ start sau khi P1 xác nhận nhánh A.** Nếu P1=nhánh B ⇒ bỏ moonshot, chuyển P3 (cắt CPU input-path, hợp lệ, rẻ hơn).

### Đã loại

* `--async-scheduling` explicit/combo: **REJECTED** — portal đã đo null và v0.22.1 bật async
  mặc định khi compatible.
* Custom ShortConv kernel như breakthrough, FlashInfer, capture-size 128, retry W4A16,
  retry n-gram hiện tại: **REJECTED**.

---

## 4. Rổ 5 Bài Nộp An Toàn Nhất Cho Cuộc Thi

Để không lãng phí thêm bất kỳ lượt submit nào và đảm bảo kết quả tốt nhất:

1. **Top Score Candidate:** `compose-cpu-lean.yml` (Score: **64.67**).
2. **Standard FP8 Candidate:** `compose-full-fp8.yml` (Score: **62.50**).
3. **v0.25.1 Candidate:** `compose-fullfp8-v0251.yml` (Score: **62.92**).
4. **FastSSE Patch Candidate:** `compose-fastsse-lean.yml` (Score: **63.40**).
5. **BF16 Anchor Candidate:** chỉ chọn submission/artifact BF16 thật sau khi đối chiếu digest.
   **Không dùng `compose-anchor-control.yml` hiện tại làm BF16 anchor vì file này đang chạy FP8.**

> ⚠️ Image pin theo digest — KHÔNG đổi sau khi chọn (điều khoản hậu kiểm BTC).
