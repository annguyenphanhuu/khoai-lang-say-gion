# KHOAI LANG SẤY GIÒN — COMPACT KNOWLEDGE BASE

> **Single Source of Truth** cho AI Agent & Team | **Cập nhật:** 2026-07-24 | **Best Score:** 64.67 (`compose-cpu-lean.yml`)
>
> ⚠️ Bản 2026-07-24 ĐÍNH CHÍNH LỚN: giả thuyết "24 layer ShortConv chạy eager" và "custom
> ShortConv kernel là breakthrough" đã bị bác bỏ. Roadmap 80+ cũng được audit lại: async,
> n-gram hiện tại và W4A16 đã đóng; floor 2.2ms/trần 79 chỉ là hypothesis.

---

## 1. Môi trường & Vật lý (đã đính chính)

* **Phần cứng:** 1 slice MiG H200 (18GB VRAM, ~247 TFLOPS FP8, **~0.6 TB/s HBM**, **3 CPU Cores**, 8GB RAM).
* **Model:** `LFM2.5-1.2B-Instruct` — hybrid. Theo `config.json` (`layer_types`):
  **16 layer = 10 short-conv + 6 GQA** (KHÔNG phải 24 ShortConv).
  `conv_L_cache=3` (conv nhân quả depthwise bề rộng 3), `conv_dim=2048`, `hidden=2048`,
  32 head / 8 KV head, `vocab=65536`.

### 1.1 Điểm vận hành: concurrency thấp, batch decode thật chưa được đo

TTFT p50 = 45ms hỗ trợ nhận định server không bão hòa ở median, nhưng không chứng minh mọi
decode step có batch=1. Workload Poisson có nhiều hội thoại đồng thời; tài liệu lịch sử từng
ước lượng trung bình 1–4, đỉnh 6–12. Mọi tối ưu CUDA Graph/padding phải dựa trên batch
histogram hoặc metrics thật, không hard-code C=1 từ TTFT p50.

### 1.2 Phân rã TPOT ≈ 3.8ms (thay cho phân tích "ShortConv eager" cũ)

| Thành phần | Ước lượng | Bản chất | Cắt được không? |
| :--- | :---: | :--- | :--- |
| **GPU/weight path** | **Hypothesis ~2.0–2.3ms** | Decode có bằng chứng mạnh là memory-sensitive, nhưng sustained bandwidth và byte traffic portal chưa đo. Full-FP8 patch đã quantize `ParallelLMHead`; câu cũ “LM head bf16 268MB/token” là sai với image này. | Có thể còn gain từ padding/kernel utilization; chưa được gọi là trần cứng |
| **CPU/frontend path** | **Hypothesis ~1.5–1.8ms** | metadata-builder + sampling + detokenize + ZMQ IPC + SSE; số local engine-vs-server không chứng minh toàn bộ phần này tuần tự trên portal | Chỉ cắt được phần nằm trên critical path thật |
| ShortConv launch overhead | Thấp | Source hỗ trợ Full Graph decode; runtime image best vẫn phải verify | Không phải breakthrough; không khẳng định tuyệt đối bằng 0 |

**Bằng chứng mạnh cho memory pressure:** bước nhảy ổn định lớn nhất từng đạt là BF16 → FP8.
Điều này hỗ trợ decode memory-sensitive, nhưng không đủ để suy toàn bộ 3.8ms thành một tổng
GPU+CPU chính xác hoặc kết luận 2.2ms là floor đạt được.

### 1.3 Vì sao "custom ShortConv kernel" bị loại (source vLLM)

* vLLM V1 mặc định `cudagraph_mode = FULL_AND_PIECEWISE`. Mamba/ShortConv backend
  (`vllm/v1/attention/backends/mamba_attn.py`) khai báo
  `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`, comment gốc:
  *"Mamba only supports decode-only full CUDAGraph capture."*
  ⇒ Ở batch decode thuần, **toàn bộ forward gồm 10 conv layer replay bằng 1 launch**;
  `causal_conv1d_update` chạy **TRONG graph**, KHÔNG có per-op launch overhead.
* `patch_mamba2a.py` hiện không được áp trong `image-cpulean/Dockerfile`, và image best dùng
  Full-FP8 image. Không được dùng file patch tồn tại trong repo làm bằng chứng runtime.
  Các failed request khi ép capture size 128 chứng minh đường FULL-decode có thể chạm
  `causal_conv1d_update`, nhưng không tự chứng minh mode/capture của baseline hiện tại.
* ⇒ Custom conv kernel không còn cơ sở là breakthrough: launch overhead nhiều khả năng đã
  được graph hóa, depthwise width-3 nhỏ, in/out projections đã FP8. Con số gain chính xác
  chưa đo; cost cao và xác suất thắng thấp.

---

## 2. Hàm điểm — "điểm nằm ở đâu" (mọi quyết định dựa vào đây)

`s = clamp((C−x)/(C−F), 0, 1)²` ; `ERS = mean(0.5·s_ttft + 0.5·s_tpot)`.

| Thành phần | Floor | Ceiling | Độ nhạy tại điểm hiện tại |
| :--- | :---: | :---: | :--- |
| TTFT | 10ms | 400ms | **~0.23 điểm/ms** (tại 45ms) |
| TPOT | 1ms | 10ms | **~7.65 điểm/ms** (tại 3.8ms) — **đáng ~33× TTFT/ms** |

**Bậc thang analytical cho một request điển hình** (giữ TTFT=45ms): 3.8ms→65 |
3.0→71 | 2.5→76 | 2.2→79 | **~2.1→80** | 1.5→86 | 1.0→92.

> Đây không phải dự báo ERS: grader lấy mean của hàm phi tuyến trên từng request và còn tail/fail.
> `2.2ms→79` không chứng minh 2.2ms là floor khả thi. Điều kiện định hướng để đạt 80+ là đưa
> phần lớn request về TPOT khoảng ≤2.1ms, hoặc kết hợp TPOT gần mức đó với TTFT/tail tốt hơn.

---

## 3. Lịch Sử Phân Tích Kết Quả (Experiment History)

| File / Cấu hình | ERS Score | Failed | TTFT p50 | TBT Median | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`compose-cpu-lean.yml`** | **64.67** | **0** | **45 ms** | **3.8 ms** | **BEST:** FP8 + Prefix + Thread=1 + NoLog. |
| `compose-full-fp8.yml` | 62.50 | 0 | 48 ms | 3.8 ms | Baseline mỏ neo FP8 chuẩn. |
| `compose-fastsse-lean.yml` | 63.40 | 0 | — | — | + fastsse SSE + renderer-num-workers=3. |
| `compose-fullfp8-v0251.yml` | 62.92 | — | — | — | Version fallback v0.25.1. |
| BF16 submission lịch sử | ~50.0 | 0 | — | — | Score đã đo; artifact BF16 hiện cần khôi phục/xác minh. |
| ~~`compose-opt80-cudagraph.yml`~~ | 63.72 | 6 | 50 ms | 4.0 ms | ❌ ĐÃ XÓA — capture-size=128 gây assertion mamba-cache. |
| ~~`compose-opt80-flashinfer.yml`~~ | 56.80 | 6 | 68 ms | 4.0 ms | ❌ ĐÃ XÓA — FlashInfer chậm prefill. |
| ~~`compose-opt80-pure-cudagraph.yml`~~ | 63.59 | 7 | 51 ms | 4.0 ms | ❌ ĐÃ XÓA — capture-size=128. |
| ~~`compose-opt80-shortconv.yml`~~ | (chưa đo) | — | — | — | ❌ ĐÃ XÓA — dựa trên giả thuyết ShortConv đã bị bác bỏ. |
| `compose-opt-draft-spec.yml` | **N/A (boot fail)** | — | — | — | ❌ **CLOSED** — EngineCore init crash; assert `validate_same_kv_cache_group` với draft hybrid LFM2 (xem §5.9). Không sửa được bằng flag. |

---

## 4. Future Plan 80+ (ranked theo evidence và information gain)

| P | Hướng | Trạng thái evidence | Tác động vào bottleneck | Cost / risk | Quyết định |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **P0** | **Audit baseline/artifact** — map score↔compose↔digest, khôi phục BF16 anchor thật, xác nhận 64.67 bằng paired/median trước khi gán causal gain | Bắt buộc | Ngăn tối ưu trên baseline/mô hình sai | Thấp | Làm trước mọi candidate |
| **P1** | **Runtime verification** — mode/capture/backend/async/quant coverage/decode batch histogram + **GPU-only time & CPU core utilization**. Vì async default-on + detok ở process riêng, mô hình "GPU+CPU tuần tự" sai; đo để chọn nhánh **(A) GPU-bound** (~2.9–3.4ms, CPU đã che) vs **(B) core-saturated** | Supported diagnostic | **Quyết định P2/P3 loại trừ nhau** (A⇒chỉ interactivity+floor-break; B⇒cắt input-path CPU) | Thấp | Làm trước P2/P3; đây là phép đo thông tin cao nhất |
| **P2a** | **`--performance-mode=interactivity`** → `compose-opt-interactivity.yml` | Supported, chưa đo project | Fine-grained small-batch graphs giảm padding; match workload latency thấp | Thấp | Candidate config số 1 (chỉ thắng nếu batch 3..15) |
| **P2b** | **Fastokens** (`VLLM_USE_FASTOKENS=1`) | ⚠️ package **KHÔNG cài trong image** (ModuleNotFoundError) | Detok/encode — nhưng nằm trên output-path async đã che (nhánh A) | **Cao — cần BUILD image mới** (pip fastokens≥0.2 + re-validate) | Chỉ làm nếu P1 ra nhánh (B) |
| **P2c** | Isolate renderer/FastSSE/OMP/logging từng biến | Hypothesis | CPU/TTFT; có thể chạm output path | Thấp–TB | Chỉ giữ delta vượt noise/median |
| **P3** | **Trace critical path TPOT rồi mới patch engine** | Cần evidence mới | Scheduler metadata, sampling, Mamba postprocess, IPC/detok/SSE | Cao | Mũi chính nếu trace tìm được ≥0.5–1ms tuần tự |
| **P4** | ~~**`draft_model` spec (draft LFM2)**~~ | ❌ **CLOSED — boot crash tại source** (§5.9): draft hybrid ⇒ 2 kv_cache_group ⇒ assert `validate_same_kv_cache_group`. | — | — | **ĐÓNG.** Cả họ LFM2 draft đều hybrid. EAGLE/MTP còn dính rollback conv-state + phải train head. |
| **P5** | **Custom online W4A8 + kernel Hopper fused** (MOONSHOT) | ⚠️ Framework `online` của image **KHÔNG có 4-bit** (§4.3) ⇒ phải TỰ viết scheme online + kernel | Giảm ~1.1ms byte weight NẾU kernel không lặp overhead tinygemm | Rất cao | Đường 80 duy nhất nếu P1=nhánh A; chỉ start sau P1 |

### 4.1 Thứ tự thực thi/gate cho nghiên cứu tiếp

1. **Evidence gate:** cấu hình/digest chính xác; cùng baseline; một biến; paired/median; đọc
   cả TTFT distribution, TPOT thật và failed count.
2. **P1:** nếu baseline đã Full Graph thì đóng `compose-opt-fullgraph.yml`; nếu narrow thật,
   sửa theo batch histogram. Không mặc định `[1,2,4,8]`, không dùng 128.
3. **P2:** `interactivity` → fastokens → isolate các frontend knob. Đây là vùng low-risk,
   nhưng không kỳ vọng riêng nó bảo đảm 80.
4. **P3:** chỉ patch đúng hot path được chứng minh. Gate: output/correctness giữ nguyên và
   TPOT giảm ổn định đủ lớn để tiến về mục tiêu ≤2.1ms.
5. **P4:** n-gram hiện tại đóng. Proposer mới chỉ qua cửa nếu acceptance đủ cao trên workload
   đại diện, TPOT end-to-end thắng, và `correctness_diff.py` PASS trên hybrid+prefix.
6. **P5:** chỉ tạo candidate khi có kernel W4A8 online native Hopper cụ thể và prefill không
   lặp lại thảm họa 22.51 của W4A16.

### 4.2 Artifact roadmap cũ — trạng thái sau audit

| Artifact | Trạng thái |
| :--- | :--- |
| `bench/verify_cudagraph.sh` | Diagnostic P1; không phải benchmark chứng minh gain |
| `compose-opt-fullgraph.yml` | Conditional only: chỉ dùng nếu log chứng minh narrow và capture sizes khớp batch thật |
| `compose-full-fp8-async.yml` / `compose-opt-async-fullgraph.yml` | **REJECTED:** async đã đo null và mặc định bật khi compatible |
| `compose-cpu-renderer3.yml` | Hypothesis phụ; isolate được nhưng expected gain thấp |
| `compose-opt-ngram.yml` | **REJECTED cho hướng hiện tại:** acceptance≈0/slower; chỉ giữ làm hồ sơ |
| `bench/correctness_diff.py` | Gate correctness hữu ích cho mọi thay đổi scheduler/spec/kernel |
| `compose-p1-fp8.yml` / `compose-p1-bf16.yml` | **P1 probe (ACTIVE):** cặp 1-biến đo slope byte-weight của TPOT ⇒ chốt GO/NO-GO cho W4A8 (P5). Công thức + ngưỡng trong header 2 file. Bản BF16 kiêm P0 anchor sạch. |
| `compose-opt-draft-spec.yml` + `image-draft-spec/` | **CLOSED (§5.9):** draft hybrid ⇒ assert kv-group ⇒ boot crash. Giữ làm hồ sơ. |

### 4.3 Cập nhật 2026-07-24 (chiều) — roofline + rule-lock quantization

**Roofline (định lượng, thay 'hypothesis 2.0–2.3ms'):** ở ~0.5 TB/s (slice MiG), đọc weight FP8
(~1.0–1.2 GB) ≈ **2.0–2.4ms = ~60% TPOT**; KV BF16 ~0.2–0.4ms (chỉ 6/16 layer, head_dim=64);
compute GEMV <0.2ms; sampling/CPU lộ ~0.5–1ms (phần lớn bị async che). ⇒ **TPOT là bài toán
băng thông ĐỌC WEIGHT.** Lever lớn duy nhất = giảm byte/param. (P1 đo empiric tỉ trọng này.)

**RULE-LOCK (đọc source `quantization/online/base.py` trong ĐÚNG image):** framework `online`
CHỈ có weight-scheme **8-bit** — `fp8_per_tensor`, `fp8_per_block`, `mxfp8` (int8 chỉ cho MoE).
**KHÔNG có 4-bit online.** ⇒ dưới luật "online quant only", lever byte-weight **CẠN ở FP8**;
mọi đường 4-bit đều cần checkpoint pre-quant (phạm luật) hoặc torchao-int4 (= W4A16 22.51).

**Kế hoạch xét lại — TẤT CẢ flag dưới đây CHƯA TỪNG submit (verify bằng grep repo):**

| Ưu tiên | Hướng | Hợp lệ? | ΔTPOT ước tính | Gate |
| :--- | :--- | :--- | :--- | :--- |
| **P-legal-1** | `--kv-cache-dtype fp8` | ✓ online, kernel CUTLASS sẵn | −0.1→−0.4ms (ctx dài) | **`correctness_diff.py` PHẢI pass** (Accuracy Gate) |
| **P-legal-2** | TTFT: `--max-num-batched-tokens` / `--scheduling-policy` / chunked-prefill / warmup prefix | ✓ | nới target TPOT: TTFT 45→22ms ⇒ 80 chỉ cần TPOT **~2.7ms** (thay vì 2.1) | isolate 1 biến/lần |
| **P-legal-3** | `--performance-mode=interactivity` (`compose-opt-interactivity.yml`, đã build) | ✓ | −0.2→−0.4ms nếu batch decode 3–15 | đọc histogram; hạ mem-util nếu OOM |
| **P-moonshot** | Custom online W4A8 + kernel Hopper fused | ✓ nếu tự viết scheme online | −~1.1ms nếu kernel > tiết kiệm băng thông | **CHỈ sau P1 = nhánh A** |

**P1 (`compose-p1-bf16.yml`) vẫn CẦN — vai trò đổi:** KHÔNG gate nhóm legal (làm bất kể). Nó
quyết định endgame ĐẮT: nhánh **A** ⇒ moonshot W4A8 là đường 80 duy nhất (định lượng phần thưởng);
nhánh **B** ⇒ có đường **LEGAL rẻ hơn** (cắt CPU input-path P3) hạ TPOT **không cần 4-bit** =
kịch bản TỐT. Roofline nghiêng A nhưng BW sai số ~50% ⇒ P1 vẫn thêm thông tin thật.

**Trần thực tế legal-only ≈ 72–75.** Chạm 80 cần moonshot (nếu A) hoặc P3 (nếu B).

#### KẾT QUẢ ĐO (portal, 2026-07-24) — P1 = NHÁNH A (xác nhận weight-bandwidth-bound)

| Bài | ERS | TBT median | TTFT p50 | failed/420 | acc_drop | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `compose-p1-bf16.yml` (b=2.0) | 48.17 | **6ms** | 56 | 7 | 0 | điểm đo T16 |
| `compose-cpu-lean.yml` (b=1.0, FP8) | 64.67 | **3.8ms** | 45 | 0 | — | điểm T8 (best) |
| `compose-kvfp8.yml` (FP8+FP8-KV) | 59.62 | 4ms | 62 | 6 | 0 | **REGRESSED → CLOSE** |

- **Slope = T16 − T8 = 6 − 3.8 = 2.2ms / (byte/param)** ⇒ **NHÁNH A xác nhận**: decode bị chặn
  bởi băng thông đọc weight (khớp roofline ~2.0–2.4ms). ~2.2ms TPOT đang "khóa" trong byte weight.
- **Dự đoán W4A8 (b=0.5):** `1.5·3.8 − 0.5·6 = 2.7ms` (±, TBT portal làm tròn số nguyên; BỎ QUA
  dequant overhead ⇒ đây là UPPER-BOUND lạc quan).
- ⚠️ **FP8-KV (`compose-kvfp8.yml`) ĐÓNG:** TBT không giảm (4 vs 3.8), TTFT tệ hơn (62 vs 45),
  6 failed. KV quá nhỏ (roofline) để bù overhead FP8-KV. Không dùng.
- **Bản đồ tới 80 (tính từ hàm điểm):** W4A8 2.7ms + TTFT giữ 45 ⇒ chỉ ~**74**. Cần **W4A8 2.7ms
  + TTFT ~20ms** mới ~**80**. ⇒ 80 đòi HAI thứ khó cùng thành công (kernel W4A8 gần-zero-overhead
  VÀ cắt TTFT 45→20). TTFT giờ là ĐỒNG-yêu-cầu, không còn optional.

#### FEASIBILITY SPIKE W4A8 (đọc source image + đo weight thật, 2026-07-24)

**SPEED = GREEN (kernel có sẵn, không phải viết CUDA từ đầu):**
- Image CÓ `CutlassW4A8LinearKernel` (`kernels/linear/mixed_precision/cutlass.py`): act **FP8-e4m3
  per-token động** (⇒ **prefill giữ FP8-nhanh**), weight int4, group-128, cap Hopper 90.
- Scheme `CompressedTensorsW4A8Fp8` lái kernel này. Repack ở `process_weights_after_loading` (lúc
  load) ⇒ **weight quantize-online NẠP được** vào kernel.
- LFM2-1.2B: mọi in/out feature đều %128 (hidden 2048, inter 12288, qkv 3072, kv 512, lm_head 65536)
  ⇒ **shape tương thích**, không chết từ đầu.
- Việc phải viết chỉ là **RTN int4-g128 online** (Python) nạp vào kernel (KHÔNG viết CUDA).
  Lõi + test: `image-w4a8/online_w4a8_rtn.py`, `image-w4a8/test_rtn_local.py` (test CPU, không cần Hopper).

**ACCURACY = YELLOW-RED (đây mới là rủi ro ràng buộc, KHÔNG phải kernel):**
- Đo int4-g128 trên WEIGHT THẬT LFM2: RTN-absmax **rel-error ~12–16%, cos ~0.992**; MSE-clip
  (vẫn online) chỉ hạ còn **~10–11%**. FP8 hiện tại ~2–3%. ⇒ int4 lệch gấp ~4–5×.
- Các method int4 chất-lượng-cao (AWQ/GPTQ, err-comp) đều cần **calibration = OFFLINE = PHẠM luật
  "online quant only"**. ⇒ online bị chặn ở naive-RTN ⇒ lỗi lớn.
- Gate portal có khoan dung (kvfp8 lossy vẫn `accuracy_drop=0`) nhưng int4 là nhiễu lớn hơn nhiều
  ⇒ **PASS/FAIL gate là đồng-xu, chỉ portal mới chốt được** (không test accuracy được local).
- Giảm rủi ro: mixed-precision — int4 cho MLP (chiếm ~phần lớn param, tiết kiệm băng thông chính),
  giữ FP8 cho attention q/k/v + lm_head (lỗi cao nhất, tốn ít băng thông).

**Kết luận:** moonshot W4A8 = **kernel không còn là rào cản**; rào cản chuyển thành **accuracy dưới
  online-only** + test-loop CHỈ-QUA-PORTAL (local không có Hopper cap90 + image cần CUDA13).

#### PHÂN TÍCH WORKLOAD (grading-workload-spec.json) — định hình lại đường 75–80

- 70 hội thoại × 6 turn = 420 req; **shared prefix 1000 tok (mọi req) + per-conv prefix 1000 tok
  (chia 6 turn)**; +150 user/turn; **output pinned 300**; Poisson. Context ~2150→4400 tok.
- ⇒ **Prefix-cache ăn cực mạnh**: prefill mỗi turn chỉ ~150–450 tok MỚI ⇒ TTFT 45ms phần lớn là
  **overhead/scheduling**, không phải compute. **`--max-model-len=32768` thừa 6–7×** (chỉ cần ~6144).
- ⇒ Batch decode ~3–15 (70 hội thoại Poisson).

**Ma trận điểm (ERS = 50·(s_ttft+s_tpot); s=((C−x)/(C−F))²):** 75⇔sum 1.5; 80⇔sum 1.6 (nay 1.295).
| | TPOT 3.8 | 3.3 | 2.7 |
| TTFT 45 | 64.7 | 68 | 74 |
| TTFT 30 | 68.5 | 72 | **78** |
| TTFT 20 | 70.9 | 74 | **80** |

**KẾT LUẬN CHIẾN LƯỢC (physics + luật, đã vét cạn lever):**
- **TPOT chỉ giảm được bằng 4-bit weight (lossy).** FP8 là sàn lossless; speculative (amortize weight-read)
  đã CHẾT cho LFM2; batch không giảm TPOT vì bandwidth-bound. ⇒ **KHÔNG có đường lossless nào tới 75+**
  về phía TPOT. 4-bit online lệch ~11–12% (int4≈nvfp4, format không cứu được) ⇒ accuracy là đồng-xu-portal.
- **TTFT giảm được LOSSLESS** (right-size max-len → nhiều KV block cho prefix cache; scheduling/chunked-prefill).
  Headroom lớn (45→~20–30). Đây là NỬA điểm không rủi ro.
- ⇒ **Đường 75–80 = Workstream A (lossless TTFT, ~68–71) + Workstream B (4-bit weight, TPOT→2.7).**
  75 cần B + TTFT nhẹ; 80 cần B + TTFT→~20. B là bắt buộc để vượt ~71; không né được accuracy gamble.

#### Ứng viên đã dựng (2026-07-24)

| File | Workstream | Biến vs cpu-lean | Rủi ro | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| `compose-rightsize.yml` | A (lossless) | max-model-len 32768→6144 | ~0 (lossless), downside chặn 64.7 | **nộp được ngay** |
| `compose-bnb-nf4.yml` | B (4-bit) | quant fp8→**bitsandbytes** (NF4 online, matmul_4bit fused) | tốc-độ-kernel + CUDA-graph + accuracy; **0 code** | probe 4-bit rẻ nhất |
| `image-w4a8/` (RTN core+test) | B (4-bit, dự phòng) | Marlin-FP4/CutlassW4A8 custom patch | code nhiều, kernel nhanh hơn BnB | dùng NẾU BnB chậm |

**Thứ tự Workstream B:** thử `bitsandbytes` (0 code) TRƯỚC → nếu accuracy pass & TPOT giảm ⇒ ~74,
xong stack right-size/TTFT. Nếu BnB chậm (kernel/eager) nhưng accuracy pass ⇒ mới build patch
Marlin-FP4 (đã biết 4-bit qua được gate, chỉ cần kernel nhanh hơn).

#### KẾT QUẢ PORTAL 2026-07-24 (cuối ngày) — các submit Workstream A/B

| Config | ERS | tbt | ttft p50 | failed | acc_drop | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `compose-rightsize.yml` | 60.82 | 4 | 62 | 7 | 0 | ❌ max-len↓ KHÔNG đổi KV pool (do mem-util); regress. ĐÓNG. |
| `compose-bnb-nf4.yml` | 30.8 | 15 | 84 | 9 | 0 | ❌ BnB kernel chậm/eager. Nhưng **acc_drop=0 ⇒ 4-bit QUA GATE**. |
| **`compose-w4a8.yml` (run 1)** | **65.06** | **3** | 50 | 6 | **0** | ✅ **BEST MỚI.** Patch custom BOOT lần đầu OK; TPOT 3.8→3.0; int4 qua gate. |
| `compose-w4a8.yml` (run 2) | 64.53 | 3 | 57 | 6 | 0 | Re-run: variance (ttft 50→57), tbt ỔN ĐỊNH 3. |

**KẾT LUẬN NGÀY:**
- ✅ **W4A8 online PROVEN**: kernel CutlassW4A8 + RTN-int4 online (`image-w4a8/`, digest
  `sha256:afa562d714...`) boot thật, TPOT 3.8→**3.0ms ỔN ĐỊNH**, `accuracy_drop=0`. Giả thuyết
  "decode weight-bandwidth-bound, 4-bit là lever" ĐÚNG.
- ⚠️ **Nhưng net gain nhỏ (best 65.06 vs 64.67)** vì: W4A8 làm prefill chậm hơn chút (int4) → TTFT
  50-57 (cpu-lean 45); + **6 failed floor** ở MỌI run hôm nay (kể cả rightsize fp8) = variance portal.
- **Tính ở ngày công bằng** (tbt 3, ttft 45, fail 0): `50·(0.826+0.605)≈**71.5**` → tiềm năng W4A8 thật ~71.
- **Trần near-term** = W4A8-optimized (MSE-clip → tbt~2.7) + fair-day ≈ **74**. **80 vẫn cần cắt TTFT** (đòn còn thiếu).

#### KẾT QUẢ CHẨN ĐOÁN 2026-07-25 — chốt nút cổ chai của tbt 3ms

| Config | tbt | ers | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| cpu-lean (hôm nay) | 3 | 64.53 | **7 failed** ⇒ failures = VARIANCE portal, KHÔNG phải W4A8. |
| w4a8 | 3 | 65.06 | = cpu-lean ⇒ **weight lever CẠN** ở FP8 (W4A8 không giúp). |
| w4a8+interactivity | 3 | 62.31 | ttft 61 ⇒ interactivity REGRESS. ĐÓNG. |
| **diag-eager** (tắt graph) | **12** | 33.93 | tbt 3→12 ⇒ **CUDA graph tiết kiệm ~9ms launch overhead**. |

**KẾT LUẬN — đã chốt nút cổ chai (không cần thuê GPU):**
- MSE-clip: **VÔ ÍCH** (cùng bit → cùng tbt; accuracy đã pass). Bỏ.
- 6-7 failed = **variance portal** (connection/HTTP), không sửa được (BTC xác nhận cơ chế).
- tbt 3ms **KHÔNG phải** weight-bandwidth (W4A8=FP8) và **KHÔNG phải** launch overhead (graph đã bắt,
  tắt graph → 12ms). ⇒ 3ms = **THỰC THI NHIỀU OP NHỎ TUẦN TỰ trên GPU** (hybrid: 10 conv + 6 attn +
  MLP + norm = rất nhiều kernel nhỏ/token, batch-1 GPU under-utilized). Đây là **sàn op-execution**.
- Cắt xuống dưới 3ms ⇒ cần **fuse op / custom kernel** (kỹ thuật lớn), KHÔNG phải flag. Đã hết đường config.

**⇒ ~65 nhiều khả năng là TRẦN THỰC của LFM2 trên MiG-H200 dưới luật hiện tại.** 75-80 đòi kernel-fusion
(research lớn) — ngoài tầm config/quant. Dossier kỹ thuật (draft-spec root-cause, roofline, W4A8 online
chạy thật, chuỗi chẩn đoán) là điểm mạnh cho vòng sau. Bài nộp tốt nhất: W4A8 65.06 hoặc cpu-lean 64.67.

#### KẾT QUẢ ĐO 2 (portal, 2026-07-24) — rightsize & BnB

| Bài | ERS | tbt | ttft | failed | acc_drop | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `compose-rightsize.yml` | 60.82 | 4 | 62 | 7 | 0 | ❌ giả thuyết SAI: max-len KHÔNG đổi KV pool (pool = mem-util). 6144 quá sát → 7 failed. BỎ. |
| `compose-bnb-nf4.yml` | 30.8 | **15** | 84 | 9 | **0** | ❌ tốc độ (kernel BnB/eager) — NHƯNG **accuracy_drop=0 ⇒ 4-bit QUA GATE.** |

**BƯỚC NGOẶT:** BnB chứng minh **4-bit accuracy PASS** (acc_drop=0), chỉ kernel BnB chậm. ⇒ dùng
**kernel nhanh CutlassW4A8** (CUDA-graph). Đã build patch `image-w4a8/` + `compose-w4a8.yml`:
- Đã verify LOCAL (CPU): syntax, import, đăng ký `--quantization online_w4a8`, RTN+pack round-trip
  khớp định dạng kernel (rel-err ~0.114), chữ ký param đúng. Image build không cần GPU.
- CHƯA verify (portal-only): kernel CUTLASS runtime + accuracy thật → submit #1, giữ #2 sửa.
- Kỳ vọng: tbt~2.7 + acc nhỏ → ~74. Fail-loud assert để log chỉ đúng lỗi.

---

## 5. Giả Thuyết ĐÃ BỊ BÁC BỎ (đừng lặp lại)

1. ❌ **"24 layer ShortConv chạy PyTorch Eager, ~2.5–3ms CPU overhead."**
   → SAI: chỉ **10 layer**; đã nằm **trong Full CUDA Graph** ở decode.
2. ❌ **"Engine vLLM KHÔNG THỂ CUDA-graph capture custom op có mutable state."**
   → SAI: capture được (in-place GPU write hợp lệ trong graph). Cản trở graph là host-device
   sync/dynamic shape, KHÔNG phải "mutable state". Default `FULL_AND_PIECEWISE`.
3. ❌ **"Custom ShortConv CUDA/Triton kernel là hướng breakthrough."**
   → SAI: op đã trong graph (không còn launch overhead), compute ~0. Gain < 1 điểm.
4. ❌ **`--max-cudagraph-capture-size=128`** → gây failed request (assertion mamba-cache
   block, PR #34571). Chỉ cap capture sizes theo batch histogram và Mamba cache thật; không
   hard-code `[1,2,4,8]` nếu chưa biết batch vượt 8 hay không.
5. ❌ **FlashInfer backend** → chậm prefill (56.80). Giữ FlashAttention (default Hopper).
6. ❌ **`--async-scheduling` explicit/combo** → portal đã đo 63.38 = control 63.38; v0.22.1
   bật async mặc định khi compatible.
7. ❌ **Online W4A16/tinygemm hiện tại** → portal 22.51, TTFT 205/935ms, TBT 6ms.
8. ❌ **N-gram speculative hiện tại** → acceptance≈0 và TPOT local 21ms > control 17ms.
   Workload spec không có evidence cho giả thuyết “restate document”.
9. ❌ **`draft_model` speculative decoding với draft LFM2 (230M/350M/bất kỳ size)** → EngineCore
   **CHẾT lúc init** (`RuntimeError: Engine core initialization failed … Failed core proc(s): {}`).
   **Root cause xác định từ SOURCE vLLM 0.22.1 trong đúng image thi đấu** (đọc tĩnh, không suy đoán):
   `SpecDecodeBaseProposer.validate_same_kv_cache_group` (`vllm/v1/spec_decode/llm_base_proposer.py`
   ~L1518-1535) có `assert len(groups)==1` — "All drafting layers should belong to the same kv
   cache group". Draft LFM2 là **hybrid**: `Lfm2ShortConvDecoderLayer → ShortConv(MambaBase →
   AttentionLayerBase)` đăng ký KV-spec kiểu **conv-state**, còn `Lfm2Attention` đăng ký KV-spec
   kiểu **attention** ⇒ layer draft rơi vào **2 kv_cache_group** ⇒ assert nổ ngay trong pha KV
   cache init của EngineCore. Comment source tự nói: *"May extend to multiple AttentionMetadata
   in the future"* = draft hybrid **CHƯA được implement**. **KHÔNG flag nào sửa** (không phải fp8,
   prefix-caching hay async); mọi draft họ LFM2 đều hybrid ⇒ **đóng vĩnh viễn** đường `draft_model`.
   EAGLE/MTP *có thể* lách lỗi boot này (head tự train thường attention-only, 1 kv group) nhưng
   vẫn dính vách **rollback conv-state trên TARGET hybrid khi verify** (correctness) + phải tự
   train head ⇒ chi phí/rủi ro cao, không khuyến nghị.
10. ❌ **int4 / W4A8 (mọi kernel) hạ tbt** → CHẾT. Đo H100 (2026-07-25, CUDA-graph sạch):
    **Machete** (SOTA Hopper int4) = **0.70x fp8** @batch-1 (fp8 42µs vs int4 60µs / layer linear).
    Dequant int4 > byte tiết kiệm khi không đói BW ⇒ đúng lý do W4A8 hòa fp8 (65). Không kernel int4
    nào (Machete/Marlin/CutlassW4A8) cứu được. **Weight-quant lever cạn ở fp8, đóng.**
11. ❌ **Tối ưu CPU input-path** (sampling/detok/prepare) → vô ích. Đo H100: tbt 3-core (1.55ms) =
    8-core (1.67ms) ⇒ **GPU-bound (nhánh A)**, CPU đã bị async che. Đóng toàn bộ hướng P3.
12. ❌ **fuse_norm_quant / fuse_act_quant flags** → đã **default-on** vllm≥0.25.1 (log config xác nhận)
    ⇒ không phải quả ngọt, baseline đã có. Chỉ `fuse_attn_quant` còn tắt (kỳ vọng ~0).
    ⇒ **tbt=3ms = op-floor GPU vật lý** (full-H100: linear 0.68ms + phi-linear 0.85ms = 1.53ms; MiG
    bóp BW ~2x → 3ms). **Đường 80+ duy nhất còn lại = speculative decoding, xem `NEXT-80.md`.**

---

## 6. Quy Tắc & Mỏ Neo An Toàn

1. **Rổ 5 bài nộp** (image pin theo digest, KHÔNG đổi sau khi chọn):
   - Ứng viên 1: `compose-cpu-lean.yml` (**64.67**).
   - Ứng viên 2: `compose-full-fp8.yml` (62.50).
   - Ứng viên 3: `compose-fullfp8-v0251.yml` (62.92).
   - Ứng viên 4: `compose-fastsse-lean.yml` (63.40).
   - Ứng viên 5: submission BF16 thật sau khi đối chiếu compose/image digest.
     `compose-anchor-control.yml` hiện chạy Full-FP8 + `--quantization=fp8`, nên KHÔNG phải
     BF16 anchor và không được dùng làm fallback accuracy.
2. **Trước mọi thử nghiệm cudagraph:** VERIFY mode trong log; đừng ép capture-size lớn.
3. **TPOT là nửa điểm phải giành** (~7.65 điểm/ms) — ưu tiên hơn TTFT (~0.23 điểm/ms).
