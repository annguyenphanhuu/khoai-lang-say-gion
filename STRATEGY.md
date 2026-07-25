# CHIẾN LƯỢC — KHOAI LANG SẤY GIÒN

> Facts & số đo: `on-tap/KNOWLEDGE_BASE.md`. Luật: `README.md` (**chỉ vLLM**, **chỉ online quant**).
> Best portal: **65.06** (`compose-w4a8.yml`) · **64.67** (`compose-cpu-lean.yml`). Hạn: **30/07/2026**.
> Cập nhật 2026-07-25: dọn toàn bộ hướng SGLang/TRT-LLM (phạm luật) và các kết luận suy từ rig H100
> sai regime (KB §5). Thay bằng **mô hình băng thông đã fit 3 điểm portal** ở §1.

---

## 1. Mô hình quyết định: decode là bài toán BYTE/STEP

Trên MiG H200 (~0.6 TB/s), decode ở batch ~27 gần như **100% bandwidth-bound**:

```
TPOT(ms) ≈ [ W_bytes + B × 35MB ] / 0.6 TB/s        (B = batch decode ≈ 27)
             └─ weight ─┘   └─ KV read/step ─┘
```

| Config | W_bytes | KV | Dự đoán | **Portal đo** |
| :-- | --: | --: | --: | --: |
| BF16 | 2.40 GB | 0.95 GB | 5.58 ms | **6.0** ✓ |
| FP8 (`cpu-lean`) | 1.20 GB | 0.95 GB | 3.58 ms | **3.8** ✓ |
| W4A8 như đang build | 0.80 GB | 0.95 GB | 2.92 ms | **3.0** ✓ |

**Fit 3/3 điểm, sai số ~5%.** Mô hình này là công cụ dự báo, không phải giả thuyết.
Hai điều nó nói ra ngay:

- **KV read ≈ 0.95 GB/step ≈ 45% tổng traffic** — cùng cỡ weight. Roofline cũ ghi "KV 0.2–0.4ms"
  vì tính cho batch 3–15; điểm vận hành thật là batch 27 × ctx ~2900.
- **`lm_head` trong `compose-w4a8.yml` đang là BF16** (KB §7.1) = 0.27GB thay vì 0.13GB.

---

## 2. Đường tới 80+ — hạ CẢ HAI số hạng byte/step

| Bước | Thay đổi | W_bytes | KV | TPOT dự đoán | s_tpot | ERS @ ttft 48 |
| :-- | :-- | --: | --: | --: | --: | --: |
| — | hiện tại (`w4a8`) | 0.80 | 0.95 | 3.0 | 0.605 | 65 |
| **1** | + `lm_head` FP8 (**1 dòng code**) | 0.67 | 0.95 | **2.70** | 0.665 | **~74** |
| **2** | + KV cache 8-bit chạy THẬT | 0.67 | 0.48 | **1.91** | 0.808 | **~81** |
| 3 | + cắt đuôi TTFT turn-1 | — | — | — | — | **~84** |

**Điểm chốt quan trọng nhất:** kết luận cũ *"80 đòi TTFT ~20ms"* chỉ đúng khi TPOT kẹt ở 2.7ms.
Nếu TPOT về 1.9ms thì **TTFT 45–50ms hiện tại đã đủ để vượt 80** — TTFT trở lại thành bonus, không phải
đồng-yêu-cầu. Đây là lý do bước 2 là cửa duy nhất cần mở.

**Rủi ro accuracy gần như đã trả xong:** int4 weight đã `acc_drop=0` (W4A8 + BnB-NF4) và fp8-KV cũng
đã `acc_drop=0` (run 59.62). Cả hai thành phần đều **đã qua Accuracy Gate riêng lẻ**.

---

## 3. Việc phải làm (theo thứ tự)

### Bước 1 — `lm_head` FP8 trong online_w4a8 · rẻ nhất, chắc nhất, ~+9 điểm

Trong [`image-w4a8/online_w4a8.py`](image-w4a8/online_w4a8.py) `get_quant_method` hiện trả `None` cho
`ParallelLMHead` ⇒ rơi về BF16. Cho nó dùng scheme FP8 (không int4 — lm_head là layer sai số cao nhất,
và FP8 đã đủ để hạ 268→134MB). Build image mới, pin digest, submit.
**Gate:** `bench/correctness_diff.py` temp=0 vs golden fp8.

### Bước 2 — KV cache 8-bit: tìm ra vì sao lần trước KHÔNG ăn ⇒ cửa 80

`--kv-cache-dtype fp8` trên nền FP8 lẽ ra phải cho TPOT 3.8 → 2.79ms theo mô hình §1. Thực tế
portal ra **tbt 4 (tệ hơn) + ttft 62 + 6 failed** ⇒ **cơ chế đúng nhưng đường thực thi trong vLLM sai**.
Giả thuyết cần loại trừ, theo thứ tự rẻ nhất — **đọc source TRONG ĐÚNG image thi đấu, đừng đoán**:

1. Backend attention đang chạy có **tiêu thụ fp8-KV native** hay **dequant về BF16 trước khi attention**?
   Nếu dequant ⇒ vẫn đọc đủ byte + thêm một pass ⇒ đúng như đã đo. Đây là giả thuyết số 1.
2. Trên model **hybrid**, `kv_cache_dtype` có áp cho cả 6 layer GQA không, hay bị bỏ qua/ép fallback?
3. `fp8_e4m3` vs `fp8_e5m2` vs **int8** (luật cho phép cả INT8) — khác kernel, khác đường code.
4. Thử lại trên base **v0.25.1** (`image-fullfp8-v0251/`) — FlashAttention + fp8-KV support mới hơn v0.22.1.

Nếu backend hiện tại không đọc fp8-KV native thì lối ra là chọn/patch backend decode có hỗ trợ.
Phần thưởng −0.8ms là lớn nhất còn lại trên bàn ⇒ đáng dồn hết thời gian còn lại vào đây.

**Bonus miễn phí:** KV 8-bit ⇒ pool KV gấp đôi ⇒ prefix cache giữ được nhiều block hơn ⇒ TTFT cũng tốt hơn.

### Bước 3 — Đuôi TTFT turn-1 (lossless, +2→5)

Chạy `bench/ers_harness.py` và **ghi lại `turn1 p50` vs `turn2+ p50`** (harness đã in sẵn, chưa ai đọc).
Nếu turn-1 thật sự ~300ms thì 1/6 request đang có `s_ttft ≈ 0` — sửa được bằng scheduling thuần
(`--max-num-batched-tokens` **cao hơn** 8192, `--long-prefill-token-threshold`), lossless. Isolate 1 biến/submit.

---

## 4. Quy tắc vận hành

1. **Portal là oracle duy nhất.** Không giới hạn submit online (chỉ *chọn* 5 bài ở cuối) ⇒ đo bằng portal.
   Rig H100 chỉ dùng cho correctness/acceptance — xem KB §5 về lý do.
2. **1 biến / 1 submit.** So sánh bằng **ERS**, không bằng `tbt` (portal làm tròn số nguyên = ~7.65 điểm).
3. 6–7 failed/420 là variance portal, không phải lỗi config. Đừng đuổi theo.
4. `correctness_diff.py` là cổng bắt buộc trước mọi submit đổi kernel/quant.
5. Image pin theo digest, **không đổi sau khi chọn** (điều khoản hậu kiểm).

## 5. Rổ 5 bài an toàn (giữ nguyên, không rủi ro)

`compose-w4a8.yml` 65.06 · `compose-cpu-lean.yml` 64.67 · `compose-fastsse-lean.yml` 63.40 ·
`compose-fullfp8-v0251.yml` 62.92 · `compose-full-fp8.yml` 62.50.

Mọi bài mới từ §3 nếu thắng sẽ đẩy dần các bài yếu ra khỏi rổ.
