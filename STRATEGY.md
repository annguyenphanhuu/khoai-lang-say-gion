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

- **KV read ≈ 1.09 GB/step ≈ 48% tổng traffic** — cùng cỡ weight. Roofline cũ ghi "KV 0.2–0.4ms"
  vì tính cho batch 3–15; điểm vận hành thật là batch 27 × ctx ~2900.
- **`lm_head` trong `compose-w4a8.yml` đang là BF16** (KB §7.1) = 0.27GB thay vì 0.13GB.

### Luật vàng rút ra từ mô hình

> **Kernel/tối ưu chỉ đáng làm nếu nó DI CHUYỂN ÍT BYTE HƠN.**
> Giảm FLOP, giảm số launch, fuse op ⇒ **đúng bằng 0 điểm** (GPU đang đứng chờ HBM).

Đây là lý do "fuse ShortConv/norm/act" chết — không phải vì op đã ở trong graph, mà vì fuse nhanh hơn
thì vẫn chờ HBM. Và ngược lại: kernel đọc ít byte thì **ăn tuyến tính**. Hợp lệ theo `README.md` §3
(`custom CUDA/Triton kernels`, `Fused attention kernels`, `memory layout`).

---

## 2. Đường tới 80+ — chỉ có một trục: byte/step

| Bước | Thay đổi | W | KV | TPOT | ERS @ ttft 48 |
| :-- | :-- | --: | --: | --: | --: |
| — | hiện tại (`w4a8`) | 0.80 | 1.09 | 3.15 | 65 |
| **K3** | `lm_head` FP8 (**1 dòng**) | 0.67 | 1.09 | **2.93** | **~72** |
| **K2** | + KV 8-bit ăn thật | 0.67 | 0.55 | **2.03** | **~80** |
| ~~K1~~ | ~~+ shared-prefix attention~~ | 0.67 | 0.32 | ~~1.65~~ | ~~84~~ — **ĐÓNG, KB §8.2** |

**TTFT giữ nguyên 48ms suốt bảng — không cần đụng tới nó.** Sai số mô hình ±5%.

**Rủi ro accuracy gần như đã trả xong:** int4 weight đã `acc_drop=0` (W4A8 + BnB-NF4) và fp8-KV cũng
đã `acc_drop=0` (run 59.62). Cả hai thành phần đều **đã qua Accuracy Gate riêng lẻ**.

### Không đáng viết (đừng mất thời gian)

Fuse ShortConv/norm/act · fused sampling · custom int4 GEMV (CUTLASS đã tối ưu, weight byte đã sàn ở
int4) · bất cứ thứ gì giảm FLOP hoặc số launch.

---

## 3. Việc phải làm (theo thứ tự)

### Bước 0 — ĐÃ LÀM (2026-07-25): đọc source trong image → xem KB §8

Docker local có đúng image thi đấu ⇒ đã đọc source, kết quả đổi hẳn ưu tiên:

- ✅ **FA3 tiêu thụ fp8-KV native, không dequant** (KB §8.1) ⇒ **không cần viết kernel Triton cho K2.**
  Cơ chế đọc-nửa-byte đã có sẵn; vấn đề chỉ là tìm ra cái gì đang chặn phần tiết kiệm.
- ❌ **Cascade attention tự tắt FULL cudagraph** (KB §8.2) ⇒ đổi 0.5ms KV lấy 9ms launch overhead.
  **K1 ĐÓNG.**
- 🔍 Nghi phạm mới cho regression fp8-KV: **hybrid page-size unification đổi `block_size`** ⇒ granularity
  prefix-cache thô hơn ⇒ ttft 45→62 (KB §8.3). Kiểm bằng log boot, không cần submit.

### K3 — `lm_head` FP8 trong online_w4a8 · rẻ nhất, chắc nhất, ~+7 điểm

Trong [`image-w4a8/online_w4a8.py`](image-w4a8/online_w4a8.py) `get_quant_method` hiện trả `None` cho
`ParallelLMHead` ⇒ rơi về BF16. Cho nó dùng scheme FP8 (không int4 — lm_head là layer sai số cao nhất,
và FP8 đã đủ để hạ 268→134MB). Build image mới, pin digest, submit.
**Gate:** `bench/correctness_diff.py` temp=0 vs golden fp8.

### K2 — KV 8-bit dequant TRONG REGISTER · lever lớn nhất còn lại (−0.8ms)

Lý do `--kv-cache-dtype fp8` ra 59.62 (tbt 4, tệ hơn): nếu backend **dequant fp8 → BF16 vào buffer**
trước attention thì nó **vẫn đọc đủ byte + thêm một pass**. Kernel phải load `float8e4nv` rồi
`.to(float32)` **ngay trong vòng lặp flash-decoding**, không materialize gì.

FA3 native fp8-KV đã sẵn (KB §8.1) ⇒ **không phải viết kernel**, chỉ phải gỡ cái đang chặn.
Thứ tự rẻ → đắt:

1. **Đọc log boot** của run fp8-KV cũ: `block_size` + `GPU KV cache size: N tokens`.
   N không tăng ~2× ⇒ page-size unification ăn hết (KB §8.3). **Miễn phí, làm trước.**
2. `--kv-cache-dtype fp8` + **pin `--block-size` tường minh** (16, rồi 32) để chặn unification đổi block.
3. `fp8_e4m3` vs `fp8_e5m2` vs **int8** (luật cho phép cả INT8) — khác kernel, khác đường code.
4. Base **v0.25.1** (`image-fullfp8-v0251/`) — support mới hơn v0.22.1.
5. Chỉ khi 1–4 chứng minh FA3 fp8 kernel *thật sự* chậm ở `head_dim=64`: viết/patch decode kernel Triton
   load `tl.float8e4nv` in-register. Đệm rất rộng — FA BF16 đọc 1.09GB = 1.8ms; Triton fp8 đọc 0.55GB,
   kể cả chỉ đạt **70% hiệu suất** vẫn ra 1.3ms ⇒ vẫn thắng. Không cần kernel giỏi, chỉ cần đọc ít byte.

**Bonus:** KV 8-bit ⇒ pool KV gấp đôi ⇒ prefix cache giữ nhiều block hơn ⇒ TTFT cũng tốt hơn.

### ~~K1 — Shared-prefix / cascade attention~~ · ĐÓNG (KB §8.2)

Cơ chế đúng và workload thoả hết điều kiện gate (`common_prefix_len` ~992 ≥ 256, `num_reqs` 27 ≥ 8):
27 sequence đang đọc lặp cùng 1000 token prefix = 324 MB/step để đọc 12 MB dữ liệu thật = **34% KV traffic**.
Nhưng `gpu_model_runner.py:3777` đặt `disable_full=use_cascade_attn` ⇒ bật cascade là **mất FULL cudagraph**,
đổi 0.5ms lấy 9ms. Chỉ mở lại được nếu làm cascade capture full-graph = research, ngoài deadline.

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
