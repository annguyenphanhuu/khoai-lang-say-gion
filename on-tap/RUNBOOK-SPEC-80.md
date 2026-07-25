# RUNBOOK-SPEC-80 — Thực thi speculative decoding tới 80đ trên pod H100

> Thực thi chi tiết của `on-tap/NEXT-80.md`. Đọc cùng `HANDOFF-RUNPOD.md` (rig + dead-ends).
> Cách dùng: chạy TỪNG BLOCK theo thứ tự, **dán output về chat** rồi mới sang bước kế.
> Nguyên tắc: **G0 correctness là cổng bắt buộc** trước G1; **1 biến/lần** khi sweep.
>
> ⚠️ GOTCHA QUAN TRỌNG: `--disable-log-stats` TẮT counter Prometheus ⇒ `/metrics` KHÔNG có
>   `spec_decode_*` ⇒ acceptance probe mù. **Khi đo acceptance (G1), BỎ `--disable-log-stats`.**
>   (Khi đo tbt thuần / nộp portal thì bật lại để bớt overhead.)

Mục tiêu số: mean_accept_len ≥ 1.5 ⇒ ~79; ≥ 2.0 ⇒ ~82. Gate: correctness PASS + net tbt < control.

---

## PHASE A — Resume pod + baseline fp8 (calibrate)

### A.1 Bật pod + verify (volume `/workspace` có sẵn model+bench)
```bash
nvidia-smi | head -3          # CUDA 13.x?  cap 9.0?
ls /workspace/model /workspace/bench
python3 -c "import vllm; print('vllm', vllm.__version__)"   # phải 0.25.1
```
**→ DÁN VỀ:** 3 dòng trên.

### A.2 Serve baseline fp8 (KHÔNG spec) — golden cho correctness + control tbt
```bash
pkill -f vllm; sleep 3
OMP_NUM_THREADS=1 taskset -c 0-2 vllm serve /workspace/model \
  --served-model-name LFM2.5-1.2B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.225 \
  --enable-prefix-caching --quantization fp8 > /workspace/base.log 2>&1 &
# chờ healthy
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)" = 200 ] && { echo HEALTHY; break; }
  sleep 5
done
```
```bash
python3 /workspace/bench/loadgen_c1.py 30                                   # control tbt
BASE=http://localhost:8000 NCONV=3 OUT=300 python3 /workspace/bench/wl_probe.py | tail -5
```
**→ DÁN VỀ:** `C=1 ... tbt_med` + ttft turn1/turn2+. Đây là **CONTROL** để so mọi bước spec.

### A.3 Chụp golden correctness từ baseline (dùng ở G0)
```bash
MODE=capture BASE=http://localhost:8000 OUT=/workspace/bench/golden.json \
  python3 /workspace/bench/correctness_diff.py
pkill -f vllm; sleep 3      # tắt baseline, nhường VRAM cho candidate spec
```
**→ DÁN VỀ:** dòng `captured 12 reference outputs`.

---

## PHASE B — Bước 0: CỔNG viability (spec rollback đúng không?)

Bật spec ngram, diff temp=0 với golden. Đây là cổng SỐNG-CHẾT của cả hướng 80.

### B.1 Serve spec ngram (LƯU Ý: KHÔNG --disable-log-stats để /metrics sống)
```bash
OMP_NUM_THREADS=1 taskset -c 0-2 vllm serve /workspace/model \
  --served-model-name LFM2.5-1.2B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.225 \
  --enable-prefix-caching --quantization fp8 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":8}' \
  > /workspace/spec.log 2>&1 &
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)" = 200 ] && { echo HEALTHY; break; }
  grep -qi "error\|assert\|traceback" /workspace/spec.log && { echo BOOT-FAIL; tail -30 /workspace/spec.log; break; }
  sleep 5
done
curl -s localhost:8000/metrics | grep spec_decode | head    # xác nhận counter tồn tại
```
**→ DÁN VỀ:** HEALTHY/BOOT-FAIL + các dòng `spec_decode`. Nếu BOOT-FAIL (assert kv-group) ⇒
báo về, đây là ngõ cụt hybrid-spec ở 0.25.1 (NEXT-80 Bước 0 nhánh FAIL: đọc source v1 spec).

### B.2 G0 — CORRECTNESS GATE (bắt buộc PASS)
```bash
MODE=diff BASE=http://localhost:8000 REF=/workspace/bench/golden.json \
  python3 /workspace/bench/correctness_diff.py; echo "EXIT=$?"
```
**→ DÁN VỀ:** dòng cuối `PASS`/`FAIL` + EXIT.
- **PASS (exit 0)** ⇒ rollback conv-state ĐÚNG trên 0.25.1 ⇒ sang Phase C + mở đường EAGLE.
- **FAIL (exit 1)** ⇒ spec đổi output ⇒ bug rollback. DỪNG nộp. Báo về để debug source.

---

## PHASE C — Bước 1: acceptance sweep (chỉ khi G0 PASS)

### C.1 Đo acceptance per-turn (nconv=1 để delta /metrics sạch — tách COPY vs gen)
```bash
BASE=http://localhost:8000 python3 /workspace/bench/spec_accept_probe.py 1
```
**→ DÁN VỀ:** bảng per-turn (accept%, acc_len, kind) + khối `/metrics TÍCH LUỸ` (MEAN_ACCEPT_LEN).
Kỳ vọng: turn COPY (restate/checklist) acc_len cao (2–4); turn gen thấp hơn.

### C.2 Đo net tbt của spec (so control A.2) — dùng chính loadgen + wl_probe
```bash
python3 /workspace/bench/loadgen_c1.py 30
BASE=http://localhost:8000 NCONV=3 OUT=300 python3 /workspace/bench/wl_probe.py | tail -5
```
**→ DÁN VỀ:** tbt_med spec vs control. Cần **tbt spec < tbt control** (spec thắng thật, không
chỉ acceptance đẹp mà verify-overhead ăn hết).

### C.3 SWEEP tham số (1 biến/lần) — chỉ nếu C.1/C.2 chưa đủ
Đổi 1 field trong --speculative-config rồi lặp B.1→C.1→C.2:
| Thử | num_speculative_tokens | prompt_lookup_min | prompt_lookup_max |
|---|---|---|---|
| s1 (mốc) | 4 | 2 | 8 |
| s2 | 6 | 2 | 8 |
| s3 | 4 | 1 | 4 |
| s4 | 6 | 3 | 12 |
**→ DÁN VỀ:** mean_accept_len + tbt_med cho từng thử. Giữ cấu hình acc_len cao nhất mà tbt còn giảm.

---

## PHASE D — Chốt & nộp portal

Điều kiện nộp: **G0 PASS + mean_accept_len ≥ 1.5 + tbt spec < control.**
1. Build image 0.25.1 + spec (image `image-fullfp8-v0251/` đã có patch Full-FP8; thêm spec qua
   `--speculative-config` ở compose — KHÔNG cần đổi Dockerfile). Pin digest.
2. `compose-spec-ngram-v0251.yml` đã sẵn cấu hình nộp (đổi digest nếu build image mới).
3. Nộp 1 slot portal, đọc score thật. Ghi vào `KNOWLEDGE_BASE.md §4.3` (score, tbt, acc_len, failed).

## PHASE E — Nếu ngram chưa đủ → EAGLE-3 (NEXT-80 Bước 2)
- Chỉ mở khi G0 PASS (đã xác nhận framework rollback đúng) nhưng acc_len ngram < 1.5.
- Check HF có EAGLE/EAGLE-3 head cho LFM2.5; nếu không → train head (self-distill vài nghìn
  prompt, vài GPU-giờ trên chính pod này). Acceptance EAGLE thường 3–4 ⇒ 80+.

---

## Bảng "dán về" nhanh
| Phase | Dán về |
| :-- | :-- |
| A | vllm version + control tbt/ttft + golden captured |
| B | HEALTHY/BOOT-FAIL + spec_decode metrics + **G0 PASS/FAIL** |
| C | acceptance per-turn + MEAN_ACCEPT_LEN + tbt spec vs control |
| C.3 | acc_len + tbt cho từng thử sweep |

## Chi phí
H100 ~$3/giờ. Phase A–C: ~1–2h. **Stop pod khi nghỉ** (volume rẻ giữ data).
