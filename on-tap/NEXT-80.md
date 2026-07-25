# NEXT-80 — Đường tới 80+ = SPECULATIVE DECODING (session sau triển khai)

> Đọc kèm `HANDOFF-RUNPOD.md` (dead-ends + rig). File này là kế hoạch tấn công.
>
> ✅ **TOOLING ĐÃ SẴN (2026-07-25):** thực thi turnkey ở **`on-tap/RUNBOOK-SPEC-80.md`**
>   (block-by-block Bước 0→1→2). `bench/spec_accept_probe.py` đã parse `/metrics`
>   (accept_rate + mean_accept_len, tách turn COPY vs gen). `compose-spec-ngram-v0251.yml`
>   = config nộp trên baseline 0.25.1 đúng. Chỉ còn thiếu: chạy trên pod H100.

## 1. Vì sao CHỈ còn spec decoding
- tbt=3ms là op-floor GPU (đã chứng minh mọi đòn quant/fusion/CPU cạn — HANDOFF). Với tbt kẹt 3ms, 80 là bất khả (cần ttft~10ms).
- tbt ~8.6đ/ms ⇒ hạ tbt là đòn đắt nhất. Cách DUY NHẤT hạ tbt mà không đổi model = **sinh nhiều token / 1 target-forward**:
  `effective_tbt = forward_time / accepted_tokens_avg`. Verify k token ở batch-1 ≈ cùng weight-read ⇒ forward_time ~ hằng.
  - accept **2 tok/forward** → tbt 3→1.5ms → s_tpot 0.60→0.89 → **+14đ → ~79**.
  - accept **3 tok** → tbt→1.0ms → s_tpot~0.95 → **~82**.
- Spec **lossless ở temp=0** nếu impl đúng (greedy: accept draft iff = target argmax) ⇒ qua được Accuracy Gate. Rủi ro = bug rollback state hybrid.

## 2. Trạng thái spec
- ❌ `draft_model` (draft LFM2 hybrid): 2 kv-group assert — đóng vĩnh viễn.
- ⚠️ n-gram: 1 lần thử acceptance≈0 — **chưa test đúng**. Workload có prompt copy-nặng ("restate the reference document as a checklist", "summarise… citing every parameter" — xem `correctness_diff.py`) ⇒ output copy từ context 2000+ tok ⇒ prompt-lookup PHẢI có hit. **RE-TEST nghiêm túc.**
- 🔓 EAGLE-3: chưa thử. Cần head cho LFM2.5 + xử conv-state rollback.

## 3. Kế hoạch (ranked, mỗi bước có gate)

### Bước 0 — CỔNG viability: hybrid + spec rollback đúng không (làm ĐẦU TIÊN)
> ❌ **ĐÃ CHẠY 2026-07-25 (pod H100 #2) → FAIL → ĐÓNG.** Kết quả isolate sạch:
> - ngram spec boot OK nhưng `correctness_diff` (temp=0) **FAIL 12/12**, token-duplication.
> - determinism-check (cùng config back-to-back) = **PASS** ⇒ cổng hợp lệ, không phải nhiễu.
> - `--mamba-cache-mode all`: bị loại trừ với `--enable-prefix-caching` (tụt `align`, không rollback);
>   khi tắt prefix-cache để `all` bật thật → mamba SSM-state được rollback nhưng **ShortConv conv_state KHÔNG**
>   (source: `mamba_utils.preprocess_mamba_all_specdec`/`postprocess_mamba_align_gpu` chỉ đụng mamba state;
>   `short_conv.py` update conv_state qua `state_indices_tensor_p`, không có nhánh rollback theo num_accepted).
> - ⇒ Bug nằm ở **target-state**, KHÔNG phụ thuộc proposer ⇒ **EAGLE (Bước 2) cũng dính** ⇒ đóng cả hướng
>   spec ở tầng config. Chỉ mở lại nếu chịu **patch source** (snapshot/restore ShortConv conv_state theo
>   num_accepted + rebuild image) — rủi ro cao, cộng no-prefix-cache đánh đổi TTFT. Khuyến nghị: chốt rổ ~65.

Bật spec ngram trên vllm 0.25.1, chạy `bench/correctness_diff.py` (temp=0 vs baseline fp8):
- **PASS** ⇒ conv-state rollback ĐÚNG trên 0.25.1 ⇒ mở đường EAGLE.
- **FAIL** ⇒ hybrid spec lỗi rollback (đúng lo ngại cũ). Debug: đọc source vllm 0.25.1 v1 spec + mamba state; hoặc chỉ dùng proposer không đụng target-state.
Lệnh mẫu: `vllm serve … --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":8}'`

### Bước 1 — n-gram / prompt-lookup (RẺ, không train) — thử ngay sau Bước 0 PASS
- Đo **acceptance per-turn** bằng `bench/spec_accept_probe.py` (✅ đã parse `/metrics`:
  `spec_decode_num_accepted_tokens` / `num_draft_tokens` / `num_drafts` ⇒ accept_rate +
  mean_accept_len; tách turn COPY vs gen; chạy `nconv=1` để delta per-turn sạch).
- Sweep `num_speculative_tokens` (3–6), `prompt_lookup_min/max`. Prompt-lookup quét CẢ context 2000-tok (shared+conv prefix) — nơi khả năng copy cao nhất.
- Gate nộp: `correctness_diff` PASS + net tbt (đo GPU-time/wl_probe) < control. Rồi build image 0.25.1 + nộp portal đo thật.
- Kỳ vọng: nếu ≥30–50% turn accept ~2–3 tok → mean tbt giảm đủ để 72–77.

### Bước 2 — EAGLE-3 (nếu ngram chưa đủ) — đòn mạnh nhất
- Check HF có EAGLE/EAGLE-3 head cho LFM2.5 chưa. Không có → **train head** (EAGLE head nhẹ, vài GPU-giờ trên chính H100 này; data = self-distill từ target trên vài nghìn prompt).
- Xử conv-state rollback (Bước 0 xác nhận framework). Acceptance EAGLE-3 thường 3–4 tok → tbt ~0.8–1.0ms → s_tpot 0.9+ → **80+**.
- Gate: correctness_diff PASS (spec lossless) + acceptance đo được + net tbt portal thắng.

### Bước 3 — cộng dồn (sau khi spec chạy)
- TTFT tune (chunked-prefill / max-num-batched-tokens / scheduler): 45→~35 = +~2đ. Spec KHÔNG giúp TTFT (token đầu vẫn full prefill).
- Fails 6-7 = variance portal (khó sửa); nhưng tbt thấp giảm tail-latency → có thể bớt vài fail.

## 4. Rig đo được gì (quan trọng)
- ✅ **acceptance rate, correctness (temp=0), net GPU-time** của spec — transfer THẲNG sang portal (không bị giới hạn BW như tbt tuyệt đối).
- ⇒ H100 rig ĐỦ để phát triển + gate spec. Chỉ số điểm cuối cùng đo trên portal.
- Nhớ: image nộp build từ **vllm 0.25.1** (luật 3 cho phép), không cần fork fastsse.

## 5. Nếu spec cũng chết
Đòn kernel còn lại duy nhất: **fuse path phi-linear** (conv/attn/norm, 0.85ms > linear) — nhưng norm/act+quant đã fuse sẵn, target hẹp (conv gating), portal không đo được, xác suất thấp. Cân nhắc dừng ở 65 + rổ an toàn (STRATEGY §4).

## 6. Checklist mở đầu session sau
1. Bật pod (volume có sẵn model+bench), serve baseline fp8, xác nhận tbt/ttft khớp (1.5ms / 37-55ms).
2. Bước 0 correctness-gate spec ngram. 3. Bước 1 acceptance sweep. 4. Build image 0.25.1 + nộp 1 slot đo portal.
