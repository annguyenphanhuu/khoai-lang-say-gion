# HANDOFF SESSION #3 — Concurrency discovery + SGLang direction (2026-07-25)

> Đọc kèm: `STRATEGY.md` (🚩 header session #3), `on-tap/KNOWLEDGE_BASE.md`, `bench/ers_harness.py`.
> Mục đích: session sau (chat mới) tiếp tục được NGAY mà không mất context.

## 0. TÓM TẮT 30 GIÂY
- Mọi bench cũ đo **concurrency=1**; portal chấm **70 hội thoại đồng thời**. Đã viết harness đồng thời
  trung thực (`bench/ers_harness.py`) + đo thật trên pod H100.
- **Trần config-lossless ~65–66 XÁC NHẬN** (baseline fp8+prefix-cache, RATE=8 ≈ portal 64.67).
- **Không knob scheduling nào thắng default.** vLLM v1 default đã tối ưu.
- **Phát hiện lớn: TPOT tăng theo batch** (8→2.2ms, 27→3.6ms) ⇒ ở batch cao decode KHÔNG thuần
  weight-bound ⇒ **4-bit weight VÀ speculative decoding đều vô dụng/hại dưới concurrency** (spec thêm
  compute vào step đã tải). Cả `NEXT-80.md` và P5-W4A8 sai tiền đề.
- **Hướng 75+ còn lại duy nhất = hạ per-token compute ở batch cao** = đổi stack (SGLang/TRT-LLM) hoặc
  fuse kernel. ĐANG THỬ: **SGLang cho LFM2.5** (kết quả ghi ở §5 bên dưới).

## 1. TRUY CẬP POD (RunPod SSH proxy) — CƠ CHẾ ĐÃ HOẠT ĐỘNG
- Lệnh: `ssh tsc989q8dlcj8t-64411fec@ssh.runpod.io` (⚠️ endpoint ĐỔI mỗi lần tạo pod mới; user id_ed25519
  KHÔNG tồn tại — key mặc định/không cần key vẫn vào). **RunPod proxy đòi PTY** ⇒ dùng `ssh -tt`.
- **Proxy ép interactive shell** ⇒ KHÔNG chạy được `ssh host 'cmd'`. Phải **feed lệnh qua stdin + `exit`**:
  `printf 'cmd1\ncmd2\nexit\n' | timeout 120 ssh -tt -o StrictHostKeyChecking=accept-new <endpoint>`
- PTY thêm escape-code + echo lệnh ⇒ lọc output: `| tr -d '\r' | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\x1b\][0-9;]*//g'`
- **Copy file lên pod:** base64. File nhỏ (<~1.5KB b64): 1 dòng `echo <b64> | base64 -d > dst`. File lớn
  (harness ~10KB): **CHUNK** b64 thành mảnh ≤1200 ký tự, `printf '%s' 'piece' >> /tmp/x.b64` từng dòng,
  rồi `base64 -d /tmp/x.b64 > dst` (1 dòng >~1500 ký tự bị PTY cắt hỏng). Helper local đã dùng:
  `/tmp/podrun.sh <local.sh>` (chạy script) và `/tmp/podcp_chunked.sh <local> <remote>` (copy chunked).

## 2. TRẠNG THÁI POD / VOLUME
- **`/workspace` SỐNG qua Stop/Start; container (pip install) CHẾT.** Sau mỗi restart phải cài lại vllm.
- Model: `/workspace/model` (LiquidAI/LFM2.5-1.2B-Instruct, đủ file). Bench: `/workspace/bench` →
  symlink tới `/workspace/repo/bench` (repo clone: `github.com/annguyenphanhuu/khoai-lang-say-gion`).
- **Cài lại vllm 0.25.1 (nhanh, từ cache):** `PIP_CACHE_DIR=/workspace/.pipcache pip install "vllm==0.25.1"`
  (cache đã populate ~6.2G ⇒ không tải lại, ~1–2 phút). ⚠️ **KHÔNG cài venv trên `/workspace`** — FUSE
  network FS ⇒ giải nén cực chậm (>15 phút). Cài vào system python (container disk) mỗi session.
- **Sau restart, CLI `vllm` KHÔNG trên PATH** ⇒ luôn dùng `python3 -m vllm.entrypoints.openai.api_server`.
- Driver pod: CUDA 13.0, H100 80GB. `libnvrtc.so.13` thiếu ⇒ `deep_gemm` import fail (WARNING vô hại).

## 3. CÁCH ĐO (ers_harness.py) — QUAN TRỌNG
- Serve emulate portal MiG-18GB trên H100: `taskset -c 0-2 OMP_NUM_THREADS=1 python3 -m
  vllm.entrypoints.openai.api_server --model /workspace/model --served-model-name LFM2.5-1.2B-Instruct
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.225 --enable-prefix-caching --quantization fp8`
  (⚠️ **KHÔNG** `--disable-log-stats` khi đo — harness cần `/metrics`).
- Chạy: `BASE=http://localhost:8000 NCONV=70 TURNS=6 OUT=300 RATE=8 SEED=42 python3 bench/ers_harness.py`
- **RATE=8 = điểm calibrate** (batch decode mean 27 ⇒ ERS_HIST ≈ 66 ≈ portal cpu-lean 64.67).
- Chấm ERS từ **histogram server-side vLLM** (portal-faithful, miễn artifact client Python-thread):
  - TTFT: `vllm:time_to_first_token_seconds` (bucket mịn, E[s] chuẩn). F=10ms C=400ms.
  - TPOT: `vllm:request_time_per_output_token_seconds` — ⚠️ **bucket đầu = 10ms quá thô** (mọi req <10ms
    dồn 1 bucket ⇒ E[s] hằng số 0.309 vô nghĩa). ⇒ harness dùng **s(mean từ _sum/_count)**. F=1ms C=10ms.
  - Harness diff histogram trước/sau ⇒ **KHÔNG cần restart server giữa các config**.
- Client-side TTFT bị thổi (~2.5x) do 70 thread Python tranh GIL ⇒ **CHỈ tin số SERVER-SIDE / ERS_HIST**.

## 4. KẾT QUẢ ĐO ĐÃ CÓ (H100 rig, RATE=8, batch~27)
| Config | ERS_HIST | mean TTFT | mean TPOT |
| :-- | :--: | :--: | :--: |
| **baseline (fp8+prefix-cache, default sched)** | **66.0** | 48ms | 3.6ms |
| max-num-batched-tokens 1024/2048/4096 | 59–61 | 84–88 ↑ | ~3.6 |
| max-num-batched-tokens 8192 | 65.2 | 53 | 3.6 |
| max-num-seqs 128 | 60.5 | 82 ↑ | 3.7 |
| long-prefill-token-threshold 2048 | 64.3 | 56 | 3.5 |
| kv-cache-dtype fp8 | 62–66 (bất ổn) | 44–78 | 3.7 (không giảm) |

**TPOT vs batch (baseline):** batch 8→2.17ms · 12→2.43 · 16→2.67 · 23→3.16 · 27→3.6 · 31→3.85.
⇒ TPOT tăng theo batch = decode compute/KV-influenced ở batch cao (KHÔNG thuần weight-bound).

## 4b. SPEC DECODING (session #2, đã đóng)
- ngram spec: boot OK nhưng **correctness G0 FAIL 12/12** (token-duplication) = bug rollback conv-state
  ShortConv. `--mamba-cache-mode all` không cứu (bị loại trừ với `--enable-prefix-caching` → tụt `align`;
  khi tắt prefix-cache để `all` bật thật thì mamba SSM-state rollback OK nhưng **ShortConv conv_state
  KHÔNG** → vẫn FAIL). Source: `mamba_utils.preprocess_mamba_all_specdec` chỉ đụng mamba state;
  `short_conv.py` update conv_state qua `state_indices_tensor_p`, không có nhánh rollback theo num_accepted.
- draft_model: boot crash (2 kv-group assert). ⇒ **cả họ spec đóng** ở vllm 0.25.1 cho LFM2 hybrid.
- **BONUS lý do đóng (session #3):** kể cả nếu sửa được correctness, spec vẫn HẠI dưới batch 27 (thêm compute).

## 5. HƯỚNG ĐANG THỬ: SGLang cho LFM2.5  ← CẬP NHẬT Ở ĐÂY
Mục tiêu: kernel/RadixAttention của SGLang có thể hạ per-token compute (hạng bottleneck mới) → TPOT < 3.6ms
ở batch 27 → 75+. **Câu hỏi sống-chết: SGLang có support LFM2.5 hybrid (ShortConv/Lfm2) không?**
- Nếu support: serve bằng SGLang (`python -m sglang.launch_server --model /workspace/model --port 8000 ...`),
  chạy `bench/ers_harness.py` y hệt (cùng OpenAI API `/v1/chat/completions` + `/metrics`), so ERS vs 66.
- Nếu KHÔNG support: ghi lại (LFM2 hybrid niche) → cân nhắc TRT-LLM hoặc dừng ở rổ ~65.
- Correctness gate BẮT BUỘC nếu đổi stack: `bench/correctness_diff.py` vs golden fp8 (đã có
  `/workspace/bench/golden.json` từ baseline vllm; hoặc chụp lại golden từ vllm fp8 rồi diff SGLang).

### KẾT QUẢ SGLANG (điền khi chạy):
- [ ] SGLang install được trên pod? (pip install "sglang[all]")
- [ ] LFM2.5 load được? (arch Lfm2ForCausalLM có trong registry SGLang?)
- [ ] Nếu chạy: ERS_HIST @ RATE=8 = ___ (so baseline 66); TTFT ___ TPOT ___
- [ ] correctness_diff vs golden fp8: PASS/FAIL

## 6. NHÁNH DỰ PHÒNG NẾU SGLANG KHÔNG ĐI ĐƯỢC
- TensorRT-LLM (kernel Hopper mạnh nhất, nhưng LFM2 hybrid support còn hiếm hơn SGLang).
- Tự fuse kernel hybrid trong vllm (ShortConv+norm+quant) — effort rất cao, gain chưa chắc.
- Nếu cả 3 fail: **chốt rổ an toàn ~65** (cpu-lean 64.67 / w4a8 65.06) trước hạn 30/07/2026.
