# SỔ ĐO RIG 27/07 — W1 (TTFT) và những gì nó lật lại

> 🔴 **ĐỌC TRƯỚC — hiệu chỉnh tối 27/07 bằng số của chính portal** (`PLAN-2026-07-27.md` §2):
> portal đo `{CPU_step, GPU_step} = {1.75, 3.48}` ⇒ **portal GPU-bound**, CPU **bị che hoàn toàn**.
> Rig thì ngược lại (CPU 3.33 > GPU 1.42) vì **CPU của rig chậm hơn portal ~1.9×**.
> ⇒ §5b–§5g dưới đây **đúng cho rig** nhưng **không chuyển sang portal**; đòn "cắt input-prep
> hybrid" (§6.2) **đã huỷ**. Mọi số CPU của rig phải **chia 1.9** trước khi quy sang portal.
> Phần vẫn còn giá trị nguyên vẹn: §0 (pin client), §1–§4b (bóc TTFT + 2 cờ frontend), §5h (đã chết).

> Máy: H100 SXM 80GB, vllm 0.25.1 pip, model `/workspace/model` `--quantization fp8`,
> `--gpu-memory-utilization 0.225`, `--max-model-len 32768`, `--enable-prefix-caching`,
> server `taskset -c 0-2`, `OMP/MKL=1`, `VLLM_LOGGING_LEVEL=WARNING`.
> Tải: `bench/ers_harness.py` RATE=8 SEED=42 (batch decode mean ≈ 27 ✓ khớp portal).
> Số đọc: **server-side histogram** (`ERS_HIST`, `ttft_split.py` trên delta `/metrics`).

## 0. ⚠️ LỖI PHƯƠNG PHÁP ĐÃ SỬA — client phải pin ra khỏi core 0-2

Chạy đầu tiên (client **không** pin): TTFT 94.5 ms, ERS_HIST 57.66.
Chạy y hệt nhưng client `taskset -c 8-40`: TTFT 44.1 ms, ERS_HIST 67.94.

⇒ **50 ms TTFT trong mọi số đo rig cũ là do client 70 thread giành core 0-2 với server.**
Từ nay: server `-c 0-2`, client `-c 8-40`. Không có ngoại lệ.

**Hệ quả tốt:** với pin đúng, rig cho **TTFT 43–48 ms** và **TPOT 3.4–3.6 ms**, so với portal
**ttft ~47–50 / TPOT 3.25**. Rig **tái lập đúng cả hai trục** ⇒ rig là mô phỏng TTFT/TPOT dùng được,
không chỉ "GPU-time tương đối".

## 1. Tên metric có thật trong 0.25.1 (W1 hàng 1) ✅

`vllm:time_to_first_token_seconds` · `vllm:request_queue_time_seconds` ·
`vllm:request_prefill_time_seconds` · `vllm:request_inference_time_seconds` ·
`vllm:request_decode_time_seconds` · `vllm:e2e_request_latency_seconds` ·
`vllm:request_time_per_output_token_seconds` · `vllm:inter_token_latency_seconds` ·
`vllm:request_prefill_kv_computed_tokens` · `vllm:iteration_tokens_total`.

Đẳng thức tự kiểm chéo (đúng tới 0.1 ms): `e2e = queue + prefill + rest + decode`,
`inference = prefill + decode`. ⇒ **`rest := ttft − queue − prefill` là frontend/CPU**, không suy đoán.

## 2. BÓC TTFT (W1 hàng 2) ✅ — **hàng đợi ≈ 0, chia đôi prefill / frontend**

| | queue | prefill | rest (frontend) | TTFT |
| :-- | --: | --: | --: | --: |
| access log ON, rep1/rep2 | 0.10 / 0.00 | 21.20 / 18.94 | **26.74 / 26.44** | 48.04 / 45.38 |
| access log OFF, rep1/rep2 | 0.11 / 0.00 | 20.79 / 18.99 | **23.47 / 24.23** | 44.37 / 43.22 |

- **Nhánh "queue_time chiếm phần lớn" bị loại**: queue = 0.0–0.1 ms. Scheduler **không** giữ request.
  ⇒ mọi lever kiểu `max-num-batched-tokens` / chính sách batching **chết** ở trục TTFT.
- Còn lại chia đôi: **prefill ~19–21 ms** và **frontend ~24–27 ms**.
- Prefill 19–21 ms trong khi `request_prefill_kv_computed_tokens = 286 tok`
  (prompt 2872, prefix-cache ăn 2586) ⇒ **286 token không thể tốn 20 ms GPU** trên H100.
  Prefill cũng là CPU/scheduling, không phải compute. Cần đo riêng (§5).

## 3. Frontend là **tokenizer**, và nó chạy **một thread duy nhất** 🔴

Container không có `CAP_SYS_PTRACE` ⇒ py-spy chết (`Permission denied`).
Thay bằng sampler in-process `bench/spyserve.py` (dump theo SIGUSR1, 100 Hz).

Top leaf frame khi đang tải:

```
100.00%  selectors.py:select            <- asyncio idle
 98.39%  threading.py:wait              <- idle
 79.58%  thread.py:_worker              <- thread pool idle
 20.42%  tokenization_utils_tokenizers.py:_encode_plus   <- LÀM VIỆC THẬT
  2.73%  core_client.py:process_outputs_socket
  2.57%  socket.py:send
```

Stack đầy đủ: `thread.py:_worker → base.py:_tokenize_prompt → hf.py:__call__ → _encode_plus`.

⇒ **Công việc CPU thật duy nhất ở frontend là tokenize prompt.** Không phải SSE, không phải
prometheus (<1%), không phải HTTP.

🔴 **`renderer_num_workers` mặc định = 1** (`vllm/config/model.py:333`). Tức là **toàn bộ 420 lần
tokenize ~2900 token đi qua ĐÚNG MỘT thread**. Đây là nghi phạm số 1 của 24–27 ms.

## 4. `--disable-uvicorn-access-log` — thật nhưng nhỏ ✅

`VLLM_LOGGING_LEVEL=WARNING` **không** tắt access log của uvicorn (log server đầy dòng
`INFO: 127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200 OK`). Bài nộp portal đang trả giá này.

Δrest = **−2.7 ms**, Δttft = **−2.9 ms**, ổn định trên cả 2 rep. Quy ra ≈ **+0.2 ERS**.
⇒ Dưới ngưỡng phân giải portal, **không đủ để tiêu lượt một mình**, nhưng miễn phí và
output-preserving ⇒ **gộp vào bài chốt**.

## 5. 🔴 NGHI VẤN LỚN — TPOT có thể là CPU-bound, không phải băng thông

Rig chạy **không MPS** (đủ 132 SM, băng thông 3.3 TB/s ≈ 5.6× MiG portal), weight fp8 ~1.2 GB.
Mô hình STRATEGY §2 (`TPOT = 2.05 + 1.63×GB`, slope = 600 GB/s) dự đoán rig phải nhanh hơn hẳn.

Đo được: **TPOT 3.40–3.65 ms**, gần như **bằng portal (3.25)**.

Biến duy nhất giống nhau giữa hai máy: **3 CPU core**. ⇒ giả thuyết: sàn TPOT là
**vòng lặp engine + output processor + detokenize incremental (Python) trên 3 core**,
không phải đọc weight, cũng không phải kernel attention.

Nếu đúng, §2b của STRATEGY ("sàn = decode attention") **giải thích sai cơ chế** — số nsys đo
attention 54% *thời gian GPU busy*, nhưng nếu GPU chỉ bận ~1/3 thời gian step thì attention
chỉ là ~18% của step. Cần thí nghiệm phân xử ở §6.

## 5b. 🔴 `--no-async-scheduling` bóc được **CPU/step vs GPU/step**

| config (nền: noaccesslog + nw2) | TTFT | prefill | rest | **TPOT** | ERS_HIST |
| :-- | --: | --: | --: | --: | --: |
| nền (async mặc định) | 37.8 | 19.1 | 18.7 | **3.33** | **70.62** |
| `--async-scheduling` (ép bật) | 37.9 | 19.1 | 18.8 | 3.42 | 69.96 |
| `--no-async-scheduling` | 33.6 | 13.5 | 20.1 | **4.75** | 61.10 |
| `--api-server-count=2` | 42.2 | 21.3 | 20.8 | 3.33 | 69.64 |

- `--async-scheduling` = **no-op ⇒ async scheduling ĐÃ BẬT sẵn** trong 0.25.1.
- Tắt nó: TPOT **3.33 → 4.75** (+1.42). Với async, TPOT ≈ max(CPU_step, GPU_step);
  không async, TPOT ≈ CPU_step + GPU_step. ⇒ **{CPU_step, GPU_step} = {3.33, 1.42}**.
- Ai là 3.33? Portal (MiG, SM ít hơn ~7×) đo TPOT **3.25** ≈ rig full-SM **3.33**.
  Cắt SM 7× mà TPOT không đổi ⇒ **3.33 là CPU_step, GPU_step ≈ 1.42**.
  Đang xác nhận trực tiếp bằng MPS 14% trên chính rig (§6.2).
- `--api-server-count=2`: **không giúp** (69.64 vs 70.62) ⇒ frontend không bị nghẽn GIL một process.

⇒ Nếu MPS xác nhận: **sàn TPOT là vòng lặp Python của EngineCore trên 3 core, KHÔNG phải
decode attention.** STRATEGY §2b đo đúng (attention = 54% *GPU busy*) nhưng **kết luận sai**:
GPU chỉ chiếm ~43% của step, nên attention ≈ 23% của step, và cắt đôi attention chỉ đáng
~0.2 ms chứ không phải 0.75 ms.

## 5c. `--renderer-num-workers` — thật nhưng nhỏ

`renderer_num_workers` mặc định **1** ⇒ toàn bộ tokenize đi qua 1 thread. Nâng lên 2/3/4
(so rep cuối, cùng nền noaccesslog): rest **24.2 → 20.5 / 21.5 / 20.9**. Δ ≈ −3.5 ms, bão hoà ở 2.
⇒ dùng **2**. Không có gì thêm ở 3–4 (chỉ có 3 core).

## 5d. Số core: **thêm core làm TỆ ĐI** (đừng đổ tại băng thông CPU)

Cùng config, chỉ đổi `taskset` của server (client luôn ở 8-40):

| cores | rest | TTFT | TPOT | ERS_HIST |
| :-- | --: | --: | --: | --: |
| 0-2 | **18.7** | **37.8** | 3.33 | **70.62** |
| 0-5 | 47.9 | 68.2 | 3.67 | 60.08 |
| 0-11 | 53.8 | 75.5 | 3.66 | 58.04 |

⇒ Nút cổ chai **không phải số core**; nhiều core làm torch/threadpool nở thread và đập nhau.
Hệ quả: mọi lever kiểu "song song hoá" đã hết cửa — phải **cắt việc CPU mỗi bước / mỗi request**.

## 4b. ✅ A/B CHỐT: nền vs ứng viên — **+0.94 ERS**, và rig **phân giải được 0.9 ERS**

2 lần boot mỗi bên, xen kẽ (base1, cand1, base2, cand2), lấy **rep ấm (2 và 3)** của mỗi boot
⇒ 4 số mỗi bên. Nền = đúng cờ portal đang dùng; ứng viên = thêm
`--disable-uvicorn-access-log --renderer-num-workers=2`.

| | ERS_HIST (4 rep ấm) | mean | sd | mean TTFT |
| :-- | :-- | --: | --: | --: |
| nền | 69.04 · 68.78 · 68.54 · 68.98 | **68.84** | 0.23 | 43.0 |
| ứng viên | 69.89 · 69.83 · 69.79 · 69.61 | **69.78** | 0.12 | 41.2 |

**Δ = +0.94 ERS** (sd gộp 0.18 ⇒ tách bạch hoàn toàn). Δttft = −1.8 ms.

🔑 **Đây mới là kết quả vận hành quan trọng nhất của ngày:** với client pin đúng và rep ấm,
**rig phân giải được Δ ≈ 0.3–0.9 ERS**, trong khi portal cần **≥4 ERS**. Rig **hơn portal ~10×**
về độ nhạy ⇒ mọi A/B từ nay làm ở rig, portal chỉ để chốt.
⚠️ Luôn bỏ **rep 1 sau mỗi lần boot** (lạnh, prefix-cache rỗng: 65–68 vs 69–70).

## 5e. MPS: **GPU dư ít nhất 2×** ⇒ xác nhận CPU-bound ở regime rig

| MPS (% SM) | TPOT | ERS_HIST |
| :-- | --: | --: |
| off (132 SM) | 3.33 | 70.62 |
| 50% (~66 SM) | 3.39–3.42 | 69.5–69.8 |
| 14% (~18 SM) | 4.46–4.98 | 36.6 (bão hoà, TTFT sập) |

Cắt **một nửa SM ⇒ TPOT không đổi**. ⇒ ở rig, `GPU_step ≈ 1.4–1.6 ms` và **TPOT bị chặn bởi CPU**.

## 5f. TPOT tuyến tính theo batch ⇒ **chi phí Python MỖI request MỖI step**

Quét RATE (cùng config):

| RATE | batch mean | TPOT | prefill | rest |
| --: | --: | --: | --: | --: |
| 2 | 7.8 | **2.15** | 14.1 | 19.2 |
| 4 | 15.8 | **2.61** | 15.9 | 17.9 |
| 8 | 27.0 | **3.33** | 19.1 | 18.7 |

Khớp `TPOT ≈ 1.60 + 0.064 × B` (ms). Intercept 1.60 ≈ `GPU_step` đo độc lập ở §5b/§5e ✓.
Slope = **64 µs / request / step** — đây chính là sàn, và nó là **Python**, không phải byte.

Chú ý: **`rest` ~18–19 ms bất biến theo tải** (RATE 2 và RATE 8 như nhau) ⇒ **không phải hàng đợi**,
mà là **độ trễ tuần tự mỗi request** = tokenize ~2900 token prompt. Khớp spy (§3).

## 5g. Profile EngineCore (process riêng) — sàn là **bookkeeping hybrid/mamba**

py-spy không attach được ⇒ cài sampler vào **mọi** process qua `sitecustomize` →
`_spyauto.py` (dump định kỳ). Leaf frames của `VLLM::EngineCore` khi đang tải:

```
 18.2%  v1/worker/utils.py:zero_block_ids          <- zero KV block mới (bắt buộc cho conv/mamba)
 16.9%  utils.py:copy_to_gpu                       <- H2D input tensors
 11.9%  socket.py:send                             <- ZMQ trả output
  7.5%  gpu_model_runner.py:_prepare_inputs
  6.9%  short_conv.py:forward_cuda
  5.0%  utils.py:mamba_get_block_table_tensor
  4.4%  triton driver.py:__call__   4.1% _ops.py:__call__
  2.8%  mamba_utils.py:collect_mamba_copy_meta     2.8% sampler.py:forward
```

⇒ ~**43% CPU/step là block-table + zero-block của kiến trúc hybrid LFM2**, ~17% là H2D copy.
Không có gì ở đây là attention. **Sàn TPOT = input-prep Python của hybrid model trên 3 core.**

## 5h. Những thứ đã thử và **CHẾT** hôm nay (đừng thử lại)

| Đòn | Kết quả | Vì sao |
| :-- | :-- | :-- |
| `--api-server-count=2` | 69.64 vs 70.62 | KB §7 đúng: `run_server()` không fork |
| `--async-scheduling` | no-op | đã default-on |
| `--max-num-seqs=64 / 128` | 66.94 / 69.53 | buffer không sized theo max_num_seqs |
| `--prefix-caching-hash-algo=xxhash` | 70.48 / 69.47 | sha256-pickle không nằm trên critical path |
| `--renderer-num-workers=3/4` | = 2 | chỉ có 3 core |
| thêm CPU core cho server | tệ hơn nhiều | torch/threadpool nở thread |
| `max_num_partial_prefills` | **không tồn tại trong v1** | `v1/core/sched/scheduler.py` không đọc nó |
| `cudagraph_mode` | đã `FULL_AND_PIECEWISE`, mixed **đã** được capture piecewise | log boot |

## 6. Đường tới 80 — bản đồ đã đổi

Ngân sách TTFT hiện tại (rig, config tốt nhất): **37.8 = 0.0 queue + 19.1 prefill + 18.7 frontend**.

1. **Frontend 18.7 ms = tokenize lại toàn bộ ~2900 token mỗi turn.** Turn 2–6 chỉ thêm ~150 token mới
   nhưng vẫn tokenize lại từ đầu. **Không có cờ CLI nào sửa được** — cần patch image: cache
   tokenization theo prefix chuỗi đã render. Nếu về ~2 ms ⇒ TTFT 37.8 → ~21 ⇒ **+4 ERS**.
   Cổng an toàn: so token-ids nối-từ-cache vs tokenize-toàn-bộ trên đúng 420 turn của workload;
   bằng nhau ⇒ output-preserving tuyệt đối.
2. **TPOT: 64 µs/request/step của input-prep hybrid.** Cắt được nửa ⇒ TPOT 3.33 → 2.47 ⇒ **+5 ERS**.
   Mục tiêu cụ thể có tên hàm: `zero_block_ids`, `copy_to_gpu`, `mamba_get_block_table_tensor`.
   Đây là **CPU**, nên đo được trên rig, và **không** cần đụng kernel attention.
3. Attention (STRATEGY §2b) tụt xuống ưu tiên 3: ở rig nó nằm trong 1.4–1.6 ms GPU vốn đã bị CPU
   che khuất. Chỉ đáng làm nếu chứng minh được portal ở regime GPU-bound (chưa chứng minh).

## 7. Còn nợ

- Chạy lại đúng W2 (FA2 / TRITON_ATTN) — cần nsys (`apt-get install nsight-systems-2026.1.3`
  hôm nay **fail exit 100**, phải `apt-get update` trước). Ưu tiên thấp theo §6.
- GPQA cho lm_head int4 (nợ từ 26/07).
- Xác nhận portal có ở regime CPU-bound không: nộp 1 lượt với `--disable-uvicorn-access-log
  --renderer-num-workers=2` **chỉ khi** gộp cùng một đòn lớn hơn (một mình chỉ +1.8 ERS).
