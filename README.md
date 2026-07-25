# Đề bài & Quy định — Vòng 1 Sơ loại (CẬP NHẬT 2026-07-18)

> **Thời gian vòng 1: 02/07/2026 – 30/07/2026.**
> Bản 18/07 (sau đợt regrade): mô tả workload chuyển sang file spec
> (`grading-workload-spec.json`), bỏ warmup — chấm cả 420 request, biên độ
> tie-break nới thành ≤1–3 điểm, danh mục tối ưu bỏ Disaggregated P/D và
> NCCL. Những thay đổi phá vỡ chiến lược được đánh dấu ⚠️.

## 1. Nhiệm vụ & Hạ tầng

Cuộc thi mô phỏng bài toán LLM serving trong môi trường production: hệ thống
không chỉ cần throughput cao mà còn phải cân bằng **độ trễ thấp**, **độ chính
xác ổn định** và **khả năng vận hành hiệu quả trên tài nguyên GPU hữu hạn**.
Đây là bài toán tối ưu serving có ràng buộc chất lượng: tối đa hóa ERS trên
workload trace cố định của BTC, đồng thời vượt qua Accuracy Gate. Hiệu năng
được chấm liên tục trên từng request thay vì theo cơ chế đạt/không đạt đơn
thuần.

**Nhiệm vụ (bản 18/07/2026):** Triển khai và tối ưu một LLM inference
server cho mô hình **LFM2.5-1.2B-Instruct** xử lý workload trace multi-turn
mô phỏng traffic production. Vòng online chỉ tối đa hoá **ERS**; **Accuracy
Gate chỉ chạy SAU vòng online**, trên tối đa 5 submissions đội tự chọn.

⚠️ Workload giờ được mô tả bằng **file spec** (bản đã tải:
`grading-workload-spec.json`) thay cho trace liệt kê từng request. Ý nghĩa
các trường (theo định nghĩa chính thức của BTC):

| Trường | Ý nghĩa | Giá trị hiện tại |
| --- | --- | ---: |
| `num_conversations` | Số hội thoại độc lập chạy **đồng thời** | 70 |
| `user_turns_per_conversation` | Số lượt hỏi của user mỗi hội thoại | 6 |
| `total_request` | Tổng số request — **TẤT CẢ được chấm, không còn warmup** ⚠️ | 420 |
| `shared_system_prefix_tokens` | System prefix GIỐNG NHAU trên mọi hội thoại | 1000 |
| `per_conversation_prefix_tokens` | Ngữ cảnh riêng từng hội thoại (bổ sung vào input turn 1) | 1000 |
| `new_user_tokens_per_turn` | Token prompt user mỗi turn (turn 1 có thêm 2 khối prefix) | 150 |
| `output_tokens_per_turn_pinned` | Token output mỗi turn (pin cứng) | 300 |
| `arrival` | Nhịp đến của request | Poisson, seed 42 |

⚠️ Hệ quả cấu trúc: context là **append-only** (system prefix chung + prefix
hội thoại + lịch sử tăng dần 450 token/turn) → prefix caching hit thật trên
turn 2–6, chỉ prefill ~150 token mới; turn 1 prefill ~2150 token.

**Hạ tầng & Môi trường đánh giá:** benchmark tự động trên hệ thống BTC;
thí sinh serve endpoint trên 1 instance MiG, BTC benchmark trực tiếp vào
endpoint:

- Hardware: 1 instance **MiG H200 (18GB VRAM, 3 core CPU, 8GB RAM)** cấp
  phát tự động mỗi lượt chấm (không đổi).
- Host: **Ubuntu 24.04 LTS, NVIDIA driver 590.x (CUDA 13.x)** ⚠️ (cũ:
  22.04 / CUDA 12.x).
- Model: `LiquidAI/LFM2.5-1.2B-Instruct` —
  weights: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct

## 2. Cách tính điểm

Các khái niệm độ trễ:

- **TTFT (Time-To-First-Token):** thời gian từ lúc gửi request đến khi nhận
  token đầu tiên.
- **TPOT (Time-Per-Output-Token):** thời gian giữa hai token liên tiếp trong
  output stream; công thức chấm dùng TPOT trung bình của request.

**ERS** = trung bình S_request trên N request, S_request ∈ [0, 1]:

- S_request = **0** nếu lỗi / timeout / trả về 0 token.
- S_request = `w·s_ttft + (1−w)·s_tpot` nếu xử lý thành công.
- `s_ttft = [clamp((C_ttft − TTFT)/(C_ttft − F_ttft), 0, 1)]^γ`
- `s_tpot = [clamp((C_tpot − TPOT_mean)/(C_tpot − F_tpot), 0, 1)]^γ`

Tham số cấu hình ⚠️ (siết mạnh so với cũ):

| Ký hiệu | Ý nghĩa | Giá trị MỚI | (Giá trị cũ) |
| --- | --- | ---: | ---: |
| F_ttft | Floor của TTFT | **10 ms** | 100 ms |
| C_ttft | Ceiling của TTFT | **400 ms** | 1500 ms |
| F_tpot | Floor của TPOT | **1 ms** | 20 ms |
| C_tpot | Ceiling của TPOT | **10 ms** | 45 ms |
| γ | Hệ số lũy thừa | 2 | 2 |
| w | Trọng số TTFT | 0.5 | 0.5 |

Với từng thành phần, độ trễ bằng hoặc thấp hơn Floor nhận `s = 1`; chạm hoặc
vượt Ceiling nhận `s = 0`. Hàm `clamp(x, 0, 1)` giữ điểm trong đoạn `[0, 1]`,
còn `γ` quyết định độ dốc của đường phạt giữa hai cận.

⚠️ Hệ quả: TBT ~13ms của config cũ giờ **VƯỢT ceiling TPOT 10ms → s_tpot
= 0** nếu không cải thiện tốc độ decode. TPOT không còn "kịch trần miễn
phí" — nó trở thành nửa số điểm phải giành lại.

**Accuracy Gate — sau vòng online:**

- KHÔNG chấm GPQA trên từng lượt nộp online ⚠️.
- Leaderboard online chủ yếu phản ánh ERS và chỉ mang tính tạm thời cho đến
  khi hoàn tất hậu kiểm và Accuracy Gate.
- Sau vòng online: đội tự chọn tối đa **5 submissions** → BTC (1) hậu
  kiểm tính hợp lệ phương án; (2) dựng endpoint, chạy **GPQA full**
  (lm_eval / bench-gpqa-diamond.sh) ⚠️ (cũ: 100 câu cố định mỗi run).
- Các submission được chọn phải dùng đúng image/digest đã nộp trong vòng
  online, không được đổi image sau khi chọn. Với mỗi bài hợp lệ, BTC dựng lại
  endpoint OpenAI-compatible và chạy `lm_eval` trên GPQA Diamond full với
  bộ lọc strict-match.
- Δ = Accuracy_baseline − Accuracy_submission, baseline BF16 mặc định
  **0.4**.
- Hàm phạt (Δ tính theo tỷ lệ, tương đương 10/16 điểm phần trăm cũ):

```
f(Δ) = 1.0                      nếu Δ ≤ 0.10
     = 1.0 − (Δ − 0.10)/0.06    nếu 0.10 < Δ < 0.16
     = 0.0                      nếu Δ ≥ 0.16
```

- Điểm mỗi submission hợp lệ: `Score = 100 × ERS × f(Δ)` (ERS lấy từ lần
  chấm online của đúng bài đó). **Điểm đội = Score tốt nhất trong các
  bài còn hợp lệ.**

## 3. Không gian Tối ưu

⚠️ **CHỈ được dùng serving framework vLLM** (cũ: toàn quyền chọn
framework). Các hướng cho phép:

- **Quantization:** chỉ các kỹ thuật **Online Quantization** ⚠️ (không
  còn AWQ/GPTQ/checkpoint pre-quantized).
- **KV Cache & Memory:** Paged Attention; KV cache quantization
  (FP8, INT8); Prefix caching và Semantic caching; Offloading CPU/NVMe.
- **Serving & Scheduling:** Dynamic/Continuous batching; Speculative
  decoding; Memory-aware scheduling. ⚠️ (bản 18/07 BỎ "Disaggregated
  prefill/decode serving" khỏi danh mục)
- **System & Runtime:** custom CUDA/Triton kernels; Fused attention
  kernels (FlashAttention, FlashInfer); memory layout & CUDA Graphs.
  ⚠️ (bản 18/07 BỎ "NCCL communication optimization")

## 4. Nộp bài & Tài nguyên

Quy trình: Develop & Package (Docker image) → Push public lên Docker Hub
→ Submit `docker-compose.yml` qua Portal → hệ thống pull image, dựng
container trên MiG H200, healthcheck, chạy benchmark **ERS** (không chạy
GPQA mỗi lượt) → Leaderboard cập nhật theo ERS.

Sau vòng online: đội chọn tối đa 5 submissions → BTC hậu kiểm hợp lệ →
chấm GPQA full → chốt Score.

- ⚠️ Bản 18/07 không còn nhắc file trace công khai; workload công bố qua
  **file spec** (mục 1, `grading-workload-spec.json`). Prompt thật chỉ được
  gửi tới endpoint lúc chấm — ngăn pre-bake/học tủ. (`trace-round1.jsonl` và
  `trace_grading_public.jsonl` trong repo là các bản của nội quy cũ.)
- Docker image baseline: `vllm/vllm-openai:v0.22.1`
  (https://hub.docker.com/layers/vllm/vllm-openai/v0.22.1/images/sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5)

File `docker-compose.yml` mẫu:

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3 #Don't change this to vllm-server
      - -m  #Don't change this to vllm-server
      - vllm.entrypoints.openai.api_server #Don't change this to vllm-server
    command:
      - --model=/model #Don't change this to vllm-server
      - --served-model-name=LFM2.5-1.2B-Instruct #Don't change this to vllm-server
      - --host=0.0.0.0 #Don't change this to vllm-server
      - --port=8000 #Don't change this to vllm-server
      - --max-model-len=32768
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 5. Quy định & Phòng chống gian lận ⚠️ (MỤC MỚI, RẤT QUAN TRỌNG)

- Phase 1 áp dụng nghiêm ngặt nguyên tắc Anti-Cheating (tab Tổng quan).
- **Yêu cầu cốt lõi:** giải pháp phải **serving LLM một cách trung thực**
  trên GPU được cấp, có tinh thần sẵn sàng phục vụ người dùng production.
  Thủ thuật đánh lừa phép đo hoặc chỉ hoạt động trên workload chấm đều được
  xem là vi phạm nghiêm trọng.
- Các hành vi **bị nghiêm cấm**:
  1. **Pre-bake, hardcode kết quả, cơ chế dual-path hoặc lách luật
     (gaming) phương pháp đo lường.** ⚠️ — điều khoản này cấm trực tiếp
     họ kỹ thuật early-first-token (exp8) và mọi cơ chế "trace đi đường
     riêng, GPQA đi đường riêng", bao gồm đệm rỗng hoặc cắt ngắn chuỗi sinh
     trái phép để né hậu kiểm.
  2. Thực hiện các lệnh gọi mạng bên ngoài.
  3. Can thiệp trái phép vào tokenizer hoặc weights của mô hình.
  4. Làm bẩn tài nguyên, làm lộ dữ liệu hoặc tráo đổi Docker image sau khi đã
     nộp bài.
- **Hậu kiểm:** online chỉ chấm ERS tự động; sau vòng thi, 5 submissions
  được chọn sẽ bị **hậu kiểm tính hợp lệ** trước khi chấm GPQA ⚠️ —
  nghĩa là BTC sẽ đọc/soi phương án, image pin, cấu hình, log, hành vi và
  luồng serving chứ không chỉ đo số. Hoạt động rà soát có thể diễn ra định kỳ
  hoặc đột xuất.
- Xử lý vi phạm: tùy mức độ, BTC có thể hủy kết quả, điều chỉnh xếp hạng, thu
  hồi điểm hoặc loại cá nhân/đội khỏi giải; quyết định được thông báo qua
  email kèm lý do tóm tắt.

## 6. Hậu kiểm, Tie-break & Khiếu nại ⚠️ (MỤC MỚI)

Tiêu chí phụ khi điểm bám sát nhau (chênh trong biên độ nhiễu **≤ 1–3
điểm** ⚠️ — bản 18/07 NỚI RỘNG từ 1–2), theo thứ tự ưu tiên:

1. Mức độ suy giảm độ chính xác.
2. Chỉ số **p95 TTFT**.
3. Tốc độ sinh văn bản.
4. Thời điểm nộp bài (ưu tiên nộp sớm hơn).

Quy trình hậu kiểm & chấm lại:

- Điểm tự động và thứ hạng online chưa phải kết quả chốt cuối cùng.
- BTC ưu tiên hậu kiểm kỹ các cặp đội cạnh tranh cao, đặc biệt nhóm
  tranh giải.
- BTC có quyền **chấm lại và lấy điểm trung vị** của các lần chạy trên
  Docker image đã chốt ⚠️ — chiến lược "farm noise" (nộp lặp để lấy max
  của các run nhiễu) bị vô hiệu hoá với nhóm tranh giải.

Khiếu nại: BTC gửi email kết quả dự kiến trước khi chốt bảng xếp hạng;
khiếu nại gửi trong tối đa **24 giờ** kể từ khi nhận email / công bố
kết quả.
