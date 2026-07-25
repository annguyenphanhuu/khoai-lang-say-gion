# RIG RunPod — chỉ dùng cho correctness, KHÔNG dùng để đo điểm

> ⚠️ **ĐỌC KB §5 TRƯỚC.** Rig H100 có băng thông 5.6× và SM 7× nhiều hơn MiG H200 của portal
> (`--gpu-memory-utilization` chỉ giới hạn dung lượng, không giới hạn BW/SM). Mọi kết luận về đòn
> **băng thông** (int4, KV-quant, kinh tế spec decode) đo ở đây đều **vô giá trị**.
> Rig chỉ đáng tin cho: **correctness temp=0**, **acceptance rate**, GPU-time *tương đối* cùng loại kernel.
> **Chốt điểm: chỉ portal** (không giới hạn submit online).

## Pod
- H100 SXM 80GB, template *RunPod PyTorch 2.8*, driver 580 / **CUDA 13.0** (image thi cần CUDA13), cap 9.0.
- **`/workspace` sống qua Stop/Start; container (pip install) CHẾT.** Cài lại vllm mỗi session.
- Model `/workspace/model` · bench `/workspace/bench` → symlink `/workspace/repo/bench`.
- 💸 ~$3/giờ → **Stop khi nghỉ**.

## Setup lại sau restart
```bash
PIP_CACHE_DIR=/workspace/.pipcache pip install "vllm==0.25.1"   # cache ~6.2G đã có, ~1-2 phút
```
- ⚠️ **KHÔNG tạo venv trên `/workspace`** (FUSE network FS ⇒ giải nén >15 phút). Cài vào system python.
- Sau restart CLI `vllm` không có trên PATH ⇒ dùng `python3 -m vllm.entrypoints.openai.api_server`.
- `libnvrtc.so.13` thiếu ⇒ `deep_gemm` import fail (WARNING vô hại).

## SSH (RunPod proxy ép PTY)
```bash
printf 'cmd1\ncmd2\nexit\n' | timeout 120 ssh -tt -o StrictHostKeyChecking=accept-new <podid>-<hash>@ssh.runpod.io
```
- Endpoint **đổi mỗi lần tạo pod mới**. Không chạy được `ssh host 'cmd'` (proxy ép interactive shell).
- Lọc escape code: `| tr -d '\r' | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\x1b\][0-9;]*//g'`
- Copy file: base64. **Chunk ≤1200 ký tự/dòng** (>~1500 bị PTY cắt hỏng), `printf '%s' 'piece' >> /tmp/x.b64`
  từng dòng rồi `base64 -d /tmp/x.b64 > dst`.

## Serve + bench
```bash
taskset -c 0-2 OMP_NUM_THREADS=1 python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model --served-model-name LFM2.5-1.2B-Instruct --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.225 --enable-prefix-caching --quantization fp8
# ⚠️ KHÔNG --disable-log-stats khi bench: harness cần /metrics

BASE=http://localhost:8000 NCONV=70 TURNS=6 OUT=300 RATE=8 SEED=42 python3 bench/ers_harness.py
```
- **RATE=8** = điểm calibrate (batch decode mean 27). Harness diff histogram trước/sau ⇒ **không cần
  restart server giữa các config**.
- Chỉ tin số **server-side** (`ERS_HIST`): client-side TTFT bị thổi ~2.5× do 70 thread Python tranh GIL.
- TPOT lấy từ `_sum/_count` chứ không từ bucket (bucket đầu = 10ms quá thô).
- `bench/correctness_diff.py` = cổng bắt buộc, temp=0, vs golden fp8.
