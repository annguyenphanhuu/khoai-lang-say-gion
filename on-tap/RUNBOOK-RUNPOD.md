# RUNBOOK — Thực thi trên RunPod (H100/H200) để đẩy LFM2 vLLM lên 75+

> Đọc cùng `on-tap/HANDOFF-RUNPOD.md` (kế hoạch) và `on-tap/KNOWLEDGE_BASE.md` (SSOT).
> Cách dùng: chạy TỪNG BLOCK theo thứ tự, **dán output về chat** cho Claude đọc rồi quyết định bước kế.
> Nguyên tắc vàng: **1 biến/lần** (mọi lần đổi mù đều regress). **correctness_diff.py là cổng bắt buộc** trước khi nộp.

Best hiện tại: **65.06** (`compose-w4a8.yml`). Cần: **75+**. Đường vật lý còn lại: **hạ tbt (fuse kernel) + cắt ttft (scheduling)**.

---

## PHASE 0 — Dựng pod + verify driver (làm ngay sau khi nạp credit)

### 0.1 Deploy pod
- RunPod → **Deploy → Pods** (KHÔNG phải Serverless).
- GPU: **1× H100 80GB PCIe/SXM** (H200 cũng được). Rẻ nhất đủ dùng.
- Template: **"RunPod PyTorch 2.x"** hoặc bất kỳ template CUDA mới. Container disk ≥ 40GB, Volume ≥ 20GB.
- Bật **SSH** (thêm public key của bạn: nội dung file `C:\Users\anngu\.ssh\id_rsa.pub`).
- Expose TCP port **8000** (để test HTTP nếu cần từ ngoài — không bắt buộc, ta test trong pod).

### 0.2 SSH vào + VERIFY DRIVER (quan trọng — image thi cần CUDA 13)
```bash
nvidia-smi
```
→ Dòng trên cùng phải là **"CUDA Version: 13.x"**. Nếu chỉ 12.x ⇒ **đổi template/pod khác** (driver mới hơn), đừng tiếp tục.

```bash
# xác nhận Hopper (cap 9.0) — kernel W4A8/FP8 cần cap90
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
docker --version && docker info | grep -i runtime   # cần nvidia runtime; nếu template không có docker, xem 0.4
```

**→ DÁN VỀ:** output của cả 3 lệnh trên.

### 0.3 Lấy code (bench scripts) lên pod
Cách A — clone GitHub (nếu bench đã push):
```bash
cd /workspace && git clone https://github.com/annguyenphanhuu/khoai-lang-say-gion.git repo && ls repo/bench
```
Cách B — upload từ máy Windows (nếu chưa push, chạy TRÊN MÁY BẠN, PowerShell):
```bash
scp -P <SSH_PORT> -r D:\khoai-lang-say-gion\bench root@<POD_IP>:/workspace/bench
```
(RunPod cho sẵn dòng `scp`/`ssh` trong tab Connect — copy port & IP từ đó.)

### 0.4 Upload model (2.2GB) lên pod
Chạy TRÊN MÁY BẠN (PowerShell):
```bash
scp -P <SSH_PORT> -r "D:\models\LFM2.5-1.2B-Instruct" root@<POD_IP>:/workspace/model
```
→ Trên pod: `ls /workspace/model` phải thấy `config.json`, `model.safetensors`, `tokenizer.json`...

### 0.5 Nếu template KHÔNG có docker (nhiều template RunPod chạy sẵn trong container)
Thì chạy vLLM **trực tiếp** thay vì `docker run`. Pull image bằng cách cài vllm trùng version, HOẶC
dùng pod template "Docker-in-Docker". **→ Dán về:** kết quả `docker info` ở 0.2 để Claude chọn nhánh
(docker-run vs chạy trực tiếp). Các block dưới viết theo docker; nếu chạy trực tiếp Claude sẽ đổi lệnh.

---

## PHASE 1 — Reproduce baseline 65 trên rig (calibrate rig ↔ portal)

Portal = MiG slice **18GB VRAM, 3 CPU core**. Ta mô phỏng bằng `--cpus 3` + `--gpu-memory-utilization`
sao cho VRAM thực ~18GB. Trên H100 80GB: `util ≈ 0.225`. (H200 141GB: `≈ 0.13`.)

### 1.1 Pull image best (W4A8)
```bash
docker pull annguyenphanhuu/vllm-w4a8:online-r1@sha256:afa562d714e185cc345a9143a762a59a8288c8147a10fe1b60e1fe3d191332b0
```

### 1.2 Chạy constrained + đo tbt (C=1) và workload-shape (ttft)
```bash
cd /workspace/repo 2>/dev/null || cd /workspace
IMG=annguyenphanhuu/vllm-w4a8:online-r1

docker run -d --name base --gpus all --cpus 3 --shm-size 2g \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e VLLM_LOGGING_LEVEL=WARNING \
  -v /workspace/model:/model:ro -v /workspace/bench:/bench:ro \
  $IMG \
  --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
  --max-model-len=32768 --gpu-memory-utilization=0.225 --tensor-parallel-size=1 \
  --enable-prefix-caching --quantization=online_w4a8 --disable-log-stats

# chờ healthy
for i in $(seq 1 60); do
  [ "$(docker exec base curl -s -o /dev/null -w '%{http_code}' localhost:8000/health 2>/dev/null)" = 200 ] && { echo HEALTHY; break; }
  docker ps -q -f name=base | grep -q . || { echo DIED; docker logs --tail 40 base; break; }
  sleep 5
done
```
```bash
# đo tbt steady-state (C=1) và ttft theo workload thi
docker exec base python3 /bench/loadgen_c1.py 30
docker exec base env BASE=http://localhost:8000 NCONV=3 OUT=300 python3 /bench/wl_probe.py
```

**→ DÁN VỀ:** dòng `C=1 ... tbt_med=...ms` + bảng `wl_probe` (ttft turn1/turn2+, tbt, hit%).
Kỳ vọng: tbt_med ~2.7–3.0ms, ttft turn2+ ~30–50ms. Đây là **mốc gốc** để so mọi thay đổi sau.

Giữ container `base` chạy để làm golden ở correctness gate (Phase 5). Hoặc `docker rm -f base` nếu cần VRAM.

---

## PHASE 2 — Đòn RẺ NHẤT trước: crash-safe fuse flags (thử ngay, 1 biến/lần)

Các flag này gộp op sẵn trong vLLM (không cần viết kernel). Crash-safe: lỗi lúc boot KHÔNG bị chấm.

### 2.1 fuse_norm_quant + fuse_act_quant
```bash
docker rm -f f1 2>/dev/null
docker run -d --name f1 --gpus all --cpus 3 --shm-size 2g \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e VLLM_LOGGING_LEVEL=WARNING \
  -v /workspace/model:/model:ro -v /workspace/bench:/bench:ro \
  annguyenphanhuu/vllm-w4a8:online-r1 \
  --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
  --max-model-len=32768 --gpu-memory-utilization=0.225 --tensor-parallel-size=1 \
  --enable-prefix-caching --quantization=online_w4a8 --disable-log-stats \
  --compilation-config='{"pass_config":{"enable_fusion":true,"enable_noop":true}}'

for i in $(seq 1 60); do
  [ "$(docker exec f1 curl -s -o /dev/null -w '%{http_code}' localhost:8000/health 2>/dev/null)" = 200 ] && { echo HEALTHY; break; }
  docker ps -q -f name=f1 | grep -q . || { echo DIED; docker logs --tail 40 f1; break; }
  sleep 5
done
docker exec f1 python3 /bench/loadgen_c1.py 30
```
**→ DÁN VỀ:** tbt_med. Nếu **thấp hơn** Phase 1 và boot OK ⇒ ứng viên tốt, sang correctness gate.
Nếu bằng/cao hơn ⇒ flag vô ích trên arch này, bỏ, sang Phase 3.

> Ghi chú: tên flag fusion pass có thể đổi theo version vLLM trong image (0.22.1). Nếu boot log báo
> unknown key ⇒ **dán log về**, Claude sẽ tra đúng tên pass từ source trong image.

---

## PHASE 3 — PROFILING: định vị chính xác 3ms decode (trước khi viết kernel)

### 3.1 GPU-forward breakdown (nsys)
```bash
chmod +x /workspace/bench/nsys_decode.sh
OUTDIR=/workspace/out MODEL_MNT=/workspace/model \
  bash /workspace/bench/nsys_decode.sh annguyenphanhuu/vllm-w4a8:online-r1 \
  --quantization=online_w4a8
```
**→ DÁN VỀ:** bảng "TOP CUDA KERNELS". Tải `/workspace/out/decode.nsys-rep` về mở Nsight nếu muốn xem timeline.
Đọc: (a) kernel nào chiếm % lớn nhất, (b) tổng thời gian gap giữa kernel (= launch/replay overhead còn lại).

### 3.2 CPU-leaf khi decode (conv có chạy eager không?)
```bash
bash /workspace/bench/verify_cudagraph.sh annguyenphanhuu/vllm-w4a8:online-r1 \
  --quantization=online_w4a8
```
**→ DÁN VỀ:** phần "(1) CUDA GRAPH MODE" + "(2) Top CPU leaf". Nếu `causal_conv1d`/`short_conv` ở top CPU ⇒
conv chạy eager ⇒ fuse ShortConv có lợi. Nếu không ⇒ đã trong graph ⇒ dồn vào kernel GEMV/attention.

> Kết quả 3.1+3.2 quyết định fuse Ở ĐÂU. Claude sẽ đọc và chỉ định kernel cụ thể (Phase 4).

---

## PHASE 4 — FUSE KERNEL (đòn chính tới 75) — CHỈ làm sau khi có profiling

Dựa trên Phase 3, Claude sẽ viết Triton/CUTLASS kernel fused cho phần nóng (ví dụ RMSNorm+FP8-quant,
SiLU-mul+FP8-quant, hoặc gộp ShortConv path). Quy trình mỗi kernel:
1. Claude viết kernel + patch vào image (thêm layer Dockerfile trong `image-w4a8/` hoặc image mới).
2. Bạn build trên pod: `docker build -f image-w4a8/Dockerfile -t local/vllm-fuse:r1 image-w4a8/`
3. Đo tbt (`loadgen_c1.py`) — **1 kernel/lần**, so với mốc Phase 1/2.
4. **Correctness gate (Phase 5) BẮT BUỘC pass** trước khi giữ.

---

## PHASE 5 — CORRECTNESS GATE (cổng bắt buộc trước mọi lần nộp)

temp=0 output của ứng viên phải **giống hệt** image best. Lệch 1 token ⇒ KHÔNG nộp.

```bash
# Pha 1: chụp golden từ image BEST (đang chạy container `base` ở :8000)
docker exec base env MODE=capture BASE=http://localhost:8000 OUT=/bench/golden.json \
  python3 /bench/correctness_diff.py
docker rm -f base   # tắt best, giải phóng VRAM

# Pha 2: bật ứng viên (image fuse) cùng port 8000, rồi diff
#   ... docker run --name cand ...  (giống Phase 1 nhưng image/flag ứng viên) ...
docker exec cand env MODE=diff BASE=http://localhost:8000 REF=/bench/golden.json \
  python3 /bench/correctness_diff.py
```
Exit 0 = PASS (an toàn nộp). Exit 1 = FAIL (đổi output, KHÔNG nộp).
**→ DÁN VỀ:** dòng cuối `PASS`/`FAIL`.

---

## PHASE 6 — TTFT (nửa còn lại: 45 → ~30ms)

Đo scheduling knob trên workload Poisson, 1 biến/lần:
```bash
# ví dụ thử max-num-batched-tokens, so ttft turn2+ với Phase 1
for MNB in 2048 4096 8192; do
  docker rm -f t1 2>/dev/null
  docker run -d --name t1 --gpus all --cpus 3 --shm-size 2g \
    -e OMP_NUM_THREADS=1 -e VLLM_LOGGING_LEVEL=WARNING \
    -v /workspace/model:/model:ro -v /workspace/bench:/bench:ro \
    annguyenphanhuu/vllm-w4a8:online-r1 \
    --model=/model --served-model-name=LFM2.5-1.2B-Instruct --host=0.0.0.0 --port=8000 \
    --max-model-len=32768 --gpu-memory-utilization=0.225 --enable-prefix-caching \
    --quantization=online_w4a8 --disable-log-stats --max-num-batched-tokens=$MNB
  for i in $(seq 1 40); do [ "$(docker exec t1 curl -s -o /dev/null -w '%{http_code}' localhost:8000/health 2>/dev/null)" = 200 ] && break; sleep 5; done
  echo "== MNB=$MNB =="; docker exec t1 env BASE=http://localhost:8000 NCONV=3 OUT=300 python3 /bench/wl_probe.py | tail -4
done
```
**→ DÁN VỀ:** ttft turn2+ cho từng MNB. Cũng có thể thử `--max-num-seqs`, chunked-prefill on/off.

---

## PHASE 7 — Chốt & nộp

- Cấu hình thắng (tbt↓ AND/OR ttft↓, correctness PASS, failed không tăng) → build image cuối, `docker push`,
  **pin digest @sha256** vào một `compose-*.yml` mới.
- Ghi kết quả vào `on-tap/KNOWLEDGE_BASE.md §4.3` (score, tbt, ttft, failed, acc_drop).
- Nhớ: startup-error KHÔNG bị chấm (thử aggressive thoải mái); cấu hình BOOT-được mà chạy tệ thì BỊ chấm.

---

## Bảng "dán về" nhanh (Claude cần gì ở mỗi phase)
| Phase | Dán về |
| :-- | :-- |
| 0 | `nvidia-smi` (CUDA 13?), compute_cap, `docker info` runtime |
| 1 | tbt_med (loadgen) + bảng wl_probe (ttft, hit%) |
| 2 | tbt_med với fuse flags + boot OK/DIED |
| 3 | TOP CUDA KERNELS + top CPU leaf + cudagraph mode |
| 5 | PASS/FAIL correctness |
| 6 | ttft turn2+ theo từng knob |

## Chi phí ước tính
H100 ~$2–3/giờ. Phase 0–3 (setup + baseline + profiling): ~1–2h ≈ $3–6. Fuse kernel iterate: tuỳ.
**Tip tiết kiệm:** `docker rm -f` container không dùng; **Stop pod** khi nghỉ (chỉ mất phí volume rẻ).
