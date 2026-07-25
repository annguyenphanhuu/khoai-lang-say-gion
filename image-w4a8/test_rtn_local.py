"""Đo sai số int4-g128 trên WEIGHT THẬT của LFM2-1.2B (CPU, không cần Hopper).

Mục đích: DỰ BÁO Accuracy Gate TRƯỚC KHI tốn submit portal. Nếu rel-error nhỏ + cosine ~1.0
trên các linear lớn (đặc biệt lm_head, down_proj) ⇒ W4A8 nhiều khả năng qua gate. Nếu lệch lớn
ở layer nào ⇒ để layer đó FP8 (skip int4).

Chạy trong image vllm (có torch+safetensors), mount model:
  docker run --rm -v /d/models/LFM2.5-1.2B-Instruct:/model:ro -v $PWD/image-w4a8:/w:ro \
    --entrypoint python3 <img> /w/test_rtn_local.py
"""
import sys, glob, os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from online_w4a8_rtn import quant_error, GROUP_SIZE

from safetensors import safe_open

MODEL = "/model"
files = glob.glob(f"{MODEL}/*.safetensors")
assert files, "no safetensors"

# chỉ soi 2D linear weight có in%128==0; bỏ embedding/norm/conv-1d-state
rows = []
worst = []
for f in files:
    with safe_open(f, framework="pt") as st:
        for k in st.keys():
            if not (k.endswith(".weight")):
                continue
            t = st.get_tensor(k)
            if t.dim() != 2:
                continue
            out_f, in_f = t.shape
            if in_f % GROUP_SIZE != 0 or in_f < GROUP_SIZE:
                rows.append((k, tuple(t.shape), "SKIP in%128!=0", None))
                continue
            rel, cos = quant_error(t)
            tag = "OK" if (rel < 0.05 and cos > 0.999) else ("WATCH" if rel < 0.10 else "RISK")
            rows.append((k, tuple(t.shape), f"rel={rel:.4f} cos={cos:.5f}", tag))
            if tag != "OK":
                worst.append((k, rel, cos))

for k, shp, msg, tag in rows:
    print(f"[{tag or '--':5}] {str(shp):18} {msg:34} {k}")

print("\n=== TỔNG KẾT ===")
graded = [r for r in rows if r[3] in ("OK", "WATCH", "RISK")]
n_ok = sum(1 for r in graded if r[3] == "OK")
print(f"{n_ok}/{len(graded)} linear đạt OK (rel<5% & cos>0.999)")
if worst:
    print("Layer cần chú ý (có thể để FP8 thay vì int4):")
    for k, rel, cos in sorted(worst, key=lambda x: -x[1])[:12]:
        print(f"   rel={rel:.4f} cos={cos:.5f}  {k}")
else:
    print("Tất cả linear int4-g128 đều OK ⇒ accuracy risk THẤP.")
