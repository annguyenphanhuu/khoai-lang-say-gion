# Loadgen 1 stream duy nhat (C=1) - dung dung diem van hanh cua portal.
import json, sys, time, urllib.request
URL = "http://localhost:8000/v1/chat/completions"
DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 45
tok = 0; gaps = []
t_end = time.time() + DUR
i = 0
while time.time() < t_end:
    i += 1
    body = json.dumps({
        "model": "LFM2.5-1.2B-Instruct", "stream": True, "temperature": 0,
        "max_tokens": 200, "seed": i,
        "messages": [{"role": "system", "content": "You are a helpful assistant. " * 40},
                     {"role": "user", "content": f"Request {i}: explain paged attention in detail."}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    st = []
    try:
        with urllib.request.urlopen(req) as r:
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    st.append(time.time())
    except Exception as e:
        print("ERR", e); break
    tok += len(st)
    gaps += [(b - a) * 1000 for a, b in zip(st, st[1:])]
gaps.sort()
print(f"C=1 tokens={tok} tbt_med={gaps[len(gaps)//2]:.2f}ms" if gaps else "no data")
