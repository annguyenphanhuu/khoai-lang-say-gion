"""Đo ACCEPTANCE của speculative decoding trên workload MÔ PHỎNG ĐÚNG đề bài.

Acceptance là tính chất của (model × dữ liệu × thuật toán draft) — KHÔNG phụ thuộc GPU
⇒ đo trên RTX 3050 là hợp lệ và chuyển thẳng sang H200.

Mô phỏng theo `grading-workload-spec.json`:
  - shared_system_prefix_tokens = 1000  (GIỐNG NHAU mọi hội thoại)
  - per_conversation_prefix_tokens = 1000 (riêng từng hội thoại)
  - user_turns_per_conversation = 6, new_user_tokens_per_turn = 150
  - output_tokens_per_turn_pinned = 300, context append-only

Cách đọc kết quả (2 nguồn, ưu tiên /metrics):
  1. /metrics (AUTHORITATIVE — NEXT-80 §2): server Prometheus phát
       vllm:spec_decode_num_draft_tokens_total     (số token DRAFT đề xuất)
       vllm:spec_decode_num_accepted_tokens_total  (số token draft ĐƯỢC CHẤP NHẬN)
       vllm:spec_decode_num_drafts_total           (số STEP draft = số target-forward có spec)
     ⇒ acceptance_rate = accepted / draft_tokens          (draft nào cũng bị verify)
     ⇒ mean_accept_len = (accepted + drafts) / drafts     (mỗi step +1 bonus token luôn nhận)
        = SỐ TOKEN / TARGET-FORWARD = HỆ SỐ CHIA tbt. Đây là con số quyết định điểm.
        effective_tbt = forward_time / mean_accept_len.  mean_accept_len 2 ⇒ tbt 3→1.5ms.
  2. SSE chunk (chỉ tham khảo): tokens/chunk. KHÔNG tin cậy — vLLM vẫn stream per-token
     kể cả khi spec nhận k token/step ⇒ dùng /metrics làm chuẩn, chunk chỉ để so nhanh.

Script snapshot /metrics TRƯỚC/SAU mỗi turn ⇒ tách được turn COPY-NẶNG (restate/checklist,
kỳ vọng accept cao) vs turn GENERATIVE. Chạy ĐA-LUỒNG làm nhiễu delta per-turn ⇒ để đo
per-turn chính xác, chạy `python3 spec_accept_probe.py 1` (1 hội thoại, tuần tự).
"""

import json
import os
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:8000")
URL = BASE + "/v1/chat/completions"
METRICS_URL = BASE + "/metrics"
MODEL = os.environ.get("MODEL", "LFM2.5-1.2B-Instruct")

# Turn nào là COPY-NẶNG (output copy nhiều từ context ⇒ prompt-lookup dễ hit).
COPY_HEAVY_TURNS = {4}  # QUESTIONS[4] = "Restate the whole reference document as a checklist"

_SPEC_KEYS = (
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_drafts_total",
)


def spec_metrics() -> dict:
    """Đọc /metrics, trả tổng tích luỹ các counter spec-decode (rỗng nếu spec OFF)."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as r:
            txt = r.read().decode()
    except Exception:
        return {}
    out: dict = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in _SPEC_KEYS:
            if line.startswith(k):
                try:
                    # Có thể có label {...}; giá trị luôn là token cuối.
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return out


def accept_stats(m0: dict, m1: dict):
    """(draft, accepted, drafts, accept_rate, mean_accept_len) từ 2 snapshot. None nếu thiếu."""
    d = m1.get(_SPEC_KEYS[0], 0) - m0.get(_SPEC_KEYS[0], 0)      # draft tokens
    a = m1.get(_SPEC_KEYS[1], 0) - m0.get(_SPEC_KEYS[1], 0)      # accepted tokens
    n = m1.get(_SPEC_KEYS[2], 0) - m0.get(_SPEC_KEYS[2], 0)      # drafts (steps)
    if d <= 0 and n <= 0:
        return None
    rate = (a / d) if d > 0 else float("nan")
    mlen = ((a + n) / n) if n > 0 else float("nan")
    return d, a, n, rate, mlen

# ~1000 token: prefix hệ thống DÙNG CHUNG cho mọi hội thoại (như đề bài).
SHARED_SYSTEM = (
    "You are a meticulous technical assistant for a large distributed database platform. "
    "Answer using only the reference material supplied in the conversation. "
    "Be precise, cite section numbers, and prefer concrete detail over generalities. "
) * 18

# ~1000 token: tài liệu riêng của từng hội thoại.
def conv_prefix(cid: int) -> str:
    return (
        f"REFERENCE DOCUMENT {cid}. Section 1: The storage engine uses a log-structured "
        f"merge tree with {cid + 3} levels and a compaction trigger at 75 percent fill. "
        f"Section 2: Replication is quorum based with replication factor {cid % 5 + 3}. "
        f"Section 3: The write path appends to a commit log, then to a memtable, then "
        f"flushes to an immutable sorted string table. Section 4: Read amplification is "
        f"bounded by the number of levels. Section 5: The compaction scheduler prioritises "
        f"levels by overlap ratio. Section 6: Backpressure is applied when the memtable "
        f"queue exceeds {cid + 8} entries. "
    ) * 9

QUESTIONS = [
    "Summarise section 1 and section 2 in detail, explaining every parameter you cite.",
    "Now explain how section 3 interacts with section 1. Cover the write path step by step.",
    "Compare section 4 and section 5. Which one bounds tail latency, and exactly why?",
    "Walk through what happens under section 6 backpressure, referencing sections 1 to 5.",
    "Restate the whole reference document as a numbered operational checklist.",
    "Given all of the above, describe the failure modes section by section.",
]
PAD = " Please answer thoroughly and restate the relevant section text as you go." * 6


def run_conversation(cid: int, out: list) -> None:
    messages = [
        {"role": "system", "content": SHARED_SYSTEM},
        {"role": "user", "content": conv_prefix(cid) + "\n\n" + QUESTIONS[0] + PAD},
    ]
    for turn in range(6):
        body = json.dumps(
            {
                "model": MODEL,
                "stream": True,
                "temperature": 0,
                "max_tokens": 300,
                "messages": messages,
            }
        ).encode()
        req = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"}
        )
        m0 = spec_metrics()
        t0 = time.time()
        chunks = 0
        text = []
        ttft = None
        with urllib.request.urlopen(req) as r:
            for line in r:
                if line.startswith(b"data: ") and b"[DONE]" not in line:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    try:
                        d = json.loads(line[6:])
                        c = d["choices"][0]["delta"].get("content")
                        if c:
                            text.append(c)
                    except Exception:
                        pass
        elapsed = time.time() - t0
        m1 = spec_metrics()
        answer = "".join(text)
        out.append((cid, turn, chunks, len(answer), ttft, elapsed, accept_stats(m0, m1)))
        messages.append({"role": "assistant", "content": answer})
        if turn + 1 < 6:
            messages.append({"role": "user", "content": QUESTIONS[turn + 1] + PAD})


if __name__ == "__main__":
    nconv = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out: list = []
    th = [threading.Thread(target=run_conversation, args=(c, out)) for c in range(nconv)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    el = time.time() - t0
    has_metrics = any(r[6] for r in out)
    print(f"\n=== {nconv} hội thoại x 6 turn, {el:.1f}s ===")
    if nconv > 1 and has_metrics:
        print("⚠️  đa-luồng ⇒ delta /metrics per-turn bị TRỘN giữa các hội thoại "
              "(chỉ TỔNG dưới cùng là chuẩn). Muốn per-turn sạch: chạy với nconv=1.")
    hdr = f"{'turn':>4} {'chunks':>7} {'chars':>7} {'ttft_ms':>8} {'sec':>6}"
    if has_metrics:
        hdr += f" {'accept%':>8} {'acc_len':>8} {'kind':>10}"
    print(hdr)
    for turn in sorted({r[1] for r in out}):
        rows = [r for r in out if r[1] == turn]
        ch = sum(r[2] for r in rows) / len(rows)
        se = sum(r[5] for r in rows) / len(rows)
        tt = sum(r[4] or 0 for r in rows) / len(rows)
        line = (f"{turn:>4} {ch:>7.1f} {sum(r[3] for r in rows) / len(rows):>7.0f} "
                f"{tt * 1000:>8.0f} {se:>6.2f}")
        if has_metrics:
            st = [r[6] for r in rows if r[6]]
            if st:
                rate = sum(s[3] for s in st) / len(st) * 100
                mlen = sum(s[4] for s in st) / len(st)
                kind = "COPY" if turn in COPY_HEAVY_TURNS else "gen"
                line += f" {rate:>8.1f} {mlen:>8.2f} {kind:>10}"
            else:
                line += f" {'-':>8} {'-':>8} {'-':>10}"
        print(line)

    tot_chunks = sum(r[2] for r in out)
    print(f"\nTONG chunks SSE = {tot_chunks} tren {len(out)} request "
          f"(max_tokens=300 moi request)")
    print(f"==> TOKEN/CHUNK trung binh ~ {300 * len(out) / max(tot_chunks, 1):.2f} "
          f"(=1.00 khi KHONG spec; >1 = do dai chap nhan trung binh)")

    # TỔNG /metrics — con số CHUẨN (không bị nhiễu đa-luồng vì cộng dồn toàn cục).
    if has_metrics:
        tot = spec_metrics()
        d = tot.get(_SPEC_KEYS[0], 0)
        a = tot.get(_SPEC_KEYS[1], 0)
        n = tot.get(_SPEC_KEYS[2], 0)
        print("\n=== /metrics TÍCH LUỸ (chuẩn, NEXT-80 gate acceptance) ===")
        print(f"  draft_tokens={d:.0f}  accepted={a:.0f}  drafts(steps)={n:.0f}")
        if d > 0:
            print(f"  ACCEPTANCE_RATE = accepted/draft = {a / d * 100:.1f}%")
        if n > 0:
            mlen = (a + n) / n
            print(f"  MEAN_ACCEPT_LEN = (accepted+drafts)/drafts = {mlen:.2f} tok/target-forward")
            print(f"  ==> effective_tbt ~ base_tbt / {mlen:.2f}  "
                  f"(vd base 3ms ⇒ {3.0 / mlen:.2f}ms; cần ≥1.5 để có điểm)")
    else:
        print("\n⚠️  /metrics KHÔNG có counter spec-decode ⇒ spec đang TẮT, hoặc "
              "tên metric khác ở version này. Kiểm tra: curl -s $BASE/metrics | grep spec_decode")
