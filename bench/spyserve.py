"""In-process sampling profiler wrapper (container has no CAP_SYS_PTRACE => no py-spy).

usage: python3 spyserve.py <module> [args...]
  SPY_OUT   where to dump (default /workspace/spy_dump.txt)
  SPY_HZ    sample rate (default 100)
Send SIGUSR1 to dump-and-reset; SIGUSR2 to reset counters only.
"""
import collections
import os
import runpy
import signal
import sys
import threading
import time

OUT = os.environ.get("SPY_OUT", "/workspace/spy_dump.txt")
HZ = float(os.environ.get("SPY_HZ", "100"))

leaf = collections.Counter()
stacks = collections.Counter()
nsample = [0]
lock = threading.Lock()


def sampler():
    period = 1.0 / HZ
    while True:
        frames = sys._current_frames()
        me = threading.get_ident()
        with lock:
            for tid, fr in frames.items():
                if tid == me:
                    continue
                st = []
                f = fr
                while f is not None:
                    st.append("%s:%s" % (os.path.basename(f.f_code.co_filename),
                                         f.f_code.co_name))
                    f = f.f_back
                if not st:
                    continue
                leaf[st[0]] += 1
                stacks[";".join(reversed(st[-12:]))] += 1
            nsample[0] += 1
        time.sleep(period)


def dump(signum=None, frame=None):
    with lock:
        n = nsample[0]
        lv = leaf.most_common(30)
        sv = stacks.most_common(15)
        leaf.clear()
        stacks.clear()
        nsample[0] = 0
    with open(OUT, "w") as fh:
        fh.write("samples=%d hz=%s\n\n== TOP LEAF FRAMES ==\n" % (n, HZ))
        for k, v in lv:
            fh.write("%7.2f%%  %6d  %s\n" % (100.0 * v / max(n, 1), v, k))
        fh.write("\n== TOP STACKS ==\n")
        for k, v in sv:
            fh.write("%7.2f%%  %6d  %s\n" % (100.0 * v / max(n, 1), v, k))


def reset(signum=None, frame=None):
    with lock:
        leaf.clear()
        stacks.clear()
        nsample[0] = 0


signal.signal(signal.SIGUSR1, dump)
signal.signal(signal.SIGUSR2, reset)
threading.Thread(target=sampler, daemon=True).start()

mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
