"""Auto sampling profiler for EVERY python process (incl. vLLM EngineCore child).

Enable with SPY_ALL=1. Dumps every SPY_EVERY seconds to $SPY_DIR/spy_<name>_<pid>.txt
with counters reset after each dump, so the last file covers the last window.
"""
import os

if os.environ.get("SPY_ALL") == "1":
    import collections
    import sys
    import threading
    import time

    _DIR = os.environ.get("SPY_DIR", "/workspace/spyauto")
    _HZ = float(os.environ.get("SPY_HZ", "100"))
    _EVERY = float(os.environ.get("SPY_EVERY", "5"))

    def _run():
        os.makedirs(_DIR, exist_ok=True)
        me = threading.get_ident()
        leaf = collections.Counter()
        stacks = collections.Counter()
        n = 0
        t_last = time.time()
        period = 1.0 / _HZ
        while True:
            for tid, fr in sys._current_frames().items():
                if tid == me:
                    continue
                st = []
                f = fr
                while f is not None:
                    st.append("%s:%s" % (os.path.basename(f.f_code.co_filename),
                                         f.f_code.co_name))
                    f = f.f_back
                if st:
                    leaf[st[0]] += 1
                    stacks[";".join(reversed(st[-14:]))] += 1
            n += 1
            if time.time() - t_last >= _EVERY:
                name = os.path.basename(sys.argv[0] or "py").replace("/", "_")[:24]
                p = "%s/spy_%s_%d.txt" % (_DIR, name, os.getpid())
                try:
                    with open(p, "w") as fh:
                        fh.write("pid=%d argv=%s samples=%d\n\n== LEAF ==\n"
                                 % (os.getpid(), " ".join(sys.argv[:3]), n))
                        for k, v in leaf.most_common(25):
                            fh.write("%7.2f%%  %6d  %s\n" % (100.0 * v / max(n, 1), v, k))
                        fh.write("\n== STACKS ==\n")
                        for k, v in stacks.most_common(12):
                            fh.write("%7.2f%%  %6d  %s\n" % (100.0 * v / max(n, 1), v, k))
                except Exception:
                    pass
                leaf.clear()
                stacks.clear()
                n = 0
                t_last = time.time()
            time.sleep(period)

    threading.Thread(target=_run, daemon=True).start()
