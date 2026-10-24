"""Moonbird relabel of the own-game rows (teacher_kind == 2) of dataset v3, plus a KataGo calibration sample.

Why: own-game rows only carry Titanium 5k labels (the weakest teacher, corr .20 with the result). Moonbird is
+485 Elo at equal nodes (LEDGER WextMoon) and Moonbird@100 nodes ~ Titanium@50k-100k (S33spot), so even a
1k-node Moonbird search is a far stronger label. Measured speed on real rows: 3.8 ms/pos @1k nodes (1 process).

Jobs:   --part calib : stratified sample of KataGo rows (kind 1) from train, per empties band -> fit Moonbird cp scale
        --part train / val / test : all kind-2 rows of that split
Output: data/nnue/final/moon{nodes}/{part}_c{NNNN}.npz  with  row (int64), cp (float32, NaN = no score), depth, nodes
        chunks are atomic (tmp+replace) and skipped if present -> resumable; STOP file C:/gitProjects/logs/STOP halts
        between chunks. Count-verified: every chunk has exactly its row count.
Usage:  python training/relabel_moonbird.py --part calib --nodes 1000 --workers 8
"""
import argparse
import os
import queue
import re
import subprocess
import threading
import time
from multiprocessing import Pool

import numpy as np

ROOT = r"C:/gitProjects/titanium-ataxx"
FINAL = ROOT + "/data/nnue/final"
MOON = ROOT + "/scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe"
STOP = r"C:/gitProjects/logs/STOP"
SCORE = re.compile(r"depth\s+(\d+).*?nodes\s+(\d+).*?score\s+(cp|mate)\s+(-?\d+)")
BANDS = [(0, 8), (9, 15), (16, 24), (25, 35), (36, 49)]


def to_fen(occ0, occ1, stm):
    rows = []
    for r in range(7):
        row, gap = "", 0
        for f in range(7):
            bit = 1 << (r * 7 + f)
            c = "x" if occ0 & bit else "o" if occ1 & bit else "."
            if c == ".":
                gap += 1
            else:
                if gap:
                    row += str(gap)
                    gap = 0
                row += c
        if gap:
            row += str(gap)
        rows.append(row)
    return "/".join(rows) + (" x 0 1" if stm == 0 else " o 0 1")


class Moon:
    def __init__(self):
        self.p = subprocess.Popen([MOON], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, encoding="utf-8", errors="replace")
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.send("uai")
        while "uaiok" not in self.line(30):
            pass
        self.send("setoption name Hash value 16")
        self.send("isready")
        while "readyok" not in self.line(30):
            pass

    def _rd(self):
        for l in iter(self.p.stdout.readline, ""):
            self.q.put(l)
        self.q.put(None)

    def line(self, t):
        l = self.q.get(timeout=t)
        if l is None:
            raise EOFError("moonbird exited")
        return l

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def label(self, fen, nodes):
        self.send("position fen " + fen)
        self.send("go nodes %d" % nodes)
        cp, dep, nn = np.nan, 0, 0
        while True:
            l = self.line(30)
            if l.startswith("info"):
                m = SCORE.search(l)
                if m:
                    dep, nn = int(m.group(1)), int(m.group(2))
                    v = int(m.group(4))
                    cp = float(max(-3000, min(3000, v))) if m.group(3) == "cp" else (3000.0 if v > 0 else -3000.0)
            elif l.startswith("bestmove"):
                return cp, dep, nn

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def run_chunk(job):
    path, rows, occ0, occ1, stm, nodes = job
    if os.path.exists(path):
        return path, len(rows), 0.0, "skip"
    t0 = time.time()
    m = Moon()
    m.send("uainewgame")
    cp = np.full(len(rows), np.nan, np.float32)
    dep = np.zeros(len(rows), np.int16)
    nn = np.zeros(len(rows), np.int32)
    for i in range(len(rows)):
        fen = to_fen(int(occ0[i]), int(occ1[i]), int(stm[i]))
        try:
            cp[i], dep[i], nn[i] = m.label(fen, nodes)
        except Exception:  # dead/hung engine: restart, leave NaN (falls back to the Titanium label)
            m.p.kill()
            m = Moon()
    m.close()
    tmp = path + ".tmp.npz"
    np.savez(tmp, row=rows.astype(np.int64), cp=cp, depth=dep, nodes=nn)
    os.replace(tmp, path)
    return path, len(rows), time.time() - t0, "nan=%d" % int(np.isnan(cp).sum())


def empties(o0, o1):
    occ = o0 | o1
    x = occ.astype(np.uint64)
    c = np.zeros(len(x), np.int32)
    for b in range(49):
        c += ((x >> np.uint64(b)) & np.uint64(1)).astype(np.int32)
    return 49 - c


def low_priority():
    """IDLE priority class: the PC stays usable; Pool workers and their Moonbird children inherit it."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:
        pass


def main():
    low_priority()
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["calib", "train", "val", "test"])
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--calib-per-band", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N rows")
    a = ap.parse_args()
    out = "%s/moon%d" % (FINAL, a.nodes)
    os.makedirs(out, exist_ok=True)
    split = "train" if a.part == "calib" else a.part
    z = np.load("%s/final_%s_v3.npz" % (FINAL, split))
    kind = z["teacher_kind"]
    o0, o1, stm = z["occ0"], z["occ1"], z["stm"]
    if a.part == "calib":
        rng = np.random.default_rng(12345)
        k1 = np.nonzero(kind == 1)[0]
        e = empties(o0[k1], o1[k1])
        rows = np.sort(np.concatenate([rng.choice(k1[(e >= lo) & (e <= hi)],
                                                  size=min(a.calib_per_band, int(((e >= lo) & (e <= hi)).sum())),
                                                  replace=False) for lo, hi in BANDS]))
    else:
        rows = np.nonzero(kind == 2)[0]
    if a.limit:
        rows = rows[:a.limit]
    tag = a.part + ("_smoke" if a.limit else "")
    jobs = []
    for ci, lo in enumerate(range(0, len(rows), a.chunk)):
        r = rows[lo:lo + a.chunk]
        jobs.append(("%s/%s_c%04d.npz" % (out, tag, ci), r, o0[r], o1[r], stm[r], a.nodes))
    print("part %s rows %d chunks %d nodes %d workers %d" % (tag, len(rows), len(jobs), a.nodes, a.workers), flush=True)
    t0 = time.time()
    done = 0
    with Pool(a.workers) as pool:
        it = pool.imap_unordered(run_chunk, jobs)
        for path, n, dt, msg in it:
            done += n
            el = time.time() - t0
            print("%s %d rows %.0fs %s | %d/%d rows, %.0fs elapsed, eta %.0fs" % (
                os.path.basename(path), n, dt, msg, done, len(rows), el,
                el / max(done, 1) * (len(rows) - done)), flush=True)
            if os.path.exists(STOP):
                print("STOP file present: terminating after current chunks", flush=True)
                pool.terminate()
                raise SystemExit(3)
    # verify: every chunk present with the exact row count and the rows match
    got = 0
    for path, r, *_ in jobs:
        f = np.load(path)
        assert len(f["row"]) == len(r) and (f["row"] == r).all(), "chunk mismatch " + path
        got += len(r)
    assert got == len(rows)
    print("VERIFIED part %s: %d rows in %d chunks, %.0fs" % (tag, got, len(jobs), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
