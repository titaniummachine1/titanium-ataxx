"""Teacher bake-off: which label source best predicts the truth on the same positions?

Sample = KataGo-sourced rows (teacher_src==1) of final_val_v2.npz that have restored Kata labels
(data/nnue/final/kata_labels_val_nopol.npz, from restore_kata_labels.py --no-policy --out-suffix _nopol).

Per position we compute fresh labels with Moonbird (1k/5k/20k nodes) and Titanium S57 (5k nodes) and run
neutral playouts (fixed strong engine vs itself, node jitter) as an independent judge.  Judges:
  J1  real KataGo self-play result of that game (outcome_p)      -- favours Kata TD labels (same game, hindsight)
  J2  mean of N Moonbird@2k self-play playouts from the position  -- engine-neutral-ish continuation
  J3  mean of N Titanium@2k self-play playouts                    -- second, different judge
Labels are mapped to win-probability by a logistic fit on half the sample and scored on the other half
(MSE / log-loss / corr) so scale differences between teachers do not decide the ranking.

Usage: python training/teacher_bakeoff.py --n 3000 --workers 6 --playouts 4
Out:   data/nnue/final/bakeoff_<tag>.npz  and a printed table.
"""
import argparse
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ataxx_move import RING, apply_move, board_str  # noqa: E402

ROOT = r"C:\gitProjects\titanium-ataxx"
MOON = ROOT + r"\scripts\moonbird\Moonbird-1.1.0-windows-amd64.exe"
TI_EXE = r"C:\gitProjects\main-ataxx\target\release\titanium-cli.exe"
TI_TUP = ROOT + r"\data\nnue\final_v2_wdl_ep8.tup"
FULL = (1 << 49) - 1
SCORE = re.compile(r"score\s+(cp|mate)\s+(-?\d+)")

REACH2 = []
for s in range(49):
    r, f = divmod(s, 7)
    m = 0
    for rr in range(max(0, r - 2), min(7, r + 3)):
        for ff in range(max(0, f - 2), min(7, f + 3)):
            if (rr, ff) != (r, f):
                m |= 1 << (rr * 7 + ff)
    REACH2.append(m)


def can_move(own, occ):
    empty = ~occ & FULL
    x = own
    while x:
        b = x & -x
        if REACH2[b.bit_length() - 1] & empty:
            return True
        x ^= b
    return False


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


class Uai:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, encoding="utf-8", errors="replace", cwd=ROOT)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.send("uai")
        while "uaiok" not in self.line(30):
            pass
        self.send("isready")
        while "readyok" not in self.line(30):
            pass

    def _rd(self):
        for l in iter(self.p.stdout.readline, ""):
            self.q.put(l)
        self.q.put(None)

    def line(self, t=60):
        l = self.q.get(timeout=t)
        if l is None:
            raise EOFError
        return l.strip()

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def go(self, fen, nodes):
        """-> (bestmove token, cp from side to move or None)"""
        self.send("position fen " + fen)
        self.send("go nodes %d" % nodes)
        cp = None
        while True:
            l = self.line()
            if l.startswith("info"):
                m = SCORE.search(l)
                if m:
                    v = int(m.group(2))
                    cp = max(-3000, min(3000, v)) if m.group(1) == "cp" else (3000 if v > 0 else -3000)
            elif l.startswith("bestmove"):
                return l.split()[1], cp

    def new(self):
        self.send("uainewgame")


W = {}


def init():
    W["moon"] = Uai([MOON])
    W["ti"] = Uai([TI_EXE, "serve", "--tuple", TI_TUP])
    W["rng"] = random.Random(os.getpid())


def playout(eng, occ0, occ1, stm, nodes, rng):
    """Strong self-play to the end under Botzone rules; returns 1/0 for the original side to move."""
    me = stm
    s = (occ0, occ1, 0, stm)
    eng.new()
    for _ in range(120):
        o0, o1, _, t = s
        own, oth = (o0, o1) if t == 0 else (o1, o0)
        occ = o0 | o1
        if occ == FULL or not can_move(own, occ) or not can_move(oth, occ):
            break
        n = max(50, int(nodes * rng.uniform(0.75, 1.25)))
        tok, _ = eng.go(to_fen(o0, o1, t), n)
        ns = apply_move(o0, o1, 0, t, tok)
        if ns is None:
            return None
        s = ns
    o0, o1, _, t = s
    c0, c1 = o0.bit_count(), o1.bit_count()
    occ = o0 | o1
    m0, m1 = can_move(o0, occ), can_move(o1, occ)
    if m0 and not m1:
        c0 = 49 - c1
    elif m1 and not m0:
        c1 = 49 - c0
    mine, theirs = (c0, c1) if me == 0 else (c1, c0)
    return 1.0 if mine > theirs else 0.0 if mine < theirs else 0.5


def work(job):
    i, occ0, occ1, stm, npl = job
    occ0, occ1 = int(occ0), int(occ1)
    fen = to_fen(occ0, occ1, stm)
    out = {"i": i}
    for n in (1000, 5000, 20000):
        W["moon"].new()
        out["moon%d" % n] = W["moon"].go(fen, n)[1]
    for name, eng in (("pm", W["moon"]), ("pt", W["ti"])):
        r = [playout(eng, occ0, occ1, stm, 2000, W["rng"]) for _ in range(npl)]
        r = [x for x in r if x is not None]
        out[name] = float(np.mean(r)) if r else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--playouts", type=int, default=4)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    z = np.load(ROOT + "/data/nnue/final/final_val_v2.npz")
    k = np.load(ROOT + "/data/nnue/final/kata_labels_val_nopol.npz")
    rows = k["row"][k["has"]]
    sel_k = np.nonzero(k["has"])[0]
    occ0, occ1 = z["occ0"][rows], z["occ1"][rows]
    empt = 49 - np.array([(int(x) | int(y)).bit_count() for x, y in zip(occ0, occ1)])
    rng = np.random.default_rng(a.seed)
    bands = [(1, 8), (9, 15), (16, 30), (31, 48)]
    pick = []
    for lo, hi in bands:
        idx = np.nonzero((empt >= lo) & (empt <= hi))[0]
        pick.append(rng.choice(idx, size=min(len(idx), a.n // len(bands)), replace=False))
    pick = np.concatenate(pick)
    print("sample", len(pick), flush=True)
    jobs = [(int(j), occ0[j], occ1[j], 0, a.playouts) for j in pick]

    t0 = time.time()
    res = []
    with Pool(a.workers, initializer=init) as pool:
        for n, r in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            res.append(r)
            if n % 200 == 0:
                print("%d/%d %.0fs" % (n, len(jobs), time.time() - t0), flush=True)
    res.sort(key=lambda r: r["i"])
    ii = np.array([r["i"] for r in res])
    # Titanium S57 static-search labels via `label` (stm-relative cp), same positions
    inp = "\n".join(board_str(int(occ0[j]), int(occ1[j]), 0, 0) for j in ii) + "\n"
    for flag in (["--tuple-native"], []):
        pr = subprocess.run([TI_EXE, "label", "--nodes", "5000", "--tuple", TI_TUP] + flag, input=inp, text=True,
                            capture_output=True)
        if pr.returncode == 0:
            break
    ti_cp = np.array([float(l.split()[0]) for l in pr.stdout.splitlines()], np.float64)
    assert len(ti_cp) == len(ii), (len(ti_cp), len(ii), pr.stderr[-300:])
    print("ti label flags", flag, "mean cp %.1f" % ti_cp.mean(), flush=True)
    for r, c in zip(res, ti_cp):
        r["ti5k"] = float(c)
    f = lambda key: np.array([np.nan if r[key] is None else r[key] for r in res], np.float64)
    td_w = k["kata_td_win"][sel_k][ii]
    td_d = k["kata_td_draw"][sel_k][ii]
    out = dict(
        row=rows[ii], empties=empt[ii],
        moon1k=f("moon1000"), moon5k=f("moon5000"), moon20k=f("moon20000"), ti5k=f("ti5k"),
        pm=f("pm"), pt=f("pt"),
        kata_td1=td_w[:, 0] + 0.5 * td_d[:, 0], kata_td4=(td_w[:, 3] + 0.5 * td_d[:, 3]),
        kata_td_mean=(td_w + 0.5 * td_d).mean(1),
        outcome_p=z["outcome_p"][rows[ii]], ataxx_p=z["ataxx_p_win"][rows[ii]] + 0.5 * z["ataxx_p_draw"][rows[ii]],
        teacher_p=z["teacher_p"][rows[ii]], old_ti_cp=z["titanium_cp"][rows[ii]],
        kata_visits=k["kata_visits"][sel_k][ii], n_out=z["n_outcomes"][rows[ii]],
    )
    path = ROOT + "/data/nnue/final/bakeoff_%s.npz" % a.tag
    np.savez(path, **out)
    print("saved", path, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
