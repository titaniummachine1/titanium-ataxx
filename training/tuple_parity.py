"""Python trainer tuple forward vs the ENGINE's tuple forward, family by family. READ-ONLY diagnosis.

For each table family of a TUP7 file, writes a copy with every OTHER family zeroed (same layout as train_stream.dump_tup)
and compares the engine (nnue-ataxx `nneval` with a TNN2 net that outputs exactly the S57 term) with the Python forward
(train_stream.features_for + materialize32 + base_eval_stm + nnue_baseline.weights) on 400 val positions.
Usage: python training/tuple_parity.py [--tup data/nnue/final_v2_wdl_ep8.tup]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnue_baseline as NB  # noqa: E402
import train_stream as S  # noqa: E402
from ataxx_move import board_str  # noqa: E402

T = S.T
ROOT = "C:/gitProjects/titanium-ataxx"
EXE = "C:/gitProjects/nnue-ataxx/target/release/titanium-cli.exe"
S57ONLY = ROOT + "/data/nnue/parity_s57only.tnn"
# TUP7 layout (bytes after the 4-byte magic): name, offset, length, element size
SEGS = [("bias", 0, 4, 4), ("s", 4, 2916, 1), ("counts", 2920, 16, 4), ("dirs", 2936, 4 * 6561, 1),
        ("q2", 29180, 9216 * 2, 2), ("envus", 29180 + 18432, 35721 * 2, 2), ("envthem", 29180 + 18432 + 71442, 35721 * 2, 2),
        ("mobus", 29180 + 18432 + 142884, 2500 * 2, 2), ("mobthem", 29180 + 18432 + 142884 + 5000, 2500 * 2, 2),
        ("capn", 29180 + 18432 + 142884 + 10000, 18 * 2, 2), ("conv", 29180 + 18432 + 142884 + 10036, 512 * 2, 2),
        ("pa", 29180 + 18432 + 142884 + 11060, 80 * 6561, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tup", default=ROOT + "/data/nnue/final_v2_wdl_ep8.tup")
    a = ap.parse_args()
    raw = open(a.tup, "rb").read()
    assert raw[:4] == b"TUP7" and len(raw) == 726440
    body = raw[4:]
    assert SEGS[-1][1] + SEGS[-1][2] == len(body), (SEGS[-1][1] + SEGS[-1][2], len(body))
    z = np.load(ROOT + "/data/nnue/final/final_val_v3.npz")
    o0, o1, st, blk = (z[k] for k in ("occ0", "occ1", "stm", "blk"))
    rows = np.random.default_rng(2).choice(len(st), 400, replace=False)
    a0, a1, ab, as_ = o0[rows].astype(np.uint64), o1[rows].astype(np.uint64), blk[rows].astype(np.uint64), st[rows]
    base = T.base_eval_stm(a0, a1, ab, as_)
    f = S.features_for(a0, a1, ab, as_)
    SP, DE = S.materialize32(*f, [0, 1, 2, 3])
    for fam in [s[0] for s in SEGS if s[0] != "counts"] + ["ALL"]:
        b = bytearray(body)
        for name, off, ln, _ in SEGS:
            if name in ("counts",) or name == fam or fam == "ALL":
                continue
            b[off:off + ln] = bytes(ln)
        path = ROOT + "/data/nnue/par_only_%s.tup" % fam
        open(path, "wb").write(b"TUP7" + bytes(b))
        bias, W, D = NB.weights(path)
        v = base + W[SP].sum(1) + D[DE].sum(1) + bias
        ref = ROOT + "/data/nnue/par_only_%s.txt" % fam
        with open(ref, "w") as fh:
            for r, x in zip(rows, v):
                fh.write("%s %d\n" % (board_str(int(o0[r]), int(o1[r]), 0, int(st[r])), int(round(x))))
        out = subprocess.run([EXE, "nneval", "--nnue", S57ONLY, "--boards", ref, "--tuple", path], capture_output=True, text=True)
        print("%-8s python |tuple| %7.2f  -> %s" % (fam, float(np.abs(v - base).mean()), (out.stdout + out.stderr).strip()), flush=True)


if __name__ == "__main__":
    main()
