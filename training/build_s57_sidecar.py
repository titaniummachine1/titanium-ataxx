"""DEPRECATED for the v2 skip term: the Python tuple forward differs from the ENGINE by ~66 cp mean (NET_ANALYSIS
section 27). Use `titanium-cli s57dump` (engine-exact) instead. Kept for the parity investigation only.

Per-row S57 static eval sidecar (Python semantics). Writes data/nnue/s57/{split}.npy (float32, logit
units = cp / 400, side-to-move relative), rows in final_{split}_v3.npz order. Uses the trainer's exact forward
(train_stream features_for + materialize32 + base_eval_stm, nnue_baseline.weights) = the engine's tuple eval.
Streams 100k-row chunks; columns are loaded once (never npz[key] inside the loop); RAM guard between chunks.
Resumable: a split whose .npy exists with the right length is skipped.
Usage: python training/build_s57_sidecar.py [--tup data/nnue/final_v2_wdl_ep8.tup] [--splits val,test,train]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnue_baseline as NB  # noqa: E402
import train_stream as S  # noqa: E402

T = S.T
ROOT = "C:/gitProjects/titanium-ataxx"
OUT = ROOT + "/data/nnue/s57"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tup", default=ROOT + "/data/nnue/final_v2_wdl_ep8.tup")
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    bias, W, D = NB.weights(a.tup)
    for split in a.splits.split(","):
        z = np.load(ROOT + "/data/nnue/final/final_%s_v3.npz" % split)
        o0, o1, blk, stm = (z[k] for k in ("occ0", "occ1", "blk", "stm"))
        o0, o1, blk = o0.astype(np.uint64), o1.astype(np.uint64), blk.astype(np.uint64)
        n = len(stm)
        dst = "%s/%s.npy" % (OUT, split)
        if os.path.exists(dst) and len(np.load(dst, mmap_mode="r")) == n:
            print("%s: exists (%d rows), skip" % (split, n), flush=True)
            continue
        res = np.zeros(n, np.float32)
        t0 = time.time()
        for lo in range(0, n, a.chunk):
            while S.free_gb() < a.min_free_gb:
                print("  low RAM (%.1f GB free), waiting" % S.free_gb(), flush=True)
                time.sleep(30)
            hi = min(n, lo + a.chunk)
            f = S.features_for(o0[lo:hi], o1[lo:hi], blk[lo:hi], stm[lo:hi])
            SP, DE = S.materialize32(*f, [0, 1, 2, 3])
            base = T.base_eval_stm(o0[lo:hi], o1[lo:hi], blk[lo:hi], stm[lo:hi])
            res[lo:hi] = ((W[SP].sum(1) + D[DE].sum(1) + bias + base) / 400.0).astype(np.float32)
            del f, SP, DE, base
            if (lo // a.chunk) % 20 == 0:
                el = time.time() - t0
                print("  %s %d/%d  %.0fs  eta %.0fs  free %.1f GB" % (split, hi, n, el, el / hi * (n - hi), S.free_gb()), flush=True)
        tmp = dst + ".tmp.npy"
        np.save(tmp, res)
        os.replace(tmp, dst)
        print("%s: wrote %s  mean %.3f std %.3f  (%.0fs)" % (split, dst, res.mean(), res.std(), time.time() - t0), flush=True)
        del o0, o1, blk, stm, res


if __name__ == "__main__":
    main()
