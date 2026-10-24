import argparse
import glob
import os
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
CACHE = os.path.join(ROOT, "data", "nnue", "cache12M_clean.npz")
LAB = os.path.join(ROOT, "data", "nnue", "ataxx_label")
OUT = os.path.join(ROOT, "data", "nnue", "cache12M_ataxx_clean.npz")
CHUNKS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dedup", action="store_true")
    a = ap.parse_args()
    z = np.load(CACHE, mmap_mode="r")
    n = len(z["occ0"])
    bounds = [(i * n // CHUNKS, (i + 1) * n // CHUNKS) for i in range(CHUNKS)]
    o0s, o1s, sts, olds, outs, wss, qs, scores = [], [], [], [], [], [], [], []
    for c, (lo, hi) in enumerate(bounds):
        f = os.path.join(LAB, "out_%d.txt" % c)
        arr = np.loadtxt(f, dtype=np.float32)
        m = arr.shape[0]
        print("chunk %d labeled %d/%d" % (c, m, hi - lo), flush=True)
        idx = slice(lo, lo + m)
        o0s.append(np.asarray(z["occ0"][idx], dtype=np.uint64))
        o1s.append(np.asarray(z["occ1"][idx], dtype=np.uint64))
        sts.append(np.asarray(z["stm"][idx], dtype=np.uint8))
        olds.append(np.asarray(z["teacher"][idx], dtype=np.float32))
        outs.append(np.asarray(z["out"][idx], dtype=np.float32))
        wss.append(np.asarray(z["wsum"][idx], dtype=np.float32))
        qs.append(arr[:, 0])
        scores.append(arr[:, 1])
    o0 = np.concatenate(o0s); o1 = np.concatenate(o1s); st = np.concatenate(sts)
    old = np.concatenate(olds); out = np.concatenate(outs); ws = np.concatenate(wss)
    q = np.concatenate(qs); score = np.concatenate(scores)
    del o0s, o1s, sts, olds, outs, wss, qs, scores
    keep = np.isfinite(q) & np.isfinite(score) & (q >= -1.0) & (q <= 1.0)
    print("valid label rows %d/%d" % (int(keep.sum()), len(q)), flush=True)
    o0, o1, st, old, out, ws, q, score = (x[keep] for x in (o0, o1, st, old, out, ws, q, score))
    empty = 49 - np.bitwise_count(o0 | o1)
    print("phase counts 0-8/8-15/15-30/30+ = %d/%d/%d/%d" % (
        int((empty < 9).sum()), int(((empty >= 9) & (empty < 16)).sum()),
        int(((empty >= 16) & (empty < 31)).sum()), int((empty >= 31).sum())), flush=True)
    print("q mean/std %.4f/%.4f score mean/std %.4f/%.4f" % (
        q.mean(), q.std(), score.mean(), score.std()), flush=True)
    print("corr q-old %.3f corr q-out %.3f" % (
        np.corrcoef(q, np.clip(old, -5000, 5000))[0, 1], np.corrcoef(q, out)[0, 1]), flush=True)
    if a.dedup:
        t0 = time.time()
        keys = np.empty(len(q), dtype=[("a", "<u8"), ("b", "<u8"), ("c", "u1")])
        keys["a"] = o0; keys["b"] = o1; keys["c"] = st
        _, first = np.unique(keys, return_index=True)
        first.sort()
        print("dedup %d -> %d (%.0fs)" % (len(q), len(first), time.time() - t0), flush=True)
        o0, o1, st, old, out, ws, q, score, empty = (x[first] for x in (o0, o1, st, old, out, ws, q, score, empty))
    np.savez_compressed(
        a.out,
        occ0=o0, occ1=o1,
        blk=np.zeros(len(o0), dtype=np.uint64),
        stm=st,
        out=out,
        wsum=ws,
        teacher=(q * 1000.0).astype(np.float32),
        sc=(np.clip(old, -1500.0, 1500.0) / 2000.0).astype(np.float32),
        ataxx_q=q.astype(np.float32),
        ataxx_score=score.astype(np.float32),
        empty=empty.astype(np.uint8),
    )
    print("wrote %s %.1fMB rows %d" % (a.out, os.path.getsize(a.out) / 1e6, len(o0)), flush=True)


if __name__ == "__main__":
    main()
