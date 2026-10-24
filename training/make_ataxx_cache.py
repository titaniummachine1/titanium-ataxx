import os
import subprocess
import sys
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
CACHE = os.path.join(ROOT, "data", "nnue", "cache12M_clean.npz")
OUT = os.environ.get("ATAXX_CACHE_OUT", os.path.join(ROOT, "data", "nnue", "cache12M_ataxx.npz"))
ENGINE = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "build", "engine.exe")
WEIGHTS = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "weights")
TMP = os.path.join(ROOT, "data", "nnue", "ataxx_label")
PHASES = ["p0-8.bin:0:9", "p8-15.bin:9:16", "p15-30.bin:16:31", "p30-50.bin:31:50"]
CHUNKS = 4


def label():
    os.makedirs(TMP, exist_ok=True)
    z = np.load(CACHE, mmap_mode="r")
    n = len(z["occ0"])
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        n = min(n, limit)
    o0 = np.asarray(z["occ0"][:n])
    o1 = np.asarray(z["occ1"][:n])
    st = np.asarray(z["stm"][:n])
    empty = 49 - np.bitwise_count(o0 | o1)
    ply = np.clip(2 * (49 - empty), 0, 400).astype(np.int32)
    bounds = [(i * n // CHUNKS, (i + 1) * n // CHUNKS) for i in range(CHUNKS)]
    procs = []
    for c, (lo, hi) in enumerate(bounds):
        inp = os.path.join(TMP, "in_%d.txt" % c)
        out = os.path.join(TMP, "out_%d.txt" % c)
        with open(inp, "w", encoding="ascii") as f:
            buf = []
            for i in range(lo, hi):
                buf.append("raw %d %d %d %d\n" % (int(o0[i]), int(o1[i]), int(st[i]), int(ply[i])))
                if len(buf) >= 100000:
                    f.writelines(buf)
                    buf.clear()
            if buf:
                f.writelines(buf)
            f.write("quit\n")
        print("chunk %d rows %d input %s" % (c, hi - lo, inp), flush=True)
        p = subprocess.Popen(
            [ENGINE, "champion.bin", "cmd", "0", "0"] + PHASES,
            cwd=WEIGHTS,
            stdin=open(inp, "r", encoding="ascii"),
            stdout=open(out, "w", encoding="ascii"),
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
    began = time.time()
    for c, p in enumerate(procs):
        rc = p.wait()
        print("chunk %d exit %d elapsed %.0fs" % (c, rc, time.time() - began), flush=True)
    merge(n, bounds)


def merge(n, bounds):
    qs = []
    ss = []
    for c, (lo, hi) in enumerate(bounds):
        arr = np.loadtxt(os.path.join(TMP, "out_%d.txt" % c), dtype=np.float32)
        print("chunk %d parsed %s" % (c, arr.shape), flush=True)
        if arr.shape[0] != hi - lo:
            raise SystemExit("chunk %d row mismatch %d != %d" % (c, arr.shape[0], hi - lo))
        qs.append(arr[:, 0])
        ss.append(arr[:, 1])
    q = np.concatenate(qs)
    score = np.concatenate(ss)
    z = np.load(CACHE, mmap_mode="r")
    old_t = np.asarray(z["teacher"][:n], dtype=np.float32)
    out = np.asarray(z["out"][:n], dtype=np.float32)
    wsum = np.asarray(z["wsum"][:n], dtype=np.float32)
    print("ataxx q mean %.4f std %.4f score mean %.4f std %.4f" % (
        q.mean(), q.std(), score.mean(), score.std()), flush=True)
    print("corr q old_teacher %.3f corr q out %.3f corr score out %.3f" % (
        np.corrcoef(q, np.clip(old_t, -5000, 5000))[0, 1],
        np.corrcoef(q, out)[0, 1],
        np.corrcoef(score, out)[0, 1]), flush=True)
    np.savez_compressed(
        OUT,
        occ0=np.asarray(z["occ0"][:n]), occ1=np.asarray(z["occ1"][:n]),
        blk=np.asarray(z["blk"][:n]), stm=np.asarray(z["stm"][:n]),
        out=out, wsum=wsum,
        teacher=(q * 1000.0).astype(np.float32),
        sc=(np.clip(old_t, -1500.0, 1500.0) / 2000.0).astype(np.float32),
        ataxx_q=q.astype(np.float32), ataxx_score=score.astype(np.float32),
    )
    print("wrote %s %.1fMB" % (OUT, os.path.getsize(OUT) / 1e6), flush=True)


if __name__ == "__main__":
    label()

