"""Proven-slice teacher labeling with Ataxx engine (engine-wdl.exe).

For each board position: set board, run raw wdl (W/D/L logits + score),
run MCTS go, read proven/pv/root visits. Output rows with:
  teacher_kind: 'proven' if the search proved the position else 'search'
  proven, pv, root_q, visits, evals, w0, w1, w2, w3 (value head logits)
Kill-safe: work is split into chunk files; completed chunks are skipped on
restart; outputs are tmp+replace. No in-place mutation of the final dataset.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
ENGINE = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "build", "engine-wdl.exe")
WEIGHTS = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "weights")
PHASES = ["p0-8.bin:0:9", "p8-15.bin:9:16", "p15-30.bin:16:31", "p30-50.bin:31:50"]
OUTDIR = os.path.join(ROOT, "data", "nnue", "proven")


def label_rows(rows, evals, engine=ENGINE):
    """rows: list of (occ0, occ1, blk, stm, ply). Returns list of dicts."""
    os.makedirs(OUTDIR, exist_ok=True)
    fd, inp = tempfile.mkstemp(prefix="pin_", suffix=".txt", dir=OUTDIR)
    fd2, outp = tempfile.mkstemp(prefix="pout_", suffix=".txt", dir=OUTDIR)
    os.close(fd)
    os.close(fd2)
    try:
        with open(inp, "w", encoding="ascii") as f:
            for r in rows:
                f.write("set %d %d %d\n%d\n" % r)
                f.write("legal\n")
                f.write("wdl %d %d %d\n%d\n" % r)
                f.write("stat\ngo evals %d\n" % evals)
            f.write("quit\n")
        with open(inp, "rb") as fin, open(outp, "wb") as fout:
            p = subprocess.run(
                [engine, "champion.bin", "cmd", "0", "0"] + PHASES,
                stdin=fin, stdout=fout, stderr=subprocess.PIPE,
                cwd=WEIGHTS, timeout=3600,
            )
        if p.returncode != 0:
            raise RuntimeError("engine exit %d: %s" % (
                p.returncode, p.stderr.decode("utf-8", "replace")[-300:]))
        with open(outp, encoding="ascii") as f:
            content = f.read()
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        res = []
        i = 0
        for (o0, o1, b, s, ply) in rows:
            leg = lines[i].strip() if i < len(lines) else ""
            w = lines[i + 1].split() if i + 1 < len(lines) else []
            st = lines[i + 2].split() if i + 2 < len(lines) else []
            g = lines[i + 3].split() if i + 3 < len(lines) else []
            i += 4
            if len(w) != 6 or len(st) != 3:
                raise RuntimeError("bad engine reply: %r / %r / %r" % (w, g, st))
            if leg == "" or (len(g) == 1 and g[0] == "pass"):
                me = bin(o0).count("1") - bin(o1).count("1")
                pv = 1.0 if me > 0 else (-1.0 if me < 0 else 0.0)
                res.append({
                    "occ0": o0, "occ1": o1, "blk": b, "stm": s, "ply": ply,
                    "wdl_q": float(w[0]), "wdl_score": float(w[1]),
                    "v0": float(w[2]), "v1": float(w[3]), "v2": float(w[4]),
                    "v3": float(w[5]),
                    "best": ["pass"], "evals": 0, "sims": 0, "root_q": pv,
                    "proven": 1, "pv": pv, "visits": 0,
                })
                continue
            if len(g) < 7:
                raise RuntimeError("bad go reply: %r" % (g,))
            res.append({
                "occ0": o0, "occ1": o1, "blk": b, "stm": s, "ply": ply,
                "wdl_q": float(w[0]), "wdl_score": float(w[1]),
                "v0": float(w[2]), "v1": float(w[3]), "v2": float(w[4]),
                "v3": float(w[5]),
                "best": [g[0], g[1], g[2], g[3]], "evals": int(g[4]), "sims": int(g[5]),
                "root_q": float(g[6]),
                "proven": int(st[0]), "pv": float(st[1]), "visits": int(st[2]),
            })
        if i != len(lines):
            raise RuntimeError("line mismatch %d != %d" % (i, len(lines)))
        return res
    finally:
        for x in (inp, outp):
            try:
                os.unlink(x)
            except OSError:
                pass


def chunk_rows(idx, occ0, occ1, blk, stm, ply):
    return [(int(occ0[i]), int(occ1[i]), int(blk[i]), int(stm[i]), int(ply[i]))
            for i in idx]


def save_chunk(rows, path):
    n = len(rows)
    arr = {
        "occ0": np.asarray([r["occ0"] for r in rows], dtype=np.uint64),
        "occ1": np.asarray([r["occ1"] for r in rows], dtype=np.uint64),
        "blk": np.asarray([r["blk"] for r in rows], dtype=np.uint64),
        "stm": np.asarray([r["stm"] for r in rows], dtype=np.uint8),
        "ply": np.asarray([r["ply"] for r in rows], dtype=np.int32),
        "proven": np.asarray([r["proven"] for r in rows], dtype=np.uint8),
        "pv": np.asarray([r["pv"] for r in rows], dtype=np.float32),
        "root_q": np.asarray([r["root_q"] for r in rows], dtype=np.float32),
        "visits": np.asarray([r["visits"] for r in rows], dtype=np.uint32),
        "evals": np.asarray([r["evals"] for r in rows], dtype=np.int32),
        "wdl_q": np.asarray([r["wdl_q"] for r in rows], dtype=np.float32),
        "wdl_score": np.asarray([r["wdl_score"] for r in rows], dtype=np.float32),
        "v0": np.asarray([r["v0"] for r in rows], dtype=np.float32),
        "v1": np.asarray([r["v1"] for r in rows], dtype=np.float32),
        "v2": np.asarray([r["v2"] for r in rows], dtype=np.float32),
        "v3": np.asarray([r["v3"] for r in rows], dtype=np.float32),
    }
    tmp = path + ".tmp"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-empty", type=int, default=30)
    a = ap.parse_args()

    z = np.load(a.in_npz, mmap_mode="r")
    o0 = np.asarray(z["occ0"], dtype=np.uint64)
    o1 = np.asarray(z["occ1"], dtype=np.uint64)
    blk = np.asarray(z["blk"], dtype=np.uint64)
    stm = np.asarray(z["stm"], dtype=np.uint8)
    empty = 49 - np.bitwise_count(o0 | o1).astype(np.int32)
    ply = np.clip(2 * (49 - empty), 0, 400).astype(np.int32)
    rng = np.random.default_rng(a.seed)
    idx = np.flatnonzero(empty >= a.min_empty)
    rng.shuffle(idx)
    idx = idx[:a.rows]
    print("candidate rows %d" % len(idx), flush=True)

    done = set()
    for ci in range(0, len(idx), a.chunk):
        if os.path.exists(os.path.join(OUTDIR, "%s_c%05d.npz" % (a.out_tag, ci))):
            done.add(ci)
    print("resume: %d/%d chunks done" % (len(done), (len(idx) + a.chunk - 1) // a.chunk), flush=True)

    jobs = []
    for ci in range(0, len(idx), a.chunk):
        if ci in done:
            continue
        jobs.append((ci, chunk_rows(idx[ci:ci + a.chunk], o0, o1, blk, stm, ply)))

    t0 = time.time()
    made = 0
    prov = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(label_rows, r, a.evals): c for c, r in jobs}
        for fut in as_completed(futs):
            ci = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print("ERROR chunk %d: %s" % (ci, e), flush=True)
                continue
            save_chunk(rows, os.path.join(OUTDIR, "%s_c%05d.npz" % (a.out_tag, ci)))
            made += len(rows)
            prov += sum(1 for r in rows if r["proven"])
            print("saved chunk %d: %d rows (%d proven) total=%d (%.0fs)" % (
                ci, len(rows), sum(1 for r in rows if r["proven"]), made, time.time() - t0), flush=True)
    man = {"tag": a.out_tag, "rows": made, "proven": prov,
           "evals": a.evals, "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUTDIR, a.out_tag + "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print(json.dumps(man, indent=2), flush=True)


if __name__ == "__main__":
    main()
