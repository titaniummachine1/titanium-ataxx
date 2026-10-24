"""Relabel teacher_src==1 rows (KataGo positions, teacher == outcome) with
Ataxx_CPP raw W/D/L telemetry, so no row is outcome-only.

Why: measured 2026-10-22 — teacher_p == outcome exactly on teacher_src=1
(20,662,235 rows, 63.5% of train). Ataxx_CPP is the strongest local teacher.

How (fast): engine-wdl "batchwdl <in> <out>" runs the raw net forward
(no MCTS) on b0 b1 side ply lines. Measured ~0.76 ms/pos single-threaded;
N processes across cores scale ~Nx. Raw eval (not searched) is the speed
trade the user ordered ("takes less time to relabel").

Ply: dataset has no ply column. Using the documented convention from
build_ataxx_dataset.py ("teacher_ply": 2*(49-empty), clamped 0..400) =
2 * popcount(occ0|occ1). blk is all-zero in this dataset (verified).

Resumable: one .npz per chunk, tmp+rename, manifest with sha; skip-if-exists.
A kill costs at most the in-flight chunk. Nothing overwrites the dataset.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
ENGINE = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "build", "engine-wdl.exe")
WEIGHTS = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "weights")
PHASES = ["p0-8.bin:0:9", "p8-15.bin:9:16", "p15-30.bin:16:31", "p30-50.bin:31:50"]
CACHE = os.path.join(ROOT, "data", "nnue", "final", "final_train.npz")
OUTDIR = os.path.join(ROOT, "data", "nnue", "final", "ataxx_relabel")
SRC = 1  # teacher_src value to relabel (KataGo / outcome-only)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def chunk_task(ci, rows, occ0, occ1, stm, outdir, keep_tmp):
    n = len(rows)
    tmp_in = os.path.abspath(os.path.join(outdir, "tmp", "in_c%04d.txt" % ci))
    tmp_out = os.path.abspath(os.path.join(outdir, "tmp", "out_c%04d.txt" % ci))
    o0 = occ0[rows]
    o1 = occ1[rows]
    s = stm[rows]
    pieces = np.bitwise_count(o0) + np.bitwise_count(o1)
    ply = np.clip(2 * pieces, 0, 400).astype(np.int64)
    with open(tmp_in, "w") as f:
        for i in range(n):
            f.write("%d %d %d %d\n" % (o0[i], o1[i], s[i], ply[i]))
    cmd = [ENGINE, "champion.bin", "cmd", "0", "0"] + PHASES
    t0 = time.time()
    p = subprocess.run(cmd, cwd=WEIGHTS, input="batchwdl %s %s\nquit\n" % (tmp_in, tmp_out),
                       text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("chunk %d engine rc=%d %s" % (ci, p.returncode, p.stderr[-300:]))
    if not os.path.exists(tmp_out):
        raise RuntimeError("chunk %d engine produced no output (stdout=%r)"
                           % (ci, (p.stdout or '')[-200:]))
    q = np.empty(n, np.float32)
    score = np.empty(n, np.float32)
    v = np.empty((n, 4), np.float32)
    got = 0
    with open(tmp_out) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            q[got] = float(parts[0])
            score[got] = float(parts[1])
            for j in range(4):
                v[got, j] = float(parts[2 + j])
            got += 1
            if got >= n:
                break
    if got != n:
        raise RuntimeError("chunk %d expected %d lines got %d" % (ci, n, got))
    # softmax over W/D/L logits
    m = v[:, :3].max(axis=1, keepdims=True)
    e = np.exp(v[:, :3] - m)
    probs = e / e.sum(axis=1, keepdims=True)
    out_path = os.path.join(outdir, "ataxx_c%04d.npz" % ci)
    tmp_npz = out_path + ".tmp.npz"
    np.savez(tmp_npz, row=rows.astype(np.int64), q=q, score=score,
             v0=v[:, 0], v1=v[:, 1], v2=v[:, 2], v3=v[:, 3],
             p_win=probs[:, 0].astype(np.float32),
             p_draw=probs[:, 1].astype(np.float32),
             p_loss=probs[:, 2].astype(np.float32),
             ply=ply.astype(np.int16))
    os.replace(tmp_npz, out_path)
    if not keep_tmp:
        try:
            os.remove(tmp_in)
            os.remove(tmp_out)
        except OSError:
            pass
    return {'ci': ci, 'lo': int(rows[0]), 'hi': int(rows[-1]) + 1, 'rows': int(n),
            'npz': out_path, 'sha': sha256(out_path),
            'elapsed_s': round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--out-dir', default=OUTDIR)
    ap.add_argument('--chunk-rows', type=int, default=250000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--src', type=int, default=SRC)
    ap.add_argument('--limit-chunks', type=int, default=0,
                    help='debug: only first N chunks (partial test, not a full relabel)')
    ap.add_argument('--keep-tmp', action='store_true')
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, 'tmp'), exist_ok=True)
    mpath = os.path.join(a.out_dir, 'manifest.json')

    z = np.load(a.cache, mmap_mode='r')
    ts = np.asarray(z['teacher_src'])
    idx = np.where(ts == a.src)[0]
    n = len(idx)
    chunks = [(ci, idx[lo:min(lo + a.chunk_rows, n)])
              for ci, lo in enumerate(range(0, n, a.chunk_rows))]
    if a.limit_chunks:
        chunks = chunks[:a.limit_chunks]
    print('src %d rows %d chunks %d x ~%d' % (a.src, n, len(chunks), a.chunk_rows), flush=True)

    man = {'cache': os.path.abspath(a.cache), 'src': a.src, 'rows': n, 'chunks': []}
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath))
            if old.get('src') == a.src and old.get('rows') == n:
                man = old
        except Exception as e:
            print('manifest unreadable, rebuilding: %s' % e, flush=True)
    done = {c['ci'] for c in man['chunks'] if os.path.exists(c['npz'])}
    print('resuming: %d/%d chunks banked' % (len(done), len(chunks)), flush=True)

    if a.verify_only:
        # Recompute one banked chunk into a separate dir and compare arrays.
        cand = [c for c in man['chunks'] if os.path.exists(c['npz'])]
        if not cand:
            raise SystemExit('no banked chunks to verify')
        c = cand[len(cand) // 2]
        vdir = os.path.join(a.out_dir, 'verify')
        os.makedirs(os.path.join(vdir, 'tmp'), exist_ok=True)
        rows = idx[c['ci'] * a.chunk_rows:(c['ci'] + 1) * a.chunk_rows]
        r = chunk_task(c['ci'], rows, np.asarray(z['occ0']), np.asarray(z['occ1']),
                       np.asarray(z['stm']), vdir, True)
        old = np.load(c['npz'])
        new = np.load(r['npz'])
        for k in ('q', 'score', 'p_win', 'p_draw', 'p_loss'):
            if not np.array_equal(old[k], new[k]):
                raise SystemExit('PARITY FAIL chunk %d key %s' % (c['ci'], k))
        if not np.array_equal(old['row'], new['row']):
            raise SystemExit('PARITY FAIL chunk %d row' % c['ci'])
        print('verify: PARITY OK chunk %d rows %d sha %s' % (c['ci'], r['rows'], r['sha']))
        return

    if a.limit_chunks:
        print('DEBUG limit-chunks %d: partial relabel only' % a.limit_chunks, flush=True)

    occ0 = np.asarray(z['occ0'])
    occ1 = np.asarray(z['occ1'])
    stm = np.asarray(z['stm'])
    pending = [(ci, rows) for ci, rows in chunks if ci not in done]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(chunk_task, ci, rows, occ0, occ1, stm, a.out_dir, a.keep_tmp): ci
                for ci, rows in pending}
        for k, fut in enumerate(as_completed(futs), 1):
            ci = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print('CHUNK %d FAILED: %s' % (ci, e), flush=True)
                continue
            man['chunks'] = [c for c in man['chunks'] if c['ci'] != ci] + [rec]
            man['chunks'].sort(key=lambda c: c['ci'])
            json.dump(man, open(mpath + '.tmp', 'w'), indent=1)
            os.replace(mpath + '.tmp', mpath)
            print('chunk %d/%d rows %d banked %.1fs (%.0fs elapsed, %d done)'
                  % (ci + 1, len(chunks), rec['rows'], rec['elapsed_s'],
                     time.time() - t0, len(man['chunks'])), flush=True)
    print('done: %d/%d chunks (%.0fs)' % (len(man['chunks']), len(chunks), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
