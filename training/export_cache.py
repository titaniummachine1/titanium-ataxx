"""Export dag.db -> packed binary cache (parse FEN ONCE, never again).

Row: occ0:u64, occ1:u64, blk:u64, stm:u8, out:f32, sc:f32 = 32 bytes.
400k rows = 12.8MB. Load = single read, zero parsing, zero sqlite.
Same bitboards the engine uses -> ports straight to Rust later.

Usage: python3 training/export_cache.py --top-wsum 400000 --out data/nnue/cache400k.npz
"""
import argparse
import sqlite3
import time

import numpy as np


def fen_to_bb(fen):
    occ0 = occ1 = blk = 0
    cells = 0
    for c in fen.strip().split()[0]:
        if c == '/':
            continue
        if c.isdigit():
            cells += int(c)
        elif c in '.xo#':
            if c == 'x':
                occ0 |= 1 << cells
            elif c == 'o':
                occ1 |= 1 << cells
            elif c == '#':
                blk |= 1 << cells
            cells += 1
    assert cells == 49, (fen, cells)
    return occ0, occ1, blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/nnue/dag.db')
    ap.add_argument('--top-wsum', type=int, default=400000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='data/nnue/cache400k.npz')
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    con.execute('pragma journal_mode=off')
    base = ('select fen, stm, visits, sum_outcome, best_score from nodes where margin_n>0')
    if a.limit:
        q = base + ' limit %d' % a.limit
    else:
        q = base + ' order by wsum desc limit %d' % a.top_wsum
    t0 = time.time()
    rows = con.execute(q).fetchall()
    con.close()
    print('fetched %d rows in %.0fs' % (len(rows), time.time() - t0), flush=True)

    n = len(rows)
    occ0 = np.zeros(n, dtype=np.uint64)
    occ1 = np.zeros(n, dtype=np.uint64)
    blk = np.zeros(n, dtype=np.uint64)
    stm = np.zeros(n, dtype=np.uint8)
    out = np.zeros(n, dtype=np.float32)
    sc = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i, (fen, s, vis, s_out, best) in enumerate(rows):
        o0, o1, b = fen_to_bb(fen)
        occ0[i], occ1[i], blk[i] = o0, o1, b
        stm[i] = s
        out[i] = max(-1.0, min(1.0, s_out / max(1, vis)))
        sc[i] = max(-2000.0, min(2000.0, best)) / 2000.0
    print('packed %d rows in %.0fs' % (n, time.time() - t0), flush=True)
    np.savez_compressed(a.out, occ0=occ0, occ1=occ1, blk=blk, stm=stm, out=out, sc=sc)
    import os
    print('wrote %s %.1fMB' % (a.out, os.path.getsize(a.out) / 1e6), flush=True)


if __name__ == '__main__':
    main()
