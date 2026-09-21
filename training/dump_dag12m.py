"""Dump dag.db usable rows (margin_n>0) -> label text chunks + base npz chunks.

Overnight step 1 of the 12.4M relabel. Streams the DB (no full fetch),
parses each FEN once, writes per-chunk Board::from_str lines for `label`
plus the bitboard/target columns for the later join. Resumable: chunks
whose outputs both exist are skipped unless --redo.

Usage: python training/dump_dag12m.py --chunks 16
  -> data/nnue/relab12m/lab_00.txt ... + base_00.npz ... + manifest.txt
"""
import argparse
import os
import sqlite3

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
    ap.add_argument('--out', default='data/nnue/relab12m')
    ap.add_argument('--chunks', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    con = sqlite3.connect(a.db)
    con.execute('pragma journal_mode=off')
    total = con.execute('select count(*) from nodes where margin_n>0').fetchone()[0]
    if a.limit:
        total = min(total, a.limit)
    per = (total + a.chunks - 1) // a.chunks
    print('usable %d chunks %d per-chunk ~%d' % (total, a.chunks, per), flush=True)

    man_path = os.path.join(a.out, 'manifest.txt')
    man = open(man_path, 'a' if os.path.exists(man_path) else 'w')
    for c in range(a.chunks):
        lo = c * per
        if lo >= total:
            break
        txt = os.path.join(a.out, 'lab_%02d.txt' % c)
        npz = os.path.join(a.out, 'base_%02d.npz' % c)
        # skip chunks fully done in a previous run (manifest already lists them,
        # unless --redo). Partial/stale chunks (e.g. from the killed smoke test
        # that reused --limit offsets) must be redone, so verify row counts.
        want = min(per, total - lo)
        done = False
        if os.path.exists(txt) and os.path.exists(npz) and not a.redo:
            try:
                z0 = np.load(npz)
                if len(z0['occ0']) == want and sum(1 for _ in open(txt, 'rb')) == want:
                    print('chunk %02d exists, skip' % c, flush=True)
                    done = True
            except Exception:
                done = False
        if done:
            continue
        q = ('select fen, stm, visits, sum_outcome, best_score, wsum from nodes '
             'where margin_n>0 order by rowid limit %d offset %d' % (per, lo))
        rows = con.execute(q).fetchall()
        n = len(rows)
        o0 = np.zeros(n, dtype=np.uint64)
        o1 = np.zeros(n, dtype=np.uint64)
        bl = np.zeros(n, dtype=np.uint64)
        stm = np.zeros(n, dtype=np.uint8)
        out = np.zeros(n, dtype=np.float32)
        sc = np.zeros(n, dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        with open(txt, 'w') as f:
            for i, (fen, s, vis, s_out, best, w) in enumerate(rows):
                x0, x1, b = fen_to_bb(fen)
                o0[i], o1[i], bl[i] = x0, x1, b
                st = int(s)
                stm[i] = st
                out[i] = max(-1.0, min(1.0, s_out / max(1, vis)))
                sc[i] = max(-2000.0, min(2000.0, best)) / 2000.0
                wsum[i] = max(0.0, w)
                # same emission as dump_fens.py (label-proven on 5k probe)
                f.write(''.join(
                    '#' if (b >> sq) & 1 else 'x' if (x0 >> sq) & 1
                    else 'o' if (x1 >> sq) & 1 else '.'
                    for sq in range(49)))
                f.write(' b\n' if st == 0 else ' w\n')
        np.savez_compressed(npz, occ0=o0, occ1=o1, blk=bl, stm=stm,
                            out=out, sc=sc, wsum=wsum)
        man.write('%02d %d\n' % (c, n))
        man.flush()
        print('chunk %02d rows %d done' % (c, n), flush=True)
    man.close()
    con.close()
    print('dump done', flush=True)


if __name__ == '__main__':
    main()
