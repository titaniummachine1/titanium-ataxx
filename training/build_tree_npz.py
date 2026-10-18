#!/usr/bin/env python3
"""Tree dump -> training npz WITH edges (hive-mind distillation).
Each visited node: target = backed-up search value.
Weight = min(|backed - static|/500, 2): gentle, bounded.
Edges: parent->child row indices (joined by board hash; TT-cutoff endpoints
without rows are dropped, counted). Mate rows dropped.
Input: tree dump(s) + sibling .edges file(s). 6-col rows (with hash) or
legacy 5-col rows (hash = -1, edges skipped for those files).
"""
import sys

import numpy as np

CAP = 5000.0


def parse(path):
    occ0, occ1, blk, stm, teach, ws, hs = [], [], [], [], [], [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.strip().split()
            if len(p) not in (5, 6):
                continue
            board, side = p[0], p[1]
            try:
                st, ba, de = int(p[2]), int(p[3]), int(p[4])
                h = int(p[5]) if len(p) == 6 else -1
            except ValueError:
                continue
            if len(board) != 49 or abs(ba) > CAP:
                continue
            o0 = o1 = b = 0
            for sq, ch in enumerate(board):
                if ch == '#':
                    b |= 1 << sq
                elif ch == 'x':
                    o0 |= 1 << sq
                elif ch == 'o':
                    o1 |= 1 << sq
            occ0.append(o0)
            occ1.append(o1)
            blk.append(b)
            stm.append(0 if side == 'b' else 1)
            teach.append(max(-1500, min(1500, ba)))
            ws.append(min(abs(ba - st) / 500.0, 2.0))
            hs.append(h)
    return occ0, occ1, blk, stm, teach, ws, hs


def main():
    outs = sys.argv[1:]
    assert outs and '--out' in outs
    oi = outs.index('--out')
    files, out = outs[:oi], outs[oi + 1]
    O0, O1, B, S, T, W, H = [], [], [], [], [], [], []
    EP, EC, edgemiss = [], [], 0
    for f in files:
        o0, o1, blk, stm, t, w, hs = parse(f)
        base = len(T)
        h2i = {}
        for i, h in enumerate(hs):
            if h != -1 and h not in h2i:
                h2i[h] = base + i
        O0.extend(o0)
        O1.extend(o1)
        B.extend(blk)
        S.extend(stm)
        T.extend(t)
        W.extend(w)
        H.extend(hs)
        try:
            with open(f + '.edges', encoding='utf-8', errors='replace') as ef:
                for line in ef:
                    p = line.strip().split()
                    if len(p) != 2:
                        continue
                    try:
                        a, b = int(p[0]), int(p[1])
                    except ValueError:
                        continue
                    if a in h2i and b in h2i:
                        EP.append(h2i[a])
                        EC.append(h2i[b])
                    else:
                        edgemiss += 1
        except OSError:
            pass
    t = np.array(T, dtype=np.float32)
    w = np.array(W, dtype=np.float64)
    n = len(t)
    z = {'occ0': np.array(O0, np.uint64), 'occ1': np.array(O1, np.uint64),
         'blk': np.array(B, np.uint64), 'stm': np.array(S, np.uint8),
         'teacher': t, 'wsum': w, 'out': np.zeros(n),
         'sc': t.astype(np.float64),
         'edge_p': np.array(EP, np.int64), 'edge_c': np.array(EC, np.int64)}
    np.savez_compressed(out, **z)
    print('wrote %s n=%d edges=%d missed=%d teach mean %+.0f wsum mean %.2f' % (
        out, n, len(EP), edgemiss, t.mean(), w.mean()))


if __name__ == '__main__':
    main()
