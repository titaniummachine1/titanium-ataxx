#!/usr/bin/env python3
"""Tree dump -> training npz (hive-mind distillation).
Each visited node: target = backed-up search value (NOT static eval).
Weight = min(|backed - static|/500, 2): gentle, bounded — volume of leaves
does the work (many tiny adjustments, not few big ones). Mate rows dropped.
Input: one or more tree dump files (board side static backed depth).
"""
import sys

import numpy as np

CAP = 5000.0


def parse(path):
    occ0, occ1, blk, stm, teach, ws = [], [], [], [], [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            board, side, st, ba, de = p
            if len(board) != 49:
                continue
            try:
                st, ba, de = int(st), int(ba), int(de)
            except ValueError:
                continue
            if abs(ba) > CAP:
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
    return occ0, occ1, blk, stm, teach, ws


def main():
    outs = sys.argv[1:]
    assert outs, 'usage: build_tree_npz.py <tree1.txt> [tree2.txt ...] --out X'
    assert '--out' in outs
    oi = outs.index('--out')
    files, out = outs[:oi], outs[oi + 1]
    O0, O1, B, S, T, W = [], [], [], [], [], []
    for f in files:
        o0, o1, blk, stm, t, w = parse(f)
        O0.extend(o0)
        O1.extend(o1)
        B.extend(blk)
        S.extend(stm)
        T.extend(t)
        W.extend(w)
    t = np.array(T, dtype=np.float32)
    w = np.array(W, dtype=np.float64)
    n = len(t)
    z = {'occ0': np.array(O0, np.uint64), 'occ1': np.array(O1, np.uint64),
         'blk': np.array(B, np.uint64), 'stm': np.array(S, np.uint8),
         'teacher': t, 'wsum': w, 'out': np.zeros(n),
         'sc': t.astype(np.float64)}
    np.savez_compressed(out, **z)
    print('wrote %s n=%d teach mean %+.0f wsum mean %.2f max %.1f' % (
        out, n, t.mean(), w.mean(), w.max()))


if __name__ == '__main__':
    main()
