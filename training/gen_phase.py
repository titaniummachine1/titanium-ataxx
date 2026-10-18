#!/usr/bin/env python3
"""Generate mid/endgame book positions via random playouts from startpos.
Usage: python3 training/gen_phase.py --games 400 --outdir data/nnue/phase
Appends sampled positions (ply 15-25 -> mid pool, ply 30+ -> end pool).
Line format matches make_book output (49 chars + side).
"""
import argparse
import os
import random
import sys

sys.path.insert(0, 'training')
import book_gate as bg

START = 'x.....o' + '.' * 35 + 'o.....x'


def board_str(cells, turn):
    return ''.join('.' if v == 0 else ('x' if v == 1 else 'o') for v in cells) + \
        (' b' if turn == 0 else ' w')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=99)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = random.Random(a.seed)
    mid, end = [], []
    for _ in range(a.games):
        cells = [1 if c == 'x' else (2 if c == 'o' else 0) for c in START]
        turn = 0
        ply = 0
        while ply < 60 and not bg.over(cells):
            if not bg.has_moves(cells, turn):
                turn = 1 - turn
                ply += 1
                continue
            mv = rng.choice(list(legal(cells, turn)))
            bg.apply(cells, turn, mv)
            turn = 1 - turn
            ply += 1
            n = sum(1 for v in cells if v in (1, 2))
            if ply in (16, 20, 24) and 16 <= n <= 32:
                mid.append(board_str(cells, turn))
            if ply % 4 == 0 and ply >= 28 and n > 28 and not bg.over(cells):
                end.append(board_str(cells, turn))
    for fn, lines in (('phase_mid.txt', mid), ('phase_end.txt', end)):
        p = os.path.join(a.outdir, fn)
        prev = open(p).read().splitlines() if os.path.exists(p) else []
        seen = set(prev)
        new = [l for l in lines if l not in seen]
        open(p, 'a').write(''.join(l + '\n' for l in new))
        print('%s: +%d (total %d)' % (p, len(new), len(prev) + len(new)))


def legal(cells, side):
    piece = 1 if side == 0 else 2
    out = []
    for sq, v in enumerate(cells):
        if v != piece:
            continue
        for t in bg.neigh2(sq):
            if cells[t] == 0:
                f, r = sq % 7, sq // 7
                tf, tr = t % 7, t // 7
                if max(abs(f - tf), abs(r - tr)) == 1:
                    out.append(bg.sq_name(t))
                else:
                    out.append(bg.sq_name(sq) + bg.sq_name(t))
    # pass if nothing
    empt = [t for t in range(49) if cells[t] == 0]
    if not out and empt:
        return ['0000']
    return out


if __name__ == '__main__':
    main()
