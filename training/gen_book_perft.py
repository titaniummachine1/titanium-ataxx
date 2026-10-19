#!/usr/bin/env python3
"""Perft-enumerated opening book: exhaustive BFS from the 4-corner startpos,
deduplicated, shuffled. No random-play bias: every line to the target depth
is represented. Format matches book_gate.parse_line (49 chars + b/w).
Fairness comes from the runner (place_bench plays BOTH seats per line).

Usage: python training/gen_book_perft.py --out data/nnue/book50k.txt
         --n 50000 --min-ply 4 --max-ply 6 --seed 7
"""
import argparse
import random
import sys
from collections import deque

sys.path.insert(0, __import__('os').path.dirname(
    __import__('os').path.abspath(__file__)))
import book_gate as bg
from gen_phase import legal as gen_legal


def board_key(cells, turn):
    return (tuple(cells), turn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--min-ply', type=int, default=4)
    ap.add_argument('--max-ply', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    start = [0] * 49
    start[0 * 7 + 0] = 1
    start[6 * 7 + 6] = 1
    start[0 * 7 + 6] = 2
    start[6 * 7 + 0] = 2
    seen = {board_key(start, 0)}
    q = deque([(start, 0, 0)])  # cells, turn(0=b,1=w), ply
    pool = []
    expanded = 0
    while q and len(pool) < a.n * 2:  # oversample, dedup + sample later
        cells, turn, ply = q.popleft()
        expanded += 1
        toks = gen_legal(cells, turn)
        if toks == ['0000']:
            continue
        for tok in toks:
            nxt = list(cells)
            if bg.apply(nxt, turn, tok) != 'ok':
                continue
            k = board_key(nxt, 1 - turn)
            if k in seen:
                continue
            seen.add(k)
            np = ply + 1
            if a.min_ply <= np <= a.max_ply and not bg.over(nxt) \
                    and bg.has_moves(nxt, 0) and bg.has_moves(nxt, 1):
                pool.append((nxt, 1 - turn))
            if np < a.max_ply:
                q.append((nxt, 1 - turn, np))
    print('expanded %d unique %d collected %d (ply %d-%d)' %
          (expanded, len(seen), len(pool), a.min_ply, a.max_ply), flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    take = pool[:a.n]
    with open(a.out, 'w') as f:
        for cells, turn in take:
            f.write(''.join('.' if v == 0 else ('x' if v == 1 else 'o')
                            for v in cells) + (' b' if turn == 0 else ' w') + '\n')
    print('wrote %d lines to %s' % (len(take), a.out), flush=True)


if __name__ == '__main__':
    main()
