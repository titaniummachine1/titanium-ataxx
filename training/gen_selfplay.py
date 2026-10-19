#!/usr/bin/env python3
"""Fresh selfplay from the REAL 4-corner startpos @50k nodes/move.

Why: old datasets contain 2-stone-start lines; the tree must not forget
real openings. Random opening 0-10 plies (variety: the engine is
deterministic) then engine-vs-itself to game end. Every ply recorded.
Output: per-worker npz chunks (occ0/occ1/blk/stm/outcome stm-relative),
joined afterwards. Rules = book_gate (consistent with gates).

Usage: python training/gen_selfplay.py --games 1500 --nodes 50000
         --workers 4 --outdir logs/selfplay50k --seed 7
"""
import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book_gate as bg
from gen_phase import legal as gen_legal
from place_bench import Engine

EXE = os.path.join('target', 'release', 'titanium-cli.exe')
CHAMP = r'C:\gitProjects\engines\titan_champ.tup'
START = 'x.....o' + '.' * 35 + 'o.....x'


def sq_bit(sq):
    return np.uint64(1) << np.uint64(sq)


def play_game(eng, rng, nodes, max_ply=400):
    cells = [1 if c == 'x' else (2 if c == 'o' else 0) for c in START]
    turn = 0
    fens = []
    # random opening 0..10 plies
    nopen = rng.randrange(0, 11)
    ply = 0
    while ply < nopen and not bg.over(cells):
        toks = [t for t in gen_legal(cells, turn) if t != '0000']
        if not toks:
            break
        tok = rng.choice(toks)
        if bg.apply(cells, turn, tok) != 'ok':
            break
        fens.append((bg.to_fen(cells, turn), turn))
        turn = 1 - turn
        ply += 1
    # engine selfplay
    passes = 0
    while ply < max_ply and not bg.over(cells):
        if not bg.has_moves(cells, turn):
            passes += 1
            if passes >= 2:
                break
            turn = 1 - turn
            ply += 1
            continue
        passes = 0
        fen = bg.to_fen(cells, turn)
        tok = eng.go(fen, nodes, timeout=300)
        if not tok or bg.apply(cells, turn, tok) != 'ok':
            return None  # forfeit/illegal: drop game
        fens.append((fen, turn))
        turn = 1 - turn
        ply += 1
    b, w = bg.counts(cells)
    if b == w:
        res = -1  # draw
    else:
        res = 0 if b > w else 1  # winning color as turn 0/1
    return [(f, t, res) for (f, t) in fens]


def worker(wi, games, nodes, seed, outdir, counter, lock):
    rng = random.Random(seed + wi * 7919)
    eng = Engine(EXE, ['serve', '--tuple', CHAMP], 'uai', [])
    eng.newgame()
    occ0, occ1, blk, stm, out = [], [], [], [], []
    done = 0
    try:
        while True:
            with lock:
                if counter[0] >= games:
                    break
                counter[0] += 1
            r = play_game(eng, rng, nodes)
            if not r:
                continue
            for fen, t, res in r:
                parts = fen.split(' ')
                rows = parts[0].split('/')
                o0 = o1 = bl = np.uint64(0)
                sq = 0
                bad = False
                for ch in '/'.join(rows):
                    if ch == '/':
                        continue
                    if ch.isdigit():
                        sq += int(ch)
                        continue
                    bit = sq_bit(sq)
                    if ch == 'x':
                        o0 |= bit
                    elif ch == 'o':
                        o1 |= bit
                    elif ch == '#':
                        bl |= bit
                    else:
                        bad = True
                    sq += 1
                if bad or sq != 49:
                    continue
                occ0.append(o0)
                occ1.append(o1)
                blk.append(bl)
                stm.append(t)
                if res == -1:
                    out.append(0.5)
                else:
                    out.append(1.0 if res == t else 0.0)
            done += 1
            if done % 25 == 0:
                fp = os.path.join(outdir, 'chunk_w%d_n%d.npz' % (wi, done))
                np.savez_compressed(fp, occ0=np.array(occ0), occ1=np.array(occ1),
                                    blk=np.array(blk), stm=np.array(stm, dtype=np.int64),
                                    outcome=np.array(out, dtype=np.float64))
    finally:
        try:
            eng.quit()
        except Exception:
            pass
    fp = os.path.join(outdir, 'chunk_w%d_final.npz' % wi)
    np.savez_compressed(fp, occ0=np.array(occ0), occ1=np.array(occ1),
                        blk=np.array(blk), stm=np.array(stm, dtype=np.int64),
                        outcome=np.array(out, dtype=np.float64))
    print('worker %d: %d games, %d positions' % (wi, done, len(out)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=1500)
    ap.add_argument('--nodes', type=int, default=50000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--outdir', default=os.path.join('logs', 'selfplay50k'))
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    counter = [0]
    lock = threading.Lock()
    import concurrent.futures as cf
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(lambda wi: worker(wi, a.games, a.nodes, a.seed,
                                      a.outdir, counter, lock),
                    range(a.workers)))
    print('DONE %d games in %.0fs' % (a.games, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
