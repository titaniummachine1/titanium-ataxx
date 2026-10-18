#!/usr/bin/env python3
"""Idle relabel worker: deep-label important positions when no gates queued.
Priority: regret positions first (blame_ranked.txt order = importance),
top 200 @1M nodes, rest @500k; then common positions (cache12M sampled by
wsum = commonness bias) @500k. Appends {fen, deep} JSONL, skips done.
Runs until STOP_AUTO appears or sources exhaust.
Usage: python3 training/relabel_idle.py --out logs/deep_idle.jsonl
  --regret data/nnue/blame_ranked.txt --regret-1m 200 --cache data/nnue/cache12M.npz
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, 'training')
import book_gate as bg
import place_bench as pb

EXE = os.path.join('target', 'release', 'titanium-cli.exe')
CHAMP = r'C:\gitProjects\engines\titan_champ.tup'
STOP = os.path.join('training', 'STOP_AUTO')


def parse_blame(line):
    parts = line.split('#')[0].strip().split()
    cells, turn = bg.parse_line(parts[0] + ' ' + parts[1])
    return bg.to_fen(cells, turn)


def scorer_setup(nodes):
    e = pb.Engine(EXE, ['serve', '--tuple', CHAMP], 'uai', [])
    e.newgame()
    return e


def label_list(fens, nodes, outf, workers, tag):
    done = set()
    if os.path.exists(outf):
        for line in open(outf):
            try:
                done.add(json.loads(line)['fen'])
            except (ValueError, KeyError):
                pass
    todo = [(i, f) for i, f in enumerate(fens) if f not in done]
    print('%s: %d todo (%d done already)' % (tag, len(todo), len(done)), flush=True)
    lock = threading.Lock()
    engines = {}

    def work(item):
        wi, (i, fen) = item
        if os.path.exists(STOP):
            return None
        if wi not in engines:
            engines[wi] = scorer_setup(nodes)
        sc, _ = engines[wi].go_score(fen, nodes, timeout=1200)
        with lock:
            open(outf, 'a').write(json.dumps({'fen': fen, 'deep': sc}) + '\n')
        return i

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            n = 0
            for r in ex.map(work, [((k % workers), t) for k, t in enumerate(todo)]):
                if r is None:
                    break
                n += 1
                if n % 50 == 0:
                    print('%s labeled %d/%d' % (tag, n, len(todo)), flush=True)
    finally:
        for e in engines.values():
            try:
                e.quit()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('logs', 'deep_idle.jsonl'))
    ap.add_argument('--regret', default=os.path.join('data', 'nnue', 'blame_ranked.txt'))
    ap.add_argument('--regret-1m', type=int, default=200)
    ap.add_argument('--common-n', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()

    if os.path.exists(STOP):
        print('STOP present, exiting', flush=True)
        return
    regret = []
    if os.path.exists(a.regret):
        for line in open(a.regret):
            line = line.strip()
            if line:
                regret.append(parse_blame(line))
    print('regret positions: %d' % len(regret), flush=True)
    label_list(regret[:a.regret_1m], 1000000, a.out, a.workers, 'regret-1m')
    if os.path.exists(STOP):
        return
    label_list(regret[a.regret_1m:], 500000, a.out, a.workers, 'regret-500k')
    if os.path.exists(STOP):
        return
    # common bias: sample cache12M weighted by wsum
    print('sampling common positions...', flush=True)
    d = np.load(os.path.join('data', 'nnue', 'cache12M.npz'), mmap_mode='r')
    w = np.asarray(d['wsum'], dtype=np.float64)
    idx = np.random.RandomState(a.seed).choice(
        len(w), size=min(a.common_n * 5, len(w)), replace=False,
        p=w / w.sum())
    fens = []
    for i in idx:
        o0, o1, blk, stm = int(d['occ0'][i]), int(d['occ1'][i]), int(d['blk'][i]), int(d['stm'][i])
        cells = []
        for sq in range(49):
            b = (o0 >> sq) & 1, (o1 >> sq) & 1, (blk >> sq) & 1
            cells.append(1 if b[0] else (2 if b[1] else (3 if b[2] else 0)))
        turn = int(stm)
        fens.append(bg.to_fen(cells, turn))
        if len(fens) >= a.common_n:
            break
    label_list(fens, 500000, a.out, a.workers, 'common-500k')
    print('relabel done', flush=True)


if __name__ == '__main__':
    main()
