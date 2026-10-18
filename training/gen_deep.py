#!/usr/bin/env python3
"""Fresh deep labels from the CURRENT champ: deep-search positions @500k.
No games needed — searches each position directly. Targets for alignment
that can't be stale (same binary that will use the tuned params).
Usage:
  python3 training/gen_deep.py --pos data/nnue/phase/phase_open.txt ... \
    --nodes 500000 --workers 4 --out logs/deep_fresh.jsonl
Line format: {"fen": ..., "deep": cp}.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, 'training')
import book_gate as bg
import place_bench as pb

EXE = os.path.join('target', 'release', 'titanium-cli.exe')
CHAMP = r'C:\gitProjects\engines\titan_champ.tup'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', nargs='+', required=True)
    ap.add_argument('--nodes', type=int, default=500000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--timeout', type=int, default=600)
    a = ap.parse_args()

    fens = []
    for p in a.pos:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            cells, turn = bg.parse_line(line)
            fens.append(bg.to_fen(cells, turn))
    print('positions: %d' % len(fens), flush=True)

    lock = threading.Lock()
    outf = open(a.out, 'w')

    def work(fi):
        cells = None
        e = pb.Engine(EXE, ['serve', '--tuple', CHAMP], 'uai', [])
        e.newgame()
        n_done = [0]
        for i in fi:
            sc, mv = e.go_score(fens[i], a.nodes, timeout=a.timeout)
            with lock:
                outf.write(json.dumps({'fen': fens[i], 'deep': sc}) + '\n')
                n_done[0] += 1
                if n_done[0] % 25 == 0:
                    outf.flush()
                    print('labeled %d/%d' % (n_done[0], len(fens)), flush=True)
        e.quit()

    chunks = [list(range(i, len(fens), a.workers)) for i in range(a.workers)]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, chunks))
    outf.close()
    print('saved %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
