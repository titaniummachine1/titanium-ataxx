import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re

RE_BEST = re.compile(r'bestmove\s+(\S+)')


def best(exe, board, nodes, tup):
    cmd = [exe, 'bestmove', board, '--nodes', str(nodes), '--tuple', tup, '--tuple-native']
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = RE_BEST.search(out)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--fens', default='data/nnue/golden_src/lab_00.txt')
    ap.add_argument('--limit', type=int, default=150)
    ap.add_argument('--a', type=int, default=200000)
    ap.add_argument('--b', type=int, default=1000000)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    boards = []
    with open(a.fens) as f:
        for line in f:
            line = line.strip()
            if line:
                boards.append(line)
            if len(boards) >= a.limit:
                break
    print('positions %d' % len(boards), flush=True)

    def job(bd):
        m1 = best(a.exe, bd, a.a, a.net)
        m2 = best(a.exe, bd, a.a, a.net)
        m3 = best(a.exe, bd, a.b, a.net)
        return bd, m1, m2, m3

    t0 = time.time()
    same_a = 0
    same_cross = 0
    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for bd, m1, m2, m3 in ex.map(job, boards):
            sa = int(m1 is not None and m1 == m2)
            sc = int(m1 is not None and m3 is not None and m1 == m3)
            same_a += sa
            same_cross += sc
            rows.append((bd, m1, m2, m3, sa, sc))
    n = len(rows)
    with open('logs/stability.tsv', 'w') as f:
        f.write('board\tm_a1\tm_a2\tm_b\tagree_a\tagree_cross\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print('same-budget agree %.3f (%d/%d)  cross(%d vs %d) agree %.3f (%d/%d)  %.0fs'
          % (same_a / max(n, 1), same_a, n, a.a, a.b, same_cross / max(n, 1), same_cross, n,
             time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
