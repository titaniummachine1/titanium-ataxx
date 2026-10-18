#!/usr/bin/env python3
"""Split book positions by game phase (fill fraction) for phase studies.
Usage: python3 training/split_phase.py --books data/nnue/book100o.txt,... \
  --outdir data/nnue/phase
Writes phase_open.txt (<16 stones), phase_mid.txt (16-32), phase_end.txt (>32).
"""
import argparse
import os
import sys

sys.path.insert(0, 'training')
import book_gate as bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--books', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    buckets = {'phase_open.txt': [], 'phase_mid.txt': [], 'phase_end.txt': []}
    for bf in a.books.split(','):
        for line in open(bf):
            line = line.strip()
            if not line:
                continue
            cells, _ = bg.parse_line(line)
            n = sum(1 for v in cells if v in (1, 2))
            if n < 16:
                buckets['phase_open.txt'].append(line)
            elif n <= 32:
                buckets['phase_mid.txt'].append(line)
            else:
                buckets['phase_end.txt'].append(line)
    for fn, lines in buckets.items():
        p = os.path.join(a.outdir, fn)
        open(p, 'w').write('\n'.join(lines) + '\n')
        print('%s: %d positions' % (p, len(lines)))


if __name__ == '__main__':
    main()
