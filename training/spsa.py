#!/usr/bin/env python3
"""SPSA tuner for Titanium search/eval constants (autaxx playbook).

Perturbs SearchParams via UAI setoption (no rebuilds), plays a short
self-play match (theta+ vs theta-) with training/place_bench.py, and
updates with the Spall schedule. Logs JSONL, one row per iteration.

Usage:
  python3 training/spsa.py --iters 100 --games 80 --nodes 1000 --workers 2 \
    --bookdir data/nnue --books book100o.txt,book100p.txt,book100q.txt \
    --state logs/spsa_r1.json --log logs/spsa_r1.jsonl
State file holds current theta; resumes automatically.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys

TITAN = os.path.join('target', 'release', 'titanium-cli.exe')
CHAMP = r'C:\gitProjects\engines\titan_champ.tup'

PARAMS = [
    # name, init, min, max, c(perturb), a(step)
    ('Tempo', 150, 0, 1000, 20.0, 40.0),
    ('ContactPen', 15, 0, 200, 3.0, 6.0),
    ('MulticapPen', 20, 0, 200, 4.0, 8.0),
    ('RFP0', 330, 0, 2000, 40.0, 80.0),
    ('RFP1', 189, 0, 2000, 50.0, 100.0),
    ('RFP2', 491, 0, 2000, 70.0, 140.0),
    ('RFP3', 754, 0, 2000, 100.0, 200.0),
]


def setup_str(theta, names, plus, ck):
    parts = []
    for n, d in zip(names, plus):
        i = {p[0]: p for p in PARAMS}[n]
        v = theta[n] + d * ck * i[4]
        v = min(i[3], max(i[2], v))
        parts.append('setoption name %s value %d' % (n, round(v)))
    return ';'.join(parts)


def run_match(a_setup, b_setup, book, games, nodes, movetime, workers, seed, out):
    cmd = [sys.executable, os.path.join('training', 'place_bench.py'),
           '--book', book, '--games', str(games),
           '--a-name', 'plus', '--a-exe', TITAN,
           '--a-args', 'serve --tuple %s' % CHAMP,
           '--a-setup', a_setup,
           '--b-name', 'minus', '--b-exe', TITAN,
           '--b-args', 'serve --tuple %s' % CHAMP,
           '--b-setup', b_setup,
           '--workers', str(workers), '--seed', str(seed), '--out', out]
    if movetime:
        cmd += ['--movetime', str(movetime)]
    else:
        cmd += ['--nodes', str(nodes)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r'total: plus (\d+) - minus (\d+) draws (\d+)', (r.stdout or '') + (r.stderr or ''))
    if not m:
        # fall back to the out file
        try:
            txt = open(out).read()
            m = re.search(r'total: plus (\d+) - minus (\d+) draws (\d+)', txt)
        except OSError:
            m = None
    if not m:
        return None
    w, l, d = map(int, m.groups())
    return w, l, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--games', type=int, default=80)
    ap.add_argument('--nodes', type=int, default=1000)
    ap.add_argument('--movetime', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--books', default='book100o.txt,book100p.txt,book100q.txt')
    ap.add_argument('--bookdir', default=os.path.join('data', 'nnue'))
    ap.add_argument('--state', default=os.path.join('logs', 'spsa_r1.json'))
    ap.add_argument('--log', default=os.path.join('logs', 'spsa_r1.jsonl'))
    ap.add_argument('--A', type=float, default=10.0)
    ap.add_argument('--params', default='',
                    help='comma list to tune a subset (default: all)')
    ap.add_argument('--astate', default='',
                    help='override per-param a values like Tempo=120,RFP0=200')
    a = ap.parse_args()

    names = [p[0] for p in PARAMS]
    if a.params:
        want = set(x.strip() for x in a.params.split(','))
        names = [n for n in names if n in want]
    amul = {}
    if a.astate:
        for kv in a.astate.split(','):
            k, v = kv.split('=')
            amul[k.strip()] = float(v)
    if os.path.exists(a.state):
        st = json.load(open(a.state))
        theta = st['theta']
        k0 = st['iter']
    else:
        theta = {p[0]: float(p[1]) for p in PARAMS}
        k0 = 0
    books = [os.path.join(a.bookdir, b.strip()) for b in a.books.split(',')]
    rng = random.Random(12345 + k0)
    alpha, gamma = 0.602, 0.101

    logf = open(a.log, 'a')
    for k in range(k0, k0 + a.iters):
        ck = 1.0 / ((k + 1) ** gamma)
        ak = 1.0 / ((k + 1 + a.A) ** alpha)
        plus = [1 if rng.random() < 0.5 else -1 for _ in names]
        sp = setup_str(theta, names, plus, ck)
        sm = setup_str(theta, names, [-d for d in plus], ck)
        book = books[k % len(books)]
        out = os.path.join('logs', 'spsa_k%d.txt' % k)
        res = run_match(sp, sm, book, a.games, a.nodes, a.movetime,
                        a.workers, 1000 + k, out)
        if res is None:
            print('iter %d: MATCH FAILED, skipping' % k, flush=True)
            continue
        w, l, d = res
        n = w + l + d
        s = (w + d / 2) / n if n else 0.5
        row = {'iter': k, 'w': w, 'l': l, 'd': d, 'score': round(s, 4)}
        for j, nm in enumerate(names):
            i = {p[0]: p for p in PARAMS}[nm]
            g = (s - 0.5) / (ck * i[4] * plus[j]) if ck * i[4] else 0.0
            step = ak * i[5] * amul.get(nm, 1.0) * g
            theta[nm] = min(i[3], max(i[2], theta[nm] + step))
            row[nm] = round(theta[nm], 2)
        row['theta'] = {nm: round(theta[nm], 2) for nm in names}
        logf.write(json.dumps(row) + '\n')
        logf.flush()
        json.dump({'iter': k + 1, 'theta': theta}, open(a.state, 'w'))
        print('iter %d: %d-%d-%d score=%.3f theta=%s' % (
            k, w, l, d, s, row['theta']), flush=True)
    logf.close()


if __name__ == '__main__':
    main()
