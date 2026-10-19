#!/usr/bin/env python3
"""SPSA on SEARCH LOSS (not games): shallow search vs our own deep labels.

Target = deep teacher scores from cache12M (recent deep labels).
Prediction = shallow search (2k nodes) score with candidate search params.
Loss = mean squared error on sigmoid-mapped scores. Search-only params
(RFP/LMR/null gates) — eval weights stay out (handled by R2 + training).

Usage:
  python3 training/search_loss.py --iters 80 --positions 300 --nodes 2000 \
    --workers 4 --state logs/sloss.json --log logs/sloss.jsonl
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import place_bench as pb

EXE = os.path.join('target', 'release', 'titanium-cli.exe')
CHAMP = r'C:\gitProjects\engines\titan_champ.tup'
NPZ = os.path.join('data', 'nnue', 'cache12M.npz')

# search-only params: name, init, min, max, c(perturb), a(step)
# NOTE: loss-scale is ~1e-3, so a-values are calibrated from smoke gradients
# (|g| ~ 1e-6..1e-4): a_i ~= desired_step / (0.24 * typical|g|). Recalibrate
# from the log if thetas freeze (a too small) or oscillate (a too big).
# R1 thetas (RFP1 189, NullMaxFill 29, ...) validated +67 in games and were
# folded into SearchParams::default 2026-10-18; R2 refines from there.
PARAMS = [
    ('LMRMode', 2, 2, 2, 0.0, 0.0),  # pinned: table mode (not tuned)
    ('LMRT0', 0, 0, 8, 1.0, 3000.0),
    ('LMRT1', 2, 0, 8, 1.0, 3000.0),
    ('LMRT2', 2, 0, 8, 1.0, 3000.0),
    ('LMRT3', 2, 0, 8, 1.0, 3000.0),
    ('LMRT4', 2, 0, 8, 1.0, 3000.0),
    ('LMRT5', 2, 0, 8, 1.0, 3000.0),
    ('LMRT6', 2, 0, 8, 1.0, 3000.0),
    ('LMRT7', 2, 0, 8, 1.0, 3000.0),
    ('LMRT8', 2, 0, 8, 1.0, 3000.0),
    ('LMRT9', 3, 0, 8, 1.0, 3000.0),
    ('LMRT10', 3, 0, 8, 1.0, 3000.0),
    ('LMRT11', 3, 0, 8, 1.0, 3000.0),
    ('LMRT12', 3, 0, 8, 1.0, 3000.0),
    ('LMRT13', 3, 0, 8, 1.0, 3000.0),
    ('LMRT14', 3, 0, 8, 1.0, 3000.0),
    ('LMRT15', 3, 0, 8, 1.0, 3000.0),
    ('LMRGap1', 2000000000, 0, 2000000000, 20000.0, 200000000.0),
    ('LMRGap2', 2000000000, 0, 2000000000, 20000.0, 200000000.0),
    # Intent ordering weights (a/c ~ 50k per R1 calibration).
    ('JumpEscape', 1500, 0, 10000, 500.0, 25000000.0),
    ('JumpAttack', 600, 0, 10000, 500.0, 25000000.0),
    ('JumpGamble', 800, 0, 10000, 500.0, 25000000.0),
    # ProbCut params (pruning = value-visible, RFP-scale c/a).
    ('PCutMargin', 250, 0, 2000, 100.0, 5000000.0),
    ('PCutDepth', 5, 3, 8, 1.0, 2000.0),
    ('PCutReduce', 3, 1, 6, 1.0, 2000.0),
    ('CurveBase', 2, 0, 6, 1.0, 3000.0),
    ('CurveSlope', 6, 0, 64, 4.0, 12000.0),
    ('CurveDepth', 2, 0, 64, 4.0, 12000.0),
    ('CurveCap', 5, 1, 8, 1.0, 3000.0),
    ('JumpDanger', 120, 0, 2000, 100.0, 5000000.0),
    ('JumpHole', 100, 0, 2000, 100.0, 5000000.0),
    ('JumpSafe', 0, 0, 2000, 100.0, 5000000.0),
    # Phase-bucket output biases (k=8 bisection on pieces).
    ('BucketBias0', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias1', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias2', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias3', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias4', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias5', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias6', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias7', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias8', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias9', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias10', 0, -2000, 2000, 100.0, 5000000.0),
    ('BucketBias11', 0, -2000, 2000, 100.0, 5000000.0),
]


def sig(x):
    return 1.0 / (1.0 + math.exp(-x / 400.0))


def bit_to_fen(occ0, occ1, blk, stm):
    ch = []
    for i in range(49):
        b = (occ0 >> i) & 1, (occ1 >> i) & 1, (blk >> i) & 1
        ch.append('x' if b[0] else ('o' if b[1] else ('#' if b[2] else '.')))
    rows = [''.join(ch[r * 7:(r + 1) * 7]) for r in range(7)]
    out = []
    for r in rows:
        s, gap = '', 0
        for c in r:
            if c == '.':
                gap += 1
            else:
                if gap:
                    s += str(gap)
                    gap = 0
                s += c
        if gap:
            s += str(gap)
        out.append(s)
    return '/'.join(out) + (' x 0 1' if stm == 0 else ' o 0 1')


def build_dataset(npos, seed):
    d = np.load(NPZ, mmap_mode='r')
    n = len(d['occ0'])
    # Vectorized filter (scalar mmap access is ~0.4s/elem on this box):
    # impactful = high weight, tense teacher (|teacher| < 600 cp).
    teacher = np.asarray(d['teacher'], dtype=np.float64)
    wsum = np.asarray(d['wsum'], dtype=np.float64)
    mask = (np.abs(teacher) < 600) & (wsum > 1.0)
    cand = np.nonzero(mask)[0][::997]
    rng = random.Random(seed)
    cand = cand.tolist()
    rng.shuffle(cand)
    ds = []
    occ0 = np.asarray(d['occ0'])
    occ1 = np.asarray(d['occ1'])
    blk = np.asarray(d['blk'])
    stm = np.asarray(d['stm'])
    for i in cand:
        t = float(teacher[i])
        if abs(t) > 2000:
            continue
        ds.append((bit_to_fen(int(occ0[i]), int(occ1[i]),
                              int(blk[i]), int(stm[i])), t))
        if len(ds) >= npos:
            break
    return ds


def load_align(path, npos, seed):
    """Targets = our own 1M-node deep labels (label --nodes 1000000).
    No regeneration: these ARE the deep search answers (teacher column)."""
    d = np.load(path, mmap_mode='r')
    n = len(d['occ0'])
    teacher = np.asarray(d['teacher'], dtype=np.float64)
    wsum = np.asarray(d['wsum'], dtype=np.float64)
    mask = (np.abs(teacher) < 600) & (wsum > 0)
    cand = np.nonzero(mask)[0]
    rng = random.Random(seed)
    cand = cand.tolist()
    rng.shuffle(cand)
    occ0 = np.asarray(d['occ0'])
    occ1 = np.asarray(d['occ1'])
    blk = np.asarray(d['blk'])
    stm = np.asarray(d['stm'])
    ds = []
    for i in cand:
        ds.append((bit_to_fen(int(occ0[i]), int(occ1[i]),
                              int(blk[i]), int(stm[i])), float(teacher[i])))
        if len(ds) >= npos:
            break
    return ds


class Scorer:
    """Persistent engine that returns info scores for FENs."""

    def __init__(self):
        self.p = subprocess.Popen(
            [EXE, 'serve', '--tuple', CHAMP],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def setup(self, lines):
        self._send('uai')
        self._until('uaiok')
        for s in lines:
            self._send(s)
        self._send('isready')
        self._until('readyok')
        self._send('uainewgame')

    def _send(self, s):
        self.p.stdin.write(s + '\n')
        self.p.stdin.flush()

    def _rl(self):
        return self.p.stdout.readline().strip()

    def _until(self, tok, limit=300):
        for _ in range(limit):
            if self._rl().startswith(tok):
                return
        raise RuntimeError('no ' + tok)

    def score(self, fen, nodes):
        self._send('position fen ' + fen)
        self._send('go nodes %d' % nodes)
        sc, mv = None, None
        for _ in range(1200):
            l = self._rl()
            if l.startswith('info '):
                toks = l.split()
                if 'cp' in toks:
                    try:
                        sc = float(toks[toks.index('cp') + 1])
                    except (ValueError, IndexError):
                        pass
            if l.startswith('bestmove'):
                ps = l.split()
                mv = ps[1] if len(ps) > 1 else '0000'
                break
        return (sc if sc is not None else 0.0), (mv or '0000')

    def quit(self):
        try:
            self._send('quit')
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


WLOCK = threading.Lock()
SCORERS = {}


def worker_job(args):
    wi, fen, nodes = args
    return SCORERS[wi].score(fen, nodes)


WORK_CTX = {}


def eval_scores(theta, fens, nodes, workers):
    """Raw (score, bestmove) per position under candidate params."""
    setup = ['setoption name %s value %d' % (k, round(v)) for k, v in theta.items()]
    global SCORERS
    SCORERS = {}
    for wi in range(workers):
        sc = Scorer()
        sc.setup(setup)
        SCORERS[wi] = sc
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(worker_job,
                              [(wi % workers, f, nodes) for wi, f in enumerate(fens)]))
    finally:
        for sc in SCORERS.values():
            sc.quit()
        SCORERS = {}
    return out


def eval_loss(theta, targets, nodes, workers):
    """Sigmoid-MSE vs deep targets + move agreement fraction."""
    fens = [f for f, _ in targets]
    out = eval_scores(theta, fens, nodes, workers)
    tot, agree = 0.0, 0
    for (s, m), (f, t) in zip(out, targets):
        tot += (sig(s) - sig(t)) ** 2
    # agreement vs deep bestmoves computed once per dataset in main loop
    return tot / len(targets), [m for _, m in out]


def main():
    global PARAMS
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=80)
    ap.add_argument('--positions', type=int, default=300)
    ap.add_argument('--shallow', type=int, default=5000,
                    help='prediction depth (gate-relevant: 5k/10k)')
    ap.add_argument('--deep-nodes', type=int, default=100000,
                    help='target depth (much deeper than shallow)')
    ap.add_argument('--deep-labels', default=os.path.join('logs', 'deep_labels.json'),
                    help='cached deep targets; generated if missing')
    ap.add_argument('--align', default='',
                    help='use our own deep labels (e.g. data/nnue/align20k.npz, 1M nodes) as targets; skips regeneration')
    ap.add_argument('--align-n', type=int, default=300)
    ap.add_argument('--jsonl', default='',
                    help='fresh deep labels {fen, deep} (e.g. logs/deep_fresh.jsonl @500k, current champ); deepest+freshest targets')
    ap.add_argument('--batch', type=int, default=0,
                    help='resample N fresh positions per iter from the pool (0 = fixed set; anti-overfit)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--params', default='')
    ap.add_argument('--state', default=os.path.join('logs', 'sloss.json'))
    ap.add_argument('--log', default=os.path.join('logs', 'sloss.jsonl'))
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--ascale', type=float, default=1.0,
                    help='global step multiplier (R2 refinement: 0.3)')
    ap.add_argument('--force-mode', type=int, default=-1,
                    help='override pinned LMRMode for this run (e.g. 3 = curve)')
    ap.add_argument('--lamb', type=float, default=0.0,
                    help='outcome-blend weight (lambda): target WDL = '
                         'lamb*outcome + (1-lamb)*sigmoid(deep); rows without '
                         'outcome use pure search. Blended back through logit.')
    a = ap.parse_args()

    names = [p[0] for p in PARAMS]
    if a.params:
        want = set(x.strip() for x in a.params.split(','))
        names = [n for n in names if n in want]
    pmap = {p[0]: p for p in PARAMS}
    if a.force_mode >= 0:
        pmap['LMRMode'] = ('LMRMode', a.force_mode, a.force_mode,
                           a.force_mode, 0.0, 0.0)
        if 'LMRMode' not in names:
            names.append('LMRMode')
    if os.path.exists(a.state):
        st = json.load(open(a.state))
        theta = st['theta']
        k0 = st['iter']
    else:
        theta = {p[0]: float(p[1]) for p in PARAMS}
        k0 = 0
    if a.force_mode >= 0:
        theta['LMRMode'] = float(a.force_mode)
    print('building dataset...', flush=True)
    if a.jsonl:
        ds = []
        for line in open(a.jsonl):
            d = json.loads(line)
            w = 1.0 / (1.0 + math.exp(-float(d['deep']) / 400.0))
            if a.lamb > 0 and 'outcome' in d and d['outcome'] is not None:
                w = a.lamb * float(d['outcome']) + (1.0 - a.lamb) * w
                w = min(1.0 - 1e-6, max(1e-6, w))
                deep_eff = 400.0 * math.log(w / (1.0 - w))
            else:
                deep_eff = float(d['deep'])
            ds.append((d['fen'], deep_eff))
        rng = random.Random(a.seed)
        rng.shuffle(ds)
        print('fresh-deep dataset: %d positions from %s' % (len(ds), a.jsonl), flush=True)
        targets = ds
        deep_moves = [None] * len(targets)
        base, _ = eval_loss({k: theta[k] for k in names}, targets,
                            a.shallow, a.workers)
        print('baseline loss: %.6f' % base, flush=True)
    elif a.align:
        ds = load_align(a.align, a.align_n, a.seed)
        print('align dataset: %d positions from %s' % (len(ds), a.align), flush=True)
        targets = ds
        deep_moves = [None] * len(targets)
        base, _ = eval_loss({k: theta[k] for k in names}, targets,
                            a.shallow, a.workers)
        print('baseline loss: %.6f' % base, flush=True)
    else:
        ds = build_dataset(a.positions, a.seed)
        print('dataset: %d positions' % len(ds), flush=True)
        # Deep targets: much-deeper search with default params (the "teacher").
        # Generated once, cached; shallow (5k/10k) learns to align with it.
        # (Only when NOT using our own 1M labels via --align.)
        if os.path.exists(a.deep_labels):
            deep = json.load(open(a.deep_labels))
            print('loaded %d deep labels' % len(deep['scores']), flush=True)
        else:
            print('deep-labeling %d positions @%d nodes...' % (len(ds), a.deep_nodes), flush=True)
            deep_out = eval_scores({}, [f for f, _ in ds], a.deep_nodes, a.workers)
            deep = {'scores': [s for s, _ in deep_out],
                    'moves': [m for _, m in deep_out],
                    'nodes': a.deep_nodes,
                    'teacher_hint': [t for _, t in ds]}
            json.dump(deep, open(a.deep_labels, 'w'))
            print('deep labels saved', flush=True)
        targets = [(f, s) for (f, _), s in zip(ds, deep['scores'])]
        deep_moves = deep.get('moves', [None] * len(targets))
        base, base_moves = eval_loss({k: theta[k] for k in names}, targets,
                                     a.shallow, a.workers)
        if deep_moves[0] is None:
            base_agree = -1.0
        else:
            base_agree = sum(1 for m, dm in zip(base_moves, deep_moves) if m == dm) / len(targets)
        print('baseline loss: %.6f agree: %.3f' % (base, base_agree), flush=True)

    rng = random.Random(999 + k0)
    logf = open(a.log, 'a')
    alpha, gamma, A = 0.602, 0.101, 10.0
    pool = list(targets)
    for k in range(k0, k0 + a.iters):
        if a.batch and len(pool) >= a.batch:
            targets = rng.sample(pool, a.batch)
        ck = 1.0 / ((k + 1) ** gamma)
        ak = 1.0 / ((k + 1 + A) ** alpha)
        plus = [1 if rng.random() < 0.5 else -1 for _ in names]
        tp = {n: theta[n] + d * ck * pmap[n][4] for n, d in zip(names, plus)}
        tm = {n: theta[n] - d * ck * pmap[n][4] for n, d in zip(names, plus)}
        for t in (tp, tm):
            for n in names:
                t[n] = min(pmap[n][3], max(pmap[n][2], t[n]))
        lp, mp_moves = eval_loss(tp, targets, a.shallow, a.workers)
        lm, mm_moves = eval_loss(tm, targets, a.shallow, a.workers)
        if deep_moves[0] is None:
            agree_p = agree_m = -1.0
        else:
            agree_p = sum(1 for m, dm in zip(mp_moves, deep_moves) if m == dm) / len(targets)
            agree_m = sum(1 for m, dm in zip(mm_moves, deep_moves) if m == dm) / len(targets)
        row = {'iter': k, 'lp': round(lp, 6), 'lm': round(lm, 6),
               'agree_p': round(agree_p, 4), 'agree_m': round(agree_m, 4)}
        for j, nm in enumerate(names):
            if pmap[nm][4] == 0:
                continue  # pinned param (e.g. mode selector): set, not tuned
            g = (lp - lm) / (2 * ck * pmap[nm][4] * plus[j])
            step = ak * pmap[nm][5] * a.ascale * g
            theta[nm] = min(pmap[nm][3], max(pmap[nm][2], theta[nm] - step))
            row[nm] = round(theta[nm], 2)
        row['theta'] = {n: round(theta[n], 2) for n in names}
        logf.write(json.dumps(row) + '\n')
        logf.flush()
        json.dump({'iter': k + 1, 'theta': theta}, open(a.state, 'w'))
        print('iter %d: loss+ %.5f (agr %.3f) loss- %.5f (agr %.3f) base %.5f theta=%s' % (
            k, lp, agree_p, lm, agree_m, base, row['theta']), flush=True)
    logf.close()


if __name__ == '__main__':
    main()
