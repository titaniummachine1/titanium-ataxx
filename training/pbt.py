#!/usr/bin/env python3
"""PBT for tuple eval: population trains briefly, bottom copies top
(exploit via --init warm start) with perturbed hyperparams (explore),
repeat. Fitness = val (cheap, every generation); gates decide at the end.

Usage: python training/pbt.py --gens 3 --epochs 8 --members 4
         --cache data/nnue/cache400kR_graft.npz --outdir data/nnue/pbt
Each member: --dirs 0,1 --base s13 --tcol teacher (fixed recipe; only
hyperparams evolve: lr, conv-lr, noise-w, smooth, blend, ow, seed).
Gen 0 inits from --start (default: best known = tupS47cyc.tup).
"""
import argparse
import os
import random
import re
import subprocess
import sys

VAL_RE = re.compile(r'flywheel: best ep \d+ val ([\d.]+) -> (\S+)')


def run_member(cache, outdir, name, hypers, epochs, init, rows=()):
    out = os.path.join(outdir, '%s.tup' % name)
    log = os.path.join(outdir, '%s.log' % name)
    cmd = [sys.executable, 'training/train_tuple.py', '--cache', cache,
           '--dirs', '0,1', '--base', 's13', '--tcol', 'teacher',
           '--blend', str(hypers['blend']),
           '--outcome-weight', str(hypers['ow']),
           '--lr', str(hypers['lr']), '--conv-lr', str(hypers['conv_lr']),
           '--noise-w', str(hypers['noise']),
           '--smooth', str(hypers['smooth']),
           '--seed', str(hypers['seed']),
           '--plateau', '0', '--epochs', str(epochs),
           '--ckpt', os.path.join(outdir, name + '_ep{ep}.tup'),
           '--out', out] + list(rows)
    if init:
        cmd += ['--init', init]
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    txt = open(log, errors='ignore').read()
    m = VAL_RE.search(txt)
    if not m:
        return None, None
    return float(m.group(1)), out


def perturb(h, rng):
    g = dict(h)
    g['lr'] = round(h['lr'] * 10 ** rng.uniform(-0.3, 0.3), 5)
    g['conv_lr'] = round(g['lr'] / 10.0, 5)
    g['noise'] = round(max(0.0, h['noise'] + rng.uniform(-5, 5)), 2)
    g['smooth'] = round(min(0.3, max(0.0, h['smooth'] + rng.uniform(-0.05, 0.05))), 3)
    g['blend'] = round(min(1.0, max(0.8, h['blend'] + rng.uniform(-0.05, 0.05))), 2)
    g['ow'] = rng.choice([0.0, 0.0, 0.1, 0.2])
    g['seed'] = rng.randrange(1 << 30)
    return g


def base_hypers(i):
    h = {'lr': 0.05, 'conv_lr': 0.005, 'noise': 0.0, 'smooth': 0.0,
         'blend': 0.9, 'ow': 0.0, 'seed': 7 + i}
    if i == 1:
        h.update(noise=10.0, smooth=0.05)
    elif i == 2:
        h.update(ow=0.1)
    elif i == 3:
        h.update(lr=0.02, smooth=0.1)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--members', type=int, default=4)
    ap.add_argument('--cache', default='data/nnue/cache400kR_graft.npz')
    ap.add_argument('--outdir', default='data/nnue/pbt')
    ap.add_argument('--start', default='data/nnue/tupS47cyc.tup')
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--rows', type=int, default=0,
                    help='passed through to train (0 = full cache)')
    a = ap.parse_args()
    a.rows = ['--rows', str(a.rows)] if a.rows else []
    os.makedirs(a.outdir, exist_ok=True)
    rng = random.Random(a.seed)
    pop = []
    for i in range(a.members):
        h = base_hypers(i)
        v, out = run_member(a.cache, a.outdir, 'g0m%d' % i, h, a.epochs,
                            a.start if os.path.exists(a.start) else '', a.rows)
        print('gen0 m%d val=%s hypers=%s' % (i, v, h), flush=True)
        pop.append({'hypers': h, 'val': v, 'tup': out})
    pop = [p for p in pop if p['val'] is not None]
    for g in range(1, a.gens + 1):
        pop.sort(key=lambda p: p['val'])
        nkeep = max(1, len(pop) // 2)
        keep, drop = pop[:nkeep], pop[nkeep:]
        print('gen%d keep %s' % (g - 1, [(round(p['val'], 1), p['tup']) for p in keep]), flush=True)
        nxt = []
        for p in keep:
            v, out = run_member(a.cache, a.outdir, 'g%dk%s' % (g, os.path.basename(p['tup'])[:9]),
                                p['hypers'], a.epochs, p['tup'], a.rows)
            print('gen%d keep val=%s <- %s' % (g, v, p['tup']), flush=True)
            nxt.append({'hypers': p['hypers'], 'val': v, 'tup': out})
        for j, _ in enumerate(drop):
            src = keep[j % len(keep)]
            h = perturb(src['hypers'], rng)
            v, out = run_member(a.cache, a.outdir, 'g%dx%d' % (g, j), h, a.epochs, src['tup'], a.rows)
            print('gen%d explore val=%s hypers=%s <- %s' % (g, v, h, src['tup']), flush=True)
            nxt.append({'hypers': h, 'val': v, 'tup': out})
        pop = [p for p in nxt if p['val'] is not None]
    pop.sort(key=lambda p: p['val'])
    print('FINAL:', [(round(p['val'], 1), p['tup'], p['hypers']) for p in pop], flush=True)


if __name__ == '__main__':
    main()
