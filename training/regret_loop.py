#!/usr/bin/env python3
"""Autonomous regret loop (user-authorized): play champ vs Moonbird, mine
losses, deep-label with graft teacher, retrain warm-started with regret
upweighting + base-data replay, gate candidate vs champ, adopt iff better.
Repeats until deadline. Research basis: PER (Schaul et al. 2016, sample by
error), reducible-loss prioritization (NeurIPS'23: chase learnable error,
not noise), NNUE quiet-data practice (arXiv 2412.17948).
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time

import numpy as np

EXE = r'C:\gitProjects\mainwt\target\release\titanium-cli.exe'
MOON = r'C:\gitProjects\titanium-ataxx\scripts\moonbird\Moonbird-1.1.0-windows-amd64.exe'
GRAFT = r'C:\gitProjects\titanium-ataxx\scripts\bench\graft_label.exe'
HERE = r'C:\gitProjects\titanium-ataxx'


def run(cmd, logf, cwd=HERE):
    logf.write('\n### %s\n' % ' '.join(cmd))
    logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=cwd)
    for line in p.stdout:
        logf.write(line)
        logf.flush()
    p.wait()
    return p.returncode


def status(msg, sf):
    line = '%s %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    sf.write(line + '\n')
    sf.flush()


def shard(lines, n, prefix):
    paths = ['%s_%d.txt' % (prefix, i) for i in range(n)]
    fhs = [open(p, 'w') for p in paths]
    for i, l in enumerate(lines):
        fhs[i % n].write(l if l.endswith('\n') else l + '\n')
    for f in fhs:
        f.close()
    return paths


def label_shards(paths, tag, workdir, nodes, logf):
    import glob
    for stale in glob.glob(os.path.join(workdir, 's_%s_*.txt' % tag)):
        try:
            os.remove(stale)
        except OSError:
            pass
    outs = [os.path.join(workdir, 's_%s_%d.txt' % (tag, i)) for i in range(len(paths))]
    procs = []
    for inp, out in zip(paths, outs):
        f_in = open(inp, 'rb')
        f_out = open(out, 'w')
        procs.append((subprocess.Popen(
            [GRAFT, 'label', '--nodes', str(nodes)],
            stdin=f_in, stdout=f_out, stderr=subprocess.DEVNULL), f_in, f_out))
    for p, f_in, f_out in procs:
        p.wait()
        f_in.close()
        f_out.close()
    return outs


pat_score = re.compile(r'^(-?\d+) (\d+) (\d+)$')


def read_scores(paths):
    out = []
    for p in paths:
        raw = open(p, 'rb').read()
        txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') \
            else raw.decode('utf-8', errors='replace')
        for line in txt.splitlines():
            m = pat_score.match(line.strip())
            if m:
                out.append(int(m.group(1)))
    return out


def sprt_llr(w, l, d, elo0=0.0, elo1=20.0):
    # Trinomial SPRT (cutechess-style): H0 = elo<=elo0 (reject), H1 = elo>=elo1
    # (promote). Draw prob = observed rate; draws carry no LLR either way.
    import math
    n = w + l + d
    if n == 0:
        return 0.0
    delta = min(0.98, max(0.001, d / n))
    s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    W0 = max(1e-9, s0 - delta / 2)
    L0 = max(1e-9, 1.0 - s0 - delta / 2)
    W1 = max(1e-9, s1 - delta / 2)
    L1 = max(1e-9, 1.0 - s1 - delta / 2)
    return w * math.log(W1 / W0) + l * math.log(L1 / L0)


GATE_BOOKS = ['data/nnue/book100.txt', 'data/nnue/book100b.txt',
              'data/nnue/book100c.txt']


def parse_book_total(path):
    try:
        txt = open(path, errors='replace').read()
    except OSError:
        return None
    m = re.search(r'total:\s*titan\s+(\d+)\s*-\s*moon\s+(\d+)\s+draws\s+(\d+)',
                  txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_gate_total(path):
    try:
        txt = open(path, errors='replace').read()
    except OSError:
        return None
    m = re.search(r'total:\s*titanium\s+(\d+)\s*-\s*[^\d]*(\d+),\s*draws\s+(\d+)',
                  txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deadline-min', type=float, default=115.0)
    ap.add_argument('--champ', default=r'C:\gitProjects\mainwt\data\nnue\tupS53r0.tup')
    ap.add_argument('--books', default='data/nnue/book100.txt,data/nnue/book100b.txt',
                    help='comma-separated book rotation for mining games AND gates')
    ap.add_argument('--base', default='data/nnue/cache400kR_graft.npz')
    ap.add_argument('--workdir', default='data/nnue/regretloop')
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--label-nodes', type=int, default=400000)
    ap.add_argument('--top', type=int, default=600)
    ap.add_argument('--min-drop', type=float, default=50.0)
    ap.add_argument('--base-rows', type=int, default=10000)
    a = ap.parse_args()
    t_end = time.time() + a.deadline_min * 60
    try:  # keep the box usable: deprioritize self, children inherit it
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    rnd = 0
    os.makedirs(a.workdir, exist_ok=True)
    slog = open('logs/regret_loop_status.txt', 'a')
    logf = open('logs/regret_loop.txt', 'a')
    champ = a.champ
    status('LOOP START champ=%s deadline=%.0fmin' % (champ, a.deadline_min), slog)
    base = np.load(a.base)
    nb = int(base['occ0'].shape[0])
    books = a.books.split(',')
    while time.time() + 25 * 60 < t_end:
        rnd += 1
        rng = random.Random(1000 + rnd)
        status('ROUND %d: games (champ vs Moonbird %dg/%dw)' % (rnd, a.games, a.workers), slog)
        glog = 'logs/regret_r%d_games.txt' % rnd
        mlog = 'logs/regret_r%d_moves.txt' % rnd
        rc = run([sys.executable, 'training/book_gate.py', '--book', books[rnd % 2],
                  '--games', str(a.games), '--nodes', '5000',
                  '--titanium', EXE, '--targs', 'serve', '--tup', champ,
                  '--opp', MOON, '--opp-nodes', '5000',
                  '--workers', str(a.workers), '--seed', str(rnd),
                  '--out', glog, '--moves-out', mlog], logf)
        if rc != 0:
            status('ROUND %d: games failed rc=%d, next round' % (rnd, rc), slog)
            continue
        status('ROUND %d: mining losses' % rnd, slog)
        rc = run([sys.executable, 'training/mine_blunders.py', '--log', mlog,
                  '--out', a.workdir, '--max-games', '0'], logf)
        bef = [l.strip() for l in open(os.path.join(a.workdir, 'befores.txt'))
               if l.strip()]
        aft = [l.strip() for l in open(os.path.join(a.workdir, 'afters.txt'))
               if l.strip()]
        meta = [l.strip() for l in open(os.path.join(a.workdir, 'meta.txt'))
                if l.strip()]
        n = min(len(bef), len(aft), len(meta))
        status('ROUND %d: mined %d positions' % (rnd, n), slog)
        if n < 20:
            status('ROUND %d: too few, more games next round' % rnd, slog)
            continue
        if n > 400:  # cap labeling budget (box stays usable), uniform subsample
            idx = sorted(rng.sample(range(n), 400))
            bef, aft, meta = [bef[i] for i in idx], [aft[i] for i in idx], \
                [meta[i] for i in idx]
            n = 400
        bp = shard(bef, 2, os.path.join(a.workdir, 'sh_b'))
        apaths = shard(aft, 2, os.path.join(a.workdir, 'sh_a'))
        status('ROUND %d: deep-labeling %d x2 @%dk (%dw graft)' % (
            rnd, n, a.label_nodes // 1000, len(bp)), slog)
        sb = label_shards(bp, 'b', a.workdir, a.label_nodes, logf)
        sa = label_shards(apaths, 'a', a.workdir, a.label_nodes, logf)
        sc_b, sc_a = read_scores(sb), read_scores(sa)
        if len(sc_b) != n or len(sc_a) != n:
            status('ROUND %d: label count mismatch %d/%d vs %d, skip' % (
                rnd, len(sc_b), len(sc_a), n), slog)
            continue
        scored = [(sc_b[i] + sc_a[i], i) for i in range(n)
                  if sc_b[i] + sc_a[i] >= a.min_drop]
        scored.sort(reverse=True)
        pick = [i for _, i in scored[:a.top]]
        status('ROUND %d: drops>=%.0f: %d, training top %d' % (
            rnd, a.min_drop, len(scored), len(pick)), slog)
        if len(pick) < 50:
            status('ROUND %d: too few regret rows, next round' % rnd, slog)
            continue
        occ0, occ1, stm, teach, ws = [], [], [], [], []
        for i in pick:
            o0 = o1 = 0
            for sq, ch in enumerate(bef[i].split()[0]):
                if ch == 'x':
                    o0 |= 1 << sq
                elif ch == 'o':
                    o1 |= 1 << sq
            occ0.append(o0)
            occ1.append(o1)
            stm.append(0 if meta[i].split()[3] == 'b' else 1)
            teach.append(max(-1500, min(1500, sc_b[i])))
            ws.append(min((sc_b[i] + sc_a[i]) / 100.0, 10.0))
        bi = rng.sample(range(nb), min(a.base_rows, nb))  # replay vs forgetting
        train = {
            'occ0': np.concatenate([np.array(occ0, np.uint64), base['occ0'][bi]]),
            'occ1': np.concatenate([np.array(occ1, np.uint64), base['occ1'][bi]]),
            'blk': np.concatenate([np.zeros(len(pick), np.uint64), base['blk'][bi]]),
            'stm': np.concatenate([np.array(stm, np.uint8), base['stm'][bi]]),
            'teacher': np.concatenate(
                [np.array(teach, np.float32), base['teacher'][bi]]),
            'wsum': np.concatenate(
                [np.array(ws, np.float64), np.ones(len(bi), np.float64)]),
            'out': np.concatenate(
                [np.zeros(len(pick), np.float64), base['out'][bi]]),
            'sc': np.concatenate(
                [np.array(teach, np.float64), base['sc'][bi]]),
        }
        tnpz = 'data/nnue/regretloop_r%d.npz' % rnd
        np.savez_compressed(tnpz, **train)
        cand = 'data/nnue/regretloop_r%d.tup' % rnd
        status('ROUND %d: training warm-from-champ lr 0.01' % rnd, slog)
        rc = run([sys.executable, 'training/train_tuple.py', '--cache', tnpz,
                  '--tcol', 'teacher',
                  '--init', champ, '--lr', '0.01', '--epochs', '12',
                  '--out', cand, '--seed', str(rnd)], logf)
        if rc != 0 or not os.path.exists(cand):
            status('ROUND %d: train failed, next round' % rnd, slog)
            continue
        # Two-sided SPRT gate on BOOK games (startpos fixed-node replays 2
        # lines; a 100-0 there can be pure line-flip artifact). Chunks of 100
        # games across 3 disjoint books; LLR bounds +-2.94 (alpha=beta=0.05),
        # H0: elo<=0 (reject) vs H1: elo>=20 (promote). Cap 400 -> undecided:
        # champ holds but candidate + data are BANKED, never discarded.
        W = L = D = 0
        verdict = None
        for ck in range(4):
            if verdict:
                break
            gl = 'logs/regret_r%d_gate%d.txt' % (rnd, ck)
            rc = run([sys.executable, 'training/book_gate.py',
                      '--book', GATE_BOOKS[(rnd + ck) % 3],
                      '--games', '100', '--nodes', '5000',
                      '--skip', str((ck * 50) % 100), '--take', '50',
                      '--titanium', EXE, '--targs', 'serve', '--tup', cand,
                      '--opp', EXE, '--oargs', 'serve', '--opp-tup', champ,
                      '--opp-nodes', '5000', '--workers', '2',
                      '--seed', str(7000 + rnd * 10 + ck), '--out', gl], logf)
            tot = parse_book_total(gl)
            if not tot:
                status('ROUND %d: gate chunk %d failed to parse' % (rnd, ck), slog)
                continue
            W, L, D = W + tot[0], L + tot[1], D + tot[2]
            llr = sprt_llr(W, L, D)
            status('ROUND %d: gate chunk %d %s pooled %d-%d-%d LLR %+.2f' % (
                rnd, ck, tot, W, L, D, llr), slog)
            if llr >= 2.94:
                verdict = 'pass'
            elif llr <= -2.94:
                verdict = 'fail'
        if verdict == 'pass':
            champ = os.path.abspath(cand)
            status('ROUND %d: ADOPTED %s (SPRT pass %d-%d-%d)' % (
                rnd, cand, W, L, D), slog)
        elif verdict == 'fail':
            status('ROUND %d: REJECTED (SPRT fail %d-%d-%d, banked %s)' % (
                rnd, W, L, D, cand), slog)
        else:
            status('ROUND %d: UNDECIDED at cap %d-%d-%d (champ holds, banked %s)' % (
                rnd, W, L, D, cand), slog)
        # Moonbird tracker: reuse the mining games (no separate matches needed).
        # Champ-vs-champ gates keep promotion resolution; vs Moonbird (~1%
        # scores) only the multi-round TREND is meaningful, and the mining
        # games already provide it for free every round.
        mt = parse_book_total(glog)
        status('ROUND %d: MOON-TRACK %s champ=%s' % (
            rnd, mt, os.path.basename(champ)), slog)
        with open('logs/moon_track.txt', 'a') as f:
            f.write('round %d champ=%s moon-track %s\n' % (
                rnd, os.path.basename(champ), mt))
    status('LOOP END champ=%s' % champ, slog)


if __name__ == '__main__':
    main()
