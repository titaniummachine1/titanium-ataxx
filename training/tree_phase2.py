#!/usr/bin/env python3
"""Tree-distill phase 2 (autonomous): wait for tree_r2 labeling to finish,
build backed-up npz, train warm from current champ, SPRT-gate, record.
Fires itself when input is ready; no babysitting, no mixing with other work.
"""
import glob
import os
import subprocess
import sys
import time

HERE = r'C:\gitProjects\titanium-ataxx'


def run(cmd, logf):
    logf.write('\n### %s\n' % ' '.join(cmd))
    logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=HERE)
    for line in p.stdout:
        logf.write(line)
        logf.flush()
    p.wait()
    return p.returncode


def stable(path, wait_s=300, stable_rounds=3):
    """True when file size stops changing (labeling done)."""
    last = -1
    same = 0
    waited = 0
    while waited < 5 * 3600:
        try:
            cur = os.path.getsize(path)
        except OSError:
            cur = -1
        if cur == last and cur > 0:
            same += 1
            if same >= stable_rounds:
                return True
        else:
            same = 0
        last = cur
        time.sleep(wait_s)
        waited += wait_s
    return False


def main():
    os.makedirs('logs', exist_ok=True)
    sf = open('logs/tree_phase2_status.txt', 'a')
    logf = open('logs/tree_phase2.txt', 'a')

    def status(m):
        line = '%s %s' % (time.strftime('%H:%M:%S'), m)
        print(line, flush=True)
        sf.write(line + '\n')
        sf.flush()

    status('PHASE2 armed: waiting for tree_r2 labeling')
    if not stable('data/nnue/tree_r2.txt'):
        status('PHASE2: timeout waiting for labels; exiting')
        return
    status('PHASE2: labels stable, building npz')
    if run([sys.executable, 'training/build_tree_npz.py',
            'data/nnue/tree_r2.txt', '--out', 'data/nnue/tree_r2.npz'], logf) != 0:
        status('PHASE2: build failed')
        return
    status('PHASE2: training warm from champ')
    if run([sys.executable, 'training/train_tuple.py', '--cache', 'data/nnue/tree_r2.npz',
            '--rows', '2000000', '--tcol', 'teacher', '--init', CHAMP(), '--lr', '0.01', '--epochs', '6',
            '--out', 'data/nnue/upS54tree2.tup', '--seed', '7'], logf) != 0:
        status('PHASE2: train failed')
        return
    status('PHASE2: gating (book SPRT, cap 400)')
    sys.path.insert(0, 'training')
    from regret_loop import run as _r, parse_book_total, sprt_llr, EXE, GATE_BOOKS
    W = L = D = 0
    verdict = None
    for ck in range(4):
        if verdict:
            break
        gl = 'logs/tree2_gate%d.txt' % ck
        _r([sys.executable, 'training/book_gate.py', '--book', GATE_BOOKS[ck % 3],
            '--games', '100', '--nodes', '5000', '--skip', str((ck * 50) % 100),
            '--take', '50', '--titanium', EXE, '--targs', 'serve',
            '--tup', os.path.join(HERE, 'data', 'nnue', 'upS54tree2.tup'),
            '--opp', EXE, '--oargs', 'serve', '--opp-tup', CHAMP(),
            '--opp-nodes', '5000', '--workers', '2',
            '--seed', str(46000 + ck), '--out', gl], logf)
        tot = parse_book_total(gl)
        if tot:
            W, L, D = W + tot[0], L + tot[1], D + tot[2]
        llr = sprt_llr(W, L, D)
        status('gate chunk %d pooled %d-%d-%d LLR %+.2f' % (ck, W, L, D, llr))
        if llr >= 2.94:
            verdict = 'pass'
        elif llr <= -2.94:
            verdict = 'fail'
    status('PHASE2 VERDICT: %s %d-%d-%d (no auto-merge; needs human promote order)' % (
        verdict or 'undecided', W, L, D))


def CHAMP():
    return r'C:\gitProjects\mainwt\data\nnue\tupS54align.tup'


if __name__ == '__main__':
    main()
