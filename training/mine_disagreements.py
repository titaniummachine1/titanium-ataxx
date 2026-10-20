import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ataxx_move as M

RE_BEST = re.compile(r'bestmove\s+(\S+)\s+score\s+(-?\d+)')


def bestmove(exe, board, nodes, tup, native):
    cmd = [exe, 'bestmove', board, '--nodes', str(nodes), '--tuple', tup]
    if native:
        cmd.append('--tuple-native')
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = RE_BEST.search(out)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def label_boards(exe, boards, nodes, tup, native):
    cmd = [exe, 'label', '--nodes', str(nodes), '--tuple', tup]
    if native:
        cmd.append('--tuple-native')
    inp = ('\n'.join(boards) + '\n').encode()
    out = subprocess.run(cmd, input=inp, capture_output=True).stdout.decode('utf-8', 'replace')
    scores = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 1:
            try:
                scores.append(float(parts[0]))
            except ValueError:
                pass
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--rows', type=int, default=2000)
    ap.add_argument('--shallow', type=int, default=5000)
    ap.add_argument('--deep', type=int, default=200000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--out', default='data/nnue/disagree.npz')
    ap.add_argument('--table', default='logs/disagree.tsv')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    z = np.load(a.cache)
    n = len(z['occ0'])
    rng = np.random.RandomState(a.seed)
    sel = np.sort(rng.choice(n, min(a.rows, n), replace=False))
    wsum = z['wsum'].astype(np.float64) if 'wsum' in z else np.ones(n)
    boards = [M.board_str(int(z['occ0'][i]), int(z['occ1'][i]), int(z['blk'][i]), int(z['stm'][i]))
              for i in sel]
    print('positions %d' % len(boards), flush=True)

    def job(bd):
        sm, ss = bestmove(a.exe, bd, a.shallow, a.net, True)
        dm, ds = bestmove(a.exe, bd, a.deep, a.net, True)
        return bd, sm, ss, dm, ds

    t0 = time.time()
    dis = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for bd, sm, ss, dm, ds in ex.map(job, boards):
            if sm and dm and sm != dm:
                dis.append((bd, sm, ss, dm, ds))
    print('disagreements %d/%d (%.1f%%) in %.0fs' % (
        len(dis), len(boards), 100.0 * len(dis) / max(len(boards), 1), time.time() - t0), flush=True)

    parents = [d[0] for d in dis]
    p_scores = label_boards(a.exe, parents, a.deep, a.net, True) if parents else []
    print('parent labels %d' % len(p_scores), flush=True)

    child_boards = []
    child_meta = []
    for i, (bd, sm, ss, dm, ds) in enumerate(dis):
        parsed = M.parse_board(bd)
        if parsed is None:
            continue
        for mv, kind in ((sm, 'shallow'), (dm, 'deep')):
            child = M.apply_move(*parsed, mv)
            if child is None:
                continue
            child_boards.append(M.board_str(*child))
            child_meta.append((i, kind, child))
    c_scores = label_boards(a.exe, child_boards, a.deep, a.net, True) if child_boards else []
    print('child labels %d' % len(c_scores), flush=True)

    occ0 = []
    occ1 = []
    blk = []
    stm = []
    teach = []
    out = []
    sc = []
    ws = []

    def push(parsed, teacher, src_i):
        o0, o1, bl, st = parsed
        occ0.append(o0)
        occ1.append(o1)
        blk.append(bl)
        stm.append(st)
        teach.append(teacher)
        out.append(float(z['out'][sel[src_i]]) if 'out' in z else 0.5)
        sc.append(float(z['sc'][sel[src_i]]) if 'sc' in z else 0.0)
        ws.append(float(wsum[sel[src_i]]))

    for i, (bd, sm, ss, dm, ds) in enumerate(dis):
        parsed = M.parse_board(bd)
        if parsed is None or i >= len(p_scores):
            continue
        push(parsed, p_scores[i], i)

    rows = []
    for (i, kind, child), s in zip(child_meta, c_scores):
        push(child, s, i)
        rows.append((dis[i][0], dis[i][1], dis[i][2], dis[i][3], dis[i][4], kind, s))

    if rows:
        with open(a.table, 'w', encoding='utf-8') as f:
            f.write('board\tshallow\tshallow_score\tdeep\tdeep_score\tkind\tchild_score\n')
            for r in rows:
                f.write('%s\t%s\t%s\t%s\t%s\t%s\t%.1f\n' % r)

    print('training rows %d (parents + children)' % len(occ0), flush=True)
    np.savez_compressed(a.out,
                        occ0=np.array(occ0, dtype=np.uint64),
                        occ1=np.array(occ1, dtype=np.uint64),
                        blk=np.array(blk, dtype=np.uint64),
                        stm=np.array(stm, dtype=np.uint8),
                        teacher=np.array(teach, dtype=np.float32),
                        out=np.array(out, dtype=np.float32),
                        sc=np.array(sc, dtype=np.float32),
                        wsum=np.array(ws, dtype=np.float32))
    print('wrote %s %d rows' % (a.out, len(occ0)), flush=True)


if __name__ == '__main__':
    main()
