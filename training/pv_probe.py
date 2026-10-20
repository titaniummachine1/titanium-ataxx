import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

RE_BEST = re.compile(r'bestmove\s+(\S+)\s+score\s+(-?\d+)')


def run_one(exe, board, nodes, tup, native):
    cmd = [exe, 'bestmove', board, '--nodes', str(nodes), '--tuple', tup]
    if native:
        cmd.append('--tuple-native')
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return None, None
    m = RE_BEST.search(out)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--tuple-native', action='store_true', default=True)
    ap.add_argument('--fens', default='data/nnue/golden_src/lab_00.txt')
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--shallow', type=int, default=5000)
    ap.add_argument('--deep', type=int, default=200000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='logs/pv_probe.csv')
    a = ap.parse_args()

    boards = []
    for path in [a.fens] + a.extra:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        boards.append(line)
        except FileNotFoundError:
            pass
    boards = boards[:a.limit]
    print('positions %d, shallow %d deep %d workers %d' % (len(boards), a.shallow, a.deep, a.workers), flush=True)

    def job(board):
        sm, ss = run_one(a.exe, board, a.shallow, a.net, a.tuple_native)
        dm, ds = run_one(a.exe, board, a.deep, a.net, a.tuple_native)
        return board, sm, ss, dm, ds

    t0 = time.time()
    rows = []
    agree = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for board, sm, ss, dm, ds in ex.map(job, boards):
            ok = sm is not None and dm is not None and sm == dm
            agree += int(ok)
            rows.append((board, sm, ss, dm, ds, int(ok)))
    n = len(rows)
    with open(a.out, 'w') as f:
        f.write('board\tshallow_move\tshallow_score\tdeep_move\tdeep_score\tagree\n')
        for r in rows:
            f.write('%s\t%s\t%s\t%s\t%s\t%d\n' % r)
    print('agreement %.3f (%d/%d) in %.0fs -> %s'
          % (agree / max(n, 1), agree, n, time.time() - t0, a.out), flush=True)


if __name__ == '__main__':
    main()
