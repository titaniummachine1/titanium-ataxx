import argparse
import concurrent.futures as cf
import re
import subprocess
import time

RE_BEST = re.compile(r'bestmove\s+(\S+)')


def best(exe, board, nodes, tup, native):
    cmd = [exe, 'bestmove', board, '--nodes', str(nodes), '--tuple', tup]
    if native:
        cmd.append('--tuple-native')
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = RE_BEST.search(out)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--fens', default='data/nnue/golden_src/lab_00.txt')
    ap.add_argument('--limit', type=int, default=60)
    ap.add_argument('--budgets', default='1000000,2000000,4000000')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='logs/depth_probe.tsv')
    a = ap.parse_args()

    budgets = [int(x) for x in a.budgets.split(',')]
    boards = []
    with open(a.fens) as f:
        for line in f:
            line = line.strip()
            if line:
                boards.append(line)
            if len(boards) >= a.limit:
                break
    print('positions %d budgets %s' % (len(boards), budgets), flush=True)

    def job(bd):
        return bd, [best(a.exe, bd, n, a.net, True) for n in budgets]

    t0 = time.time()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for bd, moves in ex.map(job, boards):
            rows.append((bd, moves))

    n = len(rows)
    print('positions %d, %.0fs' % (n, time.time() - t0), flush=True)
    for i in range(len(budgets) - 1):
        same = sum(1 for _, m in rows if m[i] is not None and m[i] == m[i + 1])
        print('  %d vs %d: agree %.3f (%d/%d)' % (budgets[i], budgets[i + 1], same / max(n, 1), same, n), flush=True)
    same_all = sum(1 for _, m in rows if m[0] is not None and len(set(m)) == 1)
    print('  all-three agree %.3f (%d/%d)' % (same_all / max(n, 1), same_all, n), flush=True)

    with open(a.out, 'w') as f:
        f.write('board\t' + '\t'.join('m%d' % b for b in budgets) + '\n')
        for bd, m in rows:
            f.write(bd + '\t' + '\t'.join(str(x) for x in m) + '\n')


if __name__ == '__main__':
    main()
