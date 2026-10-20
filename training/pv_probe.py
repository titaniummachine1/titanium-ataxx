import argparse
import os
import subprocess
import sys
import time


class Engine:
    def __init__(self, exe, args):
        self.p = subprocess.Popen([exe] + args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        self.send('uai')
        self.wait('uaiok')
        self.send('isready')
        self.wait('readyok')

    def send(self, line):
        self.p.stdin.write(line + '\n')
        self.p.stdin.flush()

    def wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('engine died waiting for ' + token)
            if line.strip().startswith(token):
                return line.strip()

    def best(self, fen, nodes):
        self.send('uainewgame')
        self.send('position fen ' + fen)
        self.send('go nodes %d' % nodes)
        return self.wait('bestmove').split()[1]

    def close(self):
        try:
            self.send('quit')
        except Exception:
            pass
        try:
            self.p.kill()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--tuple-native', action='store_true', default=True)
    ap.add_argument('--fens', default='data/nnue/golden_src/lab_00.txt')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--shallow', type=int, default=5000)
    ap.add_argument('--deep', type=int, default=200000)
    ap.add_argument('--out', default='logs/pv_probe.csv')
    a = ap.parse_args()

    fens = []
    with open(a.fens) as f:
        for line in f:
            line = line.strip()
            if line:
                fens.append(line)
            if len(fens) >= a.limit:
                break
    args = ['serve', '--tuple', a.net]
    if a.tuple_native:
        args.append('--tuple-native')
    e = Engine(a.exe, args)
    agree = 0
    t0 = time.time()
    rows = []
    for fen in fens:
        m_s = e.best(fen, a.shallow)
        m_d = e.best(fen, a.deep)
        ok = (m_s == m_d)
        agree += int(ok)
        rows.append('%s\t%s\t%s\t%d' % (fen, m_s, m_d, int(ok)))
    e.close()
    with open(a.out, 'w') as f:
        f.write('fen\tshallow\tdeep\tagree\n')
        f.write('\n'.join(rows) + '\n')
    n = len(fens)
    print('positions %d  agreement %.3f (%d/%d)  %.0fs -> %s'
          % (n, agree / max(n, 1), agree, n, time.time() - t0, a.out), flush=True)


if __name__ == '__main__':
    main()
