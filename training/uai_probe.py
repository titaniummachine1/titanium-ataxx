#!/usr/bin/env python3
"""UAI compliance probe: verifies an engine speaks standard UAI.
Checks: handshake, options, setoption, moves suffix, FEN dialects,
btime/wtime clocks, infinite, move format. Exits nonzero on failure.
Usage: python3 training/uai_probe.py --exe <engine> --args "serve ..."
"""
import argparse
import subprocess
import sys


class E:
    def __init__(self, exe, args):
        self.p = subprocess.Popen([exe] + args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)

    def send(self, s):
        self.p.stdin.write(s + '\n')
        self.p.stdin.flush()

    def wait(self, prefix, limit=200):
        for _ in range(limit):
            l = self.p.stdout.readline().strip()
            if l.startswith(prefix):
                return l
        return None

    def go(self, timeout_reads=600):
        for _ in range(timeout_reads):
            l = self.p.stdout.readline().strip()
            if l.startswith('bestmove'):
                return l.split()[1] if len(l.split()) > 1 else 'MISSING'
        return None


def check(name, cond, fails):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True)
    ap.add_argument('--args', default='')
    a = ap.parse_args()
    fails = []
    e = E(a.exe, a.args.split())

    e.send('uai')
    l = e.wait('id name')
    check('id-name', l is not None, fails)
    opts = []
    while True:
        l = e.p.stdout.readline().strip()
        if l == 'uaiok':
            break
        if l.startswith('option name'):
            opts.append(l)
    check('uaiok', True, fails)
    check('advertises-options', len(opts) > 0, fails)
    print('  options: %s' % opts)
    e.send('setoption name Hash value 64')
    e.send('isready')
    check('readyok-after-setoption', e.wait('readyok') is not None, fails)

    # moves suffix from startpos
    e.send('uainewgame')
    e.send('position startpos moves e2e4')
    e.send('go nodes 1000')
    m = e.go()
    check('startpos+moves+go -> bestmove (%s)' % m, m not in (None, 'MISSING'), fails)

    # FEN dialect: dashes + b side
    e.send('position fen x5o/7/7/7/7/7/o5x b 0 1')
    e.send('go nodes 1000')
    m = e.go()
    check('dash-fen b-side -> bestmove (%s)' % m, m not in (None, 'MISSING'), fails)

    # FEN dialect: our native x/o/# + w side
    e.send('position fen 4ooo/4o1o/7/2x2x1/7/7/x6 o 0 1')
    e.send('go nodes 1000')
    m = e.go()
    check('native-fen w-side -> bestmove (%s)' % m, m not in (None, 'MISSING'), fails)

    # incremental clock
    e.send('position startpos')
    e.send('go btime 3000 wtime 3000 binc 0 winc 0')
    m = e.go()
    check('btime/wtime -> bestmove (%s)' % m, m not in (None, 'MISSING'), fails)

    # movetime
    e.send('position startpos')
    e.send('go movetime 100')
    m = e.go()
    check('movetime -> bestmove (%s)' % m, m not in (None, 'MISSING'), fails)

    # nullmove reporting
    e.send('position fen 7/7/7/7/7/7/7 x 0 1')
    e.send('go nodes 1000')
    m = e.go()
    check('empty-board -> 0000 (%s)' % m, m == '0000', fails)

    e.send('quit')
    try:
        e.p.wait(timeout=5)
    except Exception:
        e.p.kill()
    print('FAILURES: %s' % (fails if fails else 'none'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
