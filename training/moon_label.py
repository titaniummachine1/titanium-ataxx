"""Moonbird teacher labels via UAI (persistent process, no per-row spawn).

UAI verified live: `position fen <49 .xo# + b|w>` accepted (print echoes
board), `go nodes N` streams `info ... score cp S` + `bestmove M`.
Mate scores come out as 32765 — capped to MATE_CAP on parse. One process
stays alive per chunk; respawn on failure. FEN dialect = ours (round-trip
verified on cache row 0).

Usage: python training/moon_label.py --in <board lines> --out <scores> --nodes 5000
"""
import argparse
import re
import subprocess
import sys
import time

EXE = 'scripts/moonbird/Moonbird-1.1.0-windows-amd64.exe'
MATE_CAP = 5000


def drain_banner(p, timeout=2.0):
    # banner is ~1KB with unicode art; read whatever is available without blocking.
    import os
    import time
    end = time.time() + timeout
    try:
        os.set_blocking(p.stdout.fileno(), False)
    except Exception:
        return
    try:
        while time.time() < end:
            try:
                chunk = p.stdout.read(65536)
            except Exception:
                break
            if not chunk:
                time.sleep(0.1)
    finally:
        try:
            os.set_blocking(p.stdout.fileno(), True)
        except Exception:
            pass


def spawn():
    p = subprocess.Popen([EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=False, bufsize=0)
    drain_banner(p)
    return p


def ask(p, board, nodes):
    p.stdin.write(('position fen %s\n' % board).encode())
    p.stdin.write(('go nodes %d\n' % nodes).encode())
    p.stdin.flush()
    sc = None
    buf = b''
    # Moonbird @5k on hard positions: ~2-10s/row (deep midgame lines).
    # readline loop is fine (bestmove-terminated); the earlier hang was the
    # 30s tool timeout on 2000 rows, not a protocol stall.
    while True:
        chunk = p.stdout.readline()
        if not chunk:
            raise RuntimeError('moonbird EOF')
        buf += chunk
        if b'bestmove' in chunk:
            break
    ms = re.findall(rb'score cp\s+(-?\d+)', buf)
    if ms:
        sc = max(-MATE_CAP, min(MATE_CAP, int(ms[-1])))
    assert sc is not None, 'no score: %r' % buf[-200:]
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--nodes', type=int, default=5000)
    a = ap.parse_args()
    fin = open(a.inp, 'rb') if a.inp else sys.stdin.buffer
    fout = open(a.out, 'w') if a.out else sys.stdout
    lines = [l.decode().strip() for l in fin if l.strip()]
    p = spawn()
    t0 = time.time()
    n = 0
    for i, board in enumerate(lines):
        try:
            sc = ask(p, board, a.nodes)
        except Exception as e:
            sys.stderr.write('moon_label row %d FAIL %s; respawn\n' % (i, e))
            sys.stderr.flush()
            try:
                p.kill()
            except Exception:
                pass
            p = spawn()
            sc = ask(p, board, a.nodes)
        fout.write('%d\n' % sc)
        n += 1
        if n % 500 == 0:
            sys.stderr.write('moon_label %d/%d (%.0fs)\n' % (n, len(lines), time.time() - t0))
            sys.stderr.flush()
    try:
        p.stdin.write(b'quit\n')
        p.stdin.flush()
        p.wait(timeout=10)
    except Exception:
        pass
    sys.stderr.write('moon_label done %d rows\n' % n)


if __name__ == '__main__':
    main()
