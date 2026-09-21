"""Feed label text chunks through `titanium-cli label` in parallel.

Overnight step 2 of the 12.4M relabel. One `label` process per chunk (each
is single-threaded with a shared TT inside the process), 8 workers default.
Resumable: chunks whose .out exists with matching line count are skipped
unless --redo. Corruption signal (per ORPHAN-WRITES rule): out lines must
equal in lines, else the chunk is reported FAIL and left for re-run.

Usage: python training/label_chunks.py --dir data/nnue/relab12m --workers 8
         --nodes 5000 --tuple data/nnue/tupS25ow20.tup --tuple-native
"""
import argparse
import concurrent.futures as cf
import glob
import os
import subprocess


def count_lines(p):
    with open(p, 'rb') as f:
        return sum(1 for _ in f)


def run_one(exe, txt, out, err, nodes, tup, native):
    cmd = [exe, 'label', '--nodes', str(nodes), '--tuple', tup]
    if native:
        cmd.append('--tuple-native')
    with open(txt, 'rb') as fin, open(out, 'wb') as fout, open(err, 'wb') as ferr:
        p = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=ferr)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/nnue/relab12m')
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--nodes', type=int, default=5000)
    ap.add_argument('--tuple', default='data/nnue/tupS25ow20.tup')
    ap.add_argument('--tuple-native', action='store_true', default=True)
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()

    txts = sorted(glob.glob(os.path.join(a.dir, 'lab_*.txt')))
    assert txts, 'no lab_*.txt in %s (run dump_dag12m.py first)' % a.dir
    print('chunks %d workers %d nodes %d tuple %s' % (len(txts), a.workers, a.nodes, a.tuple),
          flush=True)

    def job(txt):
        base, _ = os.path.splitext(txt)
        out, err = base + '.out', base + '.err'
        want = count_lines(txt)
        if os.path.exists(out) and not a.redo:
            got = count_lines(out)
            if got == want:
                return (txt, 'SKIP-OK %d' % got)
        rc = run_one(a.exe, txt, out, err, a.nodes, a.tuple, a.tuple_native)
        got = count_lines(out) if os.path.exists(out) else -1
        if rc == 0 and got == want:
            return (txt, 'OK %d' % got)
        return (txt, 'FAIL rc=%d got=%d want=%d' % (rc, got, want))

    fails = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for txt, msg in ex.map(job, txts):
            print('%s: %s' % (os.path.basename(txt), msg), flush=True)
            if msg.startswith('FAIL'):
                fails += 1
    print('label done, fails=%d' % fails, flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
