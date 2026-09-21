"""Overnight driver: dump -> label (6 workers) -> join, sequentially, resumable.

Launched detached; logs to data/nnue/relab12m/overnight.log.
Each step skips finished chunks, so killing + re-running resumes.
"""
import subprocess
import sys
import time


def run(cmd, log):
    log.write('\n### %s\n' % ' '.join(cmd))
    log.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        log.write(line)
        log.flush()
    p.wait()
    return p.returncode


def main():
    t0 = time.time()
    log = open('data/nnue/relab12m/overnight.log', 'a')
    log.write('\n===== overnight start %s =====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log.flush()
    rc = run([sys.executable, 'training/dump_dag12m.py', '--chunks', '16'], log)
    log.write('dump rc=%d (%.0fs)\n' % (rc, time.time() - t0))
    log.flush()
    if rc != 0:
        log.write('DUMP FAILED, abort\n')
        log.close()
        return
    t1 = time.time()
    rc = run([sys.executable, 'training/label_chunks.py', '--dir', 'data/nnue/relab12m',
              '--workers', '6', '--nodes', '5000',
              '--tuple', 'data/nnue/tupS25ow20.tup', '--tuple-native'], log)
    log.write('label rc=%d (%.0fs)\n' % (rc, time.time() - t1))
    log.flush()
    if rc != 0:
        log.write('LABEL FAILED (partial progress kept), abort\n')
        log.close()
        return
    t2 = time.time()
    rc = run([sys.executable, 'training/join12m.py', '--dir', 'data/nnue/relab12m',
              '--out', 'data/nnue/cache12M.npz'], log)
    log.write('join rc=%d (%.0fs)\n' % (rc, time.time() - t2))
    log.write('===== overnight done total %.0fs =====\n' % (time.time() - t0))
    log.flush()
    log.close()


if __name__ == '__main__':
    main()
