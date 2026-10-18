#!/usr/bin/env python3
"""Idle-triggered autonomous gate runner (user-ordered 2026-10-18).

Watches CPU: when utilization stays BELOW 50% for a full hour (any sample
>= 51% resets the timer to zero), it takes over at MAX utilization and
works the backlog queue branch-by-branch, fishtest-style: build the
branch binary, SPRT-gate vs main, log verdict + miss risk, bank results.
It PREPARES approve/reject decisions; it never merges (merges need the
user + gate_check.py + LEDGER row per standing law).

Interrupt: create training/STOP_AUTO (or run logs/stop_auto.cmd) and the
runner exits promptly between chunks. Also killable via schtasks /end.

Usage:
  python3 training/auto_gate.py --queue training/auto_queue.json
    --idle-minutes 60 --idle-threshold 50 --sample-secs 60
    --max-workers 7
State in logs/auto_gate.state.json (idle timer survives restarts).
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = os.path.join(REPO, 'training', 'STOP_AUTO')
STATE = os.path.join(REPO, 'logs', 'auto_gate.state.json')
LOCK = os.path.join(REPO, 'logs', 'auto_gate.lock')

S1 = 1 / (1 + 10 ** (-20 / 400))


def sprt_llr(w, l, d, s0=0.5, s1=S1):
    n = w + l + d
    if not n:
        return 0.0
    return ((w + d / 2) * math.log(s1 / s0) +
            (l + d / 2) * math.log((1 - s1) / (1 - s0)))


def miss_risk(w, l, d):
    n = w + l + d
    if not n:
        return 0.5
    s = (w + d / 2) / n
    se = math.sqrt(max(s * (1 - s), 1e-9) / n)
    from math import erf, sqrt
    return 1 - 0.5 * (1 + erf((0.5 - s) / (max(se, 1e-9) * sqrt(2))))


def cpu_pct():
    """Real CPU % or None (never a fake number). typeperf first, wmic backup."""
    try:
        out = subprocess.run(
            ['typeperf', r'\Processor(_Total)\% Processor Time', '-sc', '1'],
            capture_output=True, text=True, timeout=30).stdout
        # data line looks like: "10/18/2026 22:12:47.826","12.151063"
        for line in reversed(out.strip().splitlines()):
            if '","' in line:
                val = line.rsplit('","', 1)[1].strip().strip('"')
                return float(val.replace(',', '.'))
    except Exception as e:
        print('typeperf failed: %s' % str(e)[:160], flush=True)
    try:
        out = subprocess.run(
            ['wmic', 'cpu', 'get', 'loadpercentage', '/value'],
            capture_output=True, text=True, timeout=30).stdout
        m = re.search(r'LoadPercentage=(\d+)', out)
        if m:
            return int(m.group(1))
    except Exception as e:
        print('wmic failed: %s' % str(e)[:120], flush=True)
    print('CPU SENSOR FAILED (raw wmic/typeperf output above if any)', flush=True)
    return None


def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {'idle_secs': 0, 'last_check': time.time()}


def save_state(st):
    try:
        json.dump(st, open(STATE, 'w'))
    except OSError:
        pass


def idle_ready(args):
    """True once CPU < threshold for idle_minutes continuously.
    Unknown readings change nothing (no fake resets, no fake accumulation)."""
    st = load_state()
    now = time.time()
    pct = cpu_pct()
    if pct is None:
        print('cpu UNKNOWN idle %ds/%ds (held)' % (
            int(st['idle_secs']), int(args.idle_minutes * 60)), flush=True)
    elif pct >= args.idle_threshold + 1:
        st['idle_secs'] = 0
    else:
        st['idle_secs'] += max(0, now - st.get('last_check', now))
    st['last_check'] = now
    save_state(st)
    return (st['idle_secs'] >= args.idle_minutes * 60 and pct is not None), pct, st['idle_secs']


def sh(cmd, cwd=REPO, timeout=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, shell=isinstance(cmd, str))


def ensure_worktree(item):
    """Dedicated worktree per branch so gating never disturbs the dev tree.
    Returns (worktree_path, commit_hash)."""
    wt = item.get('worktree')
    br = item['branch']
    if not wt:
        wt = os.path.join(os.path.dirname(REPO), 'titanium-gate-' + item['tag'])
        item['worktree'] = wt
    if not os.path.isdir(os.path.join(wt, '.git')):
        # worktree metadata lives in main repo; check by path existence
        r = sh(['git', '-C', REPO, 'worktree', 'add', wt, br])
        if r.returncode != 0 and 'already used' not in (r.stderr or ''):
            # maybe branch needs no checkout: path exists but not a worktree
            pass
    # make sure the worktree tracks the latest commit of the branch
    r = sh(['git', '-C', wt, 'rev-parse', 'HEAD'])
    head = r.stdout.strip() if r.returncode == 0 else 'unknown'
    want = item.get('commit')
    if want and want != head:
        sh(['git', '-C', wt, 'checkout', want])
        head = want
    else:
        # pull branch tip (fast-forward only, never touch dirty tree)
        st = sh(['git', '-C', wt, 'status', '--short'])
        if st.returncode == 0 and not st.stdout.strip():
            sh(['git', '-C', wt, 'checkout', br])
            sh(['git', '-C', wt, 'merge', '--ff-only', 'origin/' + br])
            r = sh(['git', '-C', wt, 'rev-parse', 'HEAD'])
            head = r.stdout.strip() if r.returncode == 0 else head
    return wt, head


def build_branch(item):
    """Build the branch binary into its own target dir. Returns (exe, err)."""
    wt, head = ensure_worktree(item)
    item['commit_built'] = head
    bdir = os.path.join(wt, 'target-gate')
    r = sh(['cargo', 'build', '--release', '--target-dir', bdir,
            '--bin', 'titanium-cli'], cwd=wt, timeout=3600)
    if r.returncode != 0:
        return None, (r.stderr or '')[-2000:]
    exe = os.path.join(bdir, 'release', 'titanium-cli.exe')
    if not os.path.exists(exe):
        return None, 'binary missing after build'
    return exe, ''


def stop_requested():
    return os.path.exists(STOP)


def run_chunk(item, exe_b, book, games, seed, out, workers, nodes, tup):
    if item.get('cloud') is not None:
        job = {'a_exe': item['exe'], 'b_exe': exe_b, 'tup': tup,
               'book': book, 'games': games, 'nodes': nodes,
               'movetime': 0, 'seed': seed, 'workers': workers,
               'out': out}
        try:
            return item['cloud'].run_match(job)
        except Exception as e:
            print('cloud failed (%s); local fallback' % str(e)[:160], flush=True)
    cmd = [sys.executable, os.path.join('training', 'book_gate.py'),
           '--book', book, '--games', str(games),
           '--titanium', item['exe'], '--targs', 'serve', '--tup', tup,
           '--opp', exe_b, '--oargs', 'serve', '--opp-tup', tup,
           '--nodes', str(nodes), '--opp-nodes', str(nodes),
           '--workers', str(workers), '--seed', str(seed), '--out', out]
    r = sh(cmd, timeout=7200)
    try:
        txt = open(out).read()
    except OSError:
        return None
    m = re.findall(r'total: titan (\d+) - moon (\d+) draws (\d+)', txt)
    if not m:
        m = re.findall(r'(\d+) - (\d+) draws (\d+)', txt)
    if not m:
        return None
    w, l, d = map(int, m[-1])
    return w, l, d


def gate_item(item, args, logf):
    """SPRT-gate one backlog item to PASS/FAIL/UNDECIDED(cap)."""
    tag = item['tag']
    print('### %s: building %s' % (tag, item['branch']), flush=True)
    logf.write('### %s branch=%s\n' % (tag, item['branch']))
    logf.flush()
    exe_a, err = build_branch(item)
    if not exe_a:
        print('BUILD FAIL %s: %s' % (tag, err[-300:]), flush=True)
        return 'BUILDFAIL', 0, 0, 0
    item['exe'] = exe_a
    print('### %s built %s @ %s' % (tag, item['branch'], item.get('commit_built', '?')[:12]), flush=True)
    exe_b = item['main_exe']
    tup = item['tup']
    W = L = D = 0
    seed0 = item.get('seed', 100)
    ci = 0
    max_chunks = item.get('max_chunks', 20)
    while ci < max_chunks:
        if stop_requested():
            return 'STOPPED', W, L, D
        # Loop books from the start when exhausted (user-ordered extension).
        book = item['books'][ci % len(item['books'])]
        out = os.path.join(REPO, 'logs', 'auto_%s_c%d.txt' % (tag, ci))
        res = run_chunk(item, exe_b, book, item.get('games', 200),
                        seed0 + ci, out, args.max_workers,
                        item.get('nodes', 5000), tup)
        if res is None:
            print('%s chunk %d: NO RESULT' % (tag, ci), flush=True)
            continue
        w, l, d = res
        W, L, D = W + w, L + l, D + d
        v = sprt_llr(W, L, D)
        miss = miss_risk(W, L, D)
        print('%s chunk %d: %d-%d-%d cum %d-%d-%d LLR %+.2f miss %.3f' % (
            tag, ci, w, l, d, W, L, D, v, miss), flush=True)
        logf.write('%s c%d: %d-%d-%d cum %d-%d-%d LLR %+.2f miss %.3f\n' % (
            tag, ci, w, l, d, W, L, D, v, miss))
        logf.flush()
        if v >= 1.0:
            return 'PASS', W, L, D
        if v <= -1.0:
            return 'FAIL', W, L, D
        ci += 1
    return 'UNDECIDED', W, L, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', default=os.path.join('training', 'auto_queue.json'))
    ap.add_argument('--idle-minutes', type=float, default=60)
    ap.add_argument('--idle-threshold', type=float, default=50)
    ap.add_argument('--sample-secs', type=float, default=60)
    ap.add_argument('--max-workers', type=int, default=7)
    ap.add_argument('--force', action='store_true',
                    help='skip idle wait, gate immediately')
    ap.add_argument('--only', default='',
                    help='gate only this item tag (manual trigger)')
    ap.add_argument('--commit', default='',
                    help='pin a specific commit instead of branch HEAD (default: latest)')
    ap.add_argument('--cloud', action='store_true',
                    help='MANUAL ONLY: run matches on Modal cloud (default: local '
                         'background). Needs training/cloud_backend.py configured; '
                         'falls back to local with a warning otherwise.')
    a = ap.parse_args()

    if a.only:
        a.force = True
    os.makedirs(os.path.join(REPO, 'logs'), exist_ok=True)

    if not a.force:
        print('watching CPU: need <%g%% for %g min (reset at >=%g%%)' % (
            a.idle_threshold, a.idle_minutes, a.idle_threshold + 1), flush=True)
        while True:
            if stop_requested():
                print('STOP file present, exiting', flush=True)
                return
            ready, pct, secs = idle_ready(a)
            ps = ('%d' % pct) if pct is not None else '??'
            print('cpu %s%% idle %ds/%ds' % (ps, int(secs), int(a.idle_minutes * 60)), flush=True)
            if ready:
                break
            time.sleep(a.sample_secs)

    try:
        queue = json.load(open(a.queue if os.path.isabs(a.queue)
                               else os.path.join(REPO, a.queue)))
    except (OSError, ValueError) as e:
        print('queue load failed: %s' % e, flush=True)
        return
    print('IDLE TRIGGER: gating %d items at %d workers' % (len(queue), a.max_workers), flush=True)
    if os.path.exists(STOP):
        os.remove(STOP)
    logf = open(os.path.join(REPO, 'logs', 'auto_gate.log'), 'a')
    cloud = None
    if a.cloud:
        try:
            sys.path.insert(0, os.path.join(REPO, 'training'))
            import cloud_backend as cb
            cloud = cb
            print('CLOUD backend active (manual override)', flush=True)
        except Exception as e:
            print('cloud requested but backend missing (%s); local fallback' % e, flush=True)
    for item in queue:
        if a.only and item['tag'] != a.only:
            continue
        if a.commit:
            item['commit'] = a.commit
        if cloud is not None:
            item['cloud'] = cloud
        if stop_requested():
            print('STOP requested, exiting queue', flush=True)
            break
        if item.get('done') or item.get('enabled') is False:
            print('skip %s (disabled/done)' % item['tag'], flush=True)
            continue
        t0 = time.time()
        verdict, w, l, d = gate_item(item, a, logf)
        v = sprt_llr(w, l, d)
        row = ('| %s | %s | %s vs main | %s | %d | %d-%d-%d | LLR %+.2f | %s |\n' % (
            item['tag'], time.strftime('%Y-%m-%d'), item['desc'],
            item.get('gate', '5k nodes'), w + l + d, w, l, d, v, verdict))
        logf.write('RESULT %s: %d-%d-%d LLR %+.2f miss %.3f\n' % (
            tag_verdict(item['tag'], verdict), w, l, d, v, miss_risk(w, l, d)))
        logf.write(row)
        logf.flush()
        print('RESULT %s: %s %d-%d-%d LLR %+.2f (%.0fs)' % (
            item['tag'], verdict, w, l, d, v, time.time() - t0), flush=True)
    logf.close()
    # Queue drained and still idle: relabel important positions @500k
    # (top regret @1M) until interrupted. Fodder for the next alignment.
    if not stop_requested():
        print('queue drained: switching to relabel mode', flush=True)
        logf = open(os.path.join(REPO, 'logs', 'auto_gate.log'), 'a')
        logf.write('RELABEL_MODE %s\n' % time.strftime('%Y-%m-%d %H:%M'))
        logf.flush()
        r = sh([sys.executable, os.path.join('training', 'relabel_idle.py'),
                '--workers', str(a.max_workers)])
        logf.write('RELABEL_EXIT rc=%s\n' % r.returncode)
        logf.close()
    # reset idle timer so it re-arms
    save_state({'idle_secs': 0, 'last_check': time.time()})


def tag_verdict(tag, verdict):
    return '%s=%s' % (tag, verdict)


if __name__ == '__main__':
    main()
