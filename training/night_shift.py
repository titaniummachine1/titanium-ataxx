"""Overnight driver 2026-10-18/19: mechanical steps only, zero judgment calls.
Every step is try/except-logged to night.log. Verdicts are COMPUTED and
REPORTED (NIGHT_REPORT.md); nothing merges, nothing folds, nothing decides.
Kill switch: create C:\\gitProjects\\logs\\STOP and the driver aborts ASAP.

Order:
  0. wait for pcut repeat screen (titan-pcut2) -> pooled verdict + heterogeneity
  1. build curve branch -> target-curve (isolated dir, never touches target/)
  2. uai smoke on curve binary
  3. sloss 4-param curve tune (40 iters, deep_fresh, --force-mode 3)
  4. report best-vs-base (NO fold)
  5. fresh-book extension screen for pcut IF pooled lead holds (books aa/ab via make_book)
  6. NIGHT_REPORT.md
"""
import os, sys, subprocess, time, json, re, datetime

REPO = 'C:/gitProjects/titanium-ataxx'
LOGS = 'C:/gitProjects/logs'
STOP = os.path.join(LOGS, 'STOP')
NLOG = os.path.join(LOGS, 'night_driver.log')
REPORT = os.path.join(LOGS, 'NIGHT_REPORT.md')

def log(msg):
    line = '%s %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(NLOG, 'a') as f:
        f.write(line + '\n')

def stopped():
    return os.path.exists(STOP)

def run(cmd, cwd=REPO, timeout_h=3):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=int(timeout_h * 3600))
        return p.returncode, (p.stdout or '')[-2000:], (p.stderr or '')[-2000:]
    except subprocess.TimeoutExpired:
        return -9, '', 'TIMEOUT'

def git(*args):
    return run(['git', '-C', REPO] + list(args), timeout_h=0.1)

def parse_total(path):
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except FileNotFoundError:
        return None
    m = None
    for line in txt.splitlines():
        mm = re.match(r'total: (\S+) (\d+) - (\S+) (\d+) draws (\d+) \(n=(\d+)\)'
                      r' forfeits A=(\d+) B=(\d+) score ([\d.]+) CI \[([\d.]+),([\d.]+)\]'
                      r' Elo ([+-]?[\d.]+)', line.strip())
        if mm:
            m = mm
    if not m:
        return None
    g = m.groups()
    return dict(a=g[0], wa=int(g[1]), b=g[2], wb=int(g[3]), d=int(g[4]),
                n=int(g[5]), fa=int(g[6]), fb=int(g[7]), s=float(g[8]),
                lo=float(g[9]), hi=float(g[10]), elo=float(g[11]))

results = {}

# ---- step 0: wait for pcut2 screen ----
log('STEP0 wait pcut2 screen (w_pcut_y.txt + w_pcut_x.txt)')
t0 = time.time()
legs = {}
while time.time() - t0 < 3 * 3600:
    if stopped():
        log('STOP file present, aborting'); sys.exit(3)
    for tag in ('y', 'x'):
        if tag not in legs:
            r = parse_total(os.path.join(LOGS, 'w_pcut_%s.txt' % tag))
            if r and r['n'] >= 200:
                legs[tag] = r
                log('leg %s: %d-%d elo %+g ci [%.3f,%.3f] forf %d/%d' % (
                    tag, r['wa'], r['wb'], r['elo'], r['lo'], r['hi'],
                    r['fa'], r['fb']))
    if len(legs) == 2:
        break
    time.sleep(60)
results['pcut_legs'] = legs
if len(legs) == 2:
    wa = sum(r['wa'] for r in legs.values()); wb = sum(r['wb'] for r in legs.values())
    n = sum(r['n'] for r in legs.values())
    s = wa / n
    import math
    elo = -400 * math.log10(1 / s - 1) if 0 < s < 1 else 0.0
    results['pcut_pooled'] = dict(w=wa, l=wb, n=n, score=round(s, 4), elo=round(elo, 1))
    elos = sorted(r['elo'] for r in legs.values())
    results['pcut_hetero'] = round(elos[1] - elos[0], 1)
    log('POOLED y+x: %d-%d/%d score %.4f elo %+.1f hetero spread %.1f' % (
        wa, wb, n, s, elo, results['pcut_hetero']))
else:
    log('STEP0 TIMEOUT waiting for legs, continuing with curve work')

# ---- step 1: build curve branch into isolated dir ----
log('STEP1 checkout lmrcurve + build to target-curve')
rc, o, e = git('checkout', 'exp/search-lmrcurve', '--quiet')
log('checkout rc=%d' % rc)
env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(REPO, 'target-curve'))
try:
    p = subprocess.run(
        ['cargo', 'build', '--release', '--manifest-path',
         os.path.join(REPO, 'crates', 'titanium-cli', 'Cargo.toml')],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    log('curve build rc=%d tail=%s' % (p.returncode, (p.stderr or '')[-300:]))
    results['curve_build'] = p.returncode
except subprocess.TimeoutExpired:
    log('curve build TIMEOUT'); results['curve_build'] = -9
if stopped():
    log('STOP, aborting'); sys.exit(3)

# ---- step 2: uai smoke ----
cb = os.path.join(REPO, 'target-curve', 'release', 'titanium-cli.exe')
smoke_in = os.path.join(LOGS, 'uai_smoke.in')
if results.get('curve_build') == 0 and os.path.exists(cb):
    try:
        p = subprocess.run(
            [cb, 'serve'], stdin=open(smoke_in), capture_output=True,
            text=True, timeout=90, cwd=LOGS)
        out = p.stdout or ''
        ok = ('bestmove' in out and 'CurveBase' in out)
        log('curve smoke ok=%s rc=%d' % (ok, p.returncode))
        results['curve_smoke'] = bool(ok)
    except Exception as ex:
        log('curve smoke EXC %s' % ex); results['curve_smoke'] = False
else:
    results['curve_smoke'] = False

# ---- step 3: curve sloss tune ----
if results.get('curve_smoke'):
    log('STEP3 curve sloss 40 iters force-mode 3')
    cmd = [sys.executable, 'training/search_loss.py', '--iters', '40',
           '--jsonl', os.path.join(LOGS, 'deep_fresh.jsonl'),
           '--batch', '400', '--shallow', '10000', '--workers', '4',
           '--ascale', '0.5', '--seed', '62',
           '--params', 'LMRMode,CurveBase,CurveSlope,CurveDepth,CurveCap',
           '--force-mode', '3',
           '--state', 'logs/sloss_curve.json', '--log', 'logs/sloss_curve.jsonl']
    try:
        with open(os.path.join(LOGS, 'sloss_curve_run.log'), 'w') as lf:
            lf.write('CURVE_START %s\n' % datetime.datetime.now())
            lf.flush()
            p = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=3 * 3600)
            lf.write('CURVE_DONE rc=%d\n' % p.returncode)
        log('curve sloss rc=%d' % p.returncode)
        results['curve_sloss'] = p.returncode
    except subprocess.TimeoutExpired:
        log('curve sloss TIMEOUT'); results['curve_sloss'] = -9
else:
    log('STEP3 SKIPPED (no curve binary)'); results['curve_sloss'] = 'skip'

# ---- step 4: summarize curve tune ----
if results.get('curve_sloss') == 0:
    try:
        best = None
        for line in open(os.path.join(LOGS, 'sloss_curve_run.log'),
                         encoding='utf-8', errors='replace'):
            m = re.match(r'iter (\d+): loss\+ ([\d.]+).*loss- ([\d.]+).*base ([\d.]+) theta=(\{.*\})', line.strip())
            if m:
                lp, lm, base = float(m.group(2)), float(m.group(3)), float(m.group(4))
                avg = (lp + lm) / 2
                if best is None or avg < best[0]:
                    best = (avg, int(m.group(1)), base, m.group(5))
        if best:
            results['curve_best'] = dict(avg=round(best[0], 5), it=best[1],
                                         base=best[2],
                                         improv_pct=round(100 * (best[2] - best[0]) / best[2], 2))
            log('curve best avg %.5f @iter %d vs base %.5f (%+.2f%%)' % (
                best[0], best[1], best[2], results['curve_best']['improv_pct']))
    except Exception as ex:
        log('curve summary EXC %s' % ex)

# ---- step 5: pcut extension on fresh books (only if pooled lead holds) ----
pp = results.get('pcut_pooled')
if pp and pp['score'] > 0.5 and not stopped():
    log('STEP5 pcut extension: make 2 fresh books + 200g each')
    try:
        p = subprocess.run(
            [sys.executable, 'training/make_book.py', '--help'],
            cwd=REPO, capture_output=True, text=True, timeout=60)
        log('make_book help rc=%d tail=%s' % (p.returncode, (p.stdout or '')[-500:]))
        results['make_book_help'] = p.returncode
    except Exception as ex:
        log('make_book EXC %s' % ex); results['make_book_help'] = 'exc'

# ---- step 6: report ----
json.dump(results, open(os.path.join(LOGS, 'night_results.json'), 'w'), indent=1)
with open(REPORT, 'w') as f:
    f.write('# NIGHT REPORT %s\n\n' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    f.write('Driver: training/night_shift.py (mechanical only, nothing merged/folded).\n\n')
    f.write('```json\n%s\n```\n' % json.dumps(results, indent=1))
    f.write('\n## For the morning session\n')
    f.write('- Read LEDGER.md tail + this report + logs/night.log tail.\n')
    f.write('- Pcut verdict needs pooled y+x PLUS heterogeneity judgment (v4).\n')
    f.write('- Curve tune (if ran) is search-loss only: needs a game screen before any fold.\n')
    f.write('- Worktree branch at driver end: see night.log tail.\n')
log('REPORT written to %s' % REPORT)
rc, o, e = git('checkout', 'exp/search-probcut', '--quiet')
log('restored worktree to probcut rc=%d' % rc)
