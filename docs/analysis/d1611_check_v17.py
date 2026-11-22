# D1611 independent check (analysis only, read-only, chunked: never loads a train file whole) of a proof merge:
#   python -I d1611_check_v17.py BASE.tdat NEW.tdat TAIL N_NEW proof_all.tdat excluded.keys
# 1 rows(NEW) = rows(BASE) + N_NEW; 2 the held-out tail is byte-identical; 3 BASE head vs NEW rows [0, head): rows differ
# only in outcome / cp / proof_kind and only where NEW has proof_kind 6; 4 the appended rows are all proven and consistent;
# 5 every proof_kind-6 row: |cp| >= 98000 and outcome = (cp > 0); 6 own key code (numpy splitmix = hub pos_key): every head
# row of a usable proven position (both-ways and excluded keys removed) carries proof_kind 6 and the proof's sign, and
# every proof_kind-6 row is such a position.
import os, sys
import numpy as np
REC = np.dtype([('o0', '<u8'), ('o1', '<u8'), ('out', '<f4'), ('cp', '<f4'), ('w', '<f4'), ('p3', '<f4'), ('cp5k', '<f4'),
                ('zx', '<i4'), ('stm', 'u1'), ('tk', 'u1'), ('pk', 'u1'), ('rs', 'u1'), ('cl', 'u1'), ('pad', 'V3')])
assert REC.itemsize == 48
U = np.uint64
def splitmix(x):
    with np.errstate(over='ignore'):
        x = x + U(0x9E3779B97F4A7C15)
        x = (x ^ (x >> U(30))) * U(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U(27))) * U(0x94D049BB133111EB)
        return x ^ (x >> U(31))
def keys(a):
    m = np.where(a['stm'] == 0, a['o0'], a['o1'])
    o = np.where(a['stm'] == 0, a['o1'], a['o0'])
    return splitmix(splitmix(m) ^ ((o << U(17)) | (o >> U(47))))
base, new, tail, n_new, pall, exk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
nb, nn = os.path.getsize(base) // 48, os.path.getsize(new) // 48
fails = []
def check(ok, msg):
    print(("OK   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)
check(nn == nb + n_new, f"rows {nn:,} = base {nb:,} + new {n_new:,}")
# usable proofs, computed here independently of the hub tool
p = np.fromfile(pall, dtype=REC)
pr = np.abs(p['cp']) >= 98000
p = p[pr]
pk_, win = keys(p), p['cp'] > 0
ex = set(int(l, 16) for l in open(exk) if l.strip())
order = np.argsort(pk_, kind='stable'); pk_, win = pk_[order], win[order]
u, first, cnt = np.unique(pk_, return_index=True, return_counts=True)
wmin = np.minimum.reduceat(win.astype(np.int8), first); wmax = np.maximum.reduceat(win.astype(np.int8), first)
both = wmin != wmax
exm = np.array([int(k) in ex for k in u.tolist()]) if len(ex) else np.zeros(len(u), bool)
keep = ~both & ~exm
uk, uw = u[keep], wmax[keep].astype(bool)
print(f"proof-line rows proven {int(pr.sum()):,} of {len(pr):,}; distinct {len(u):,}; both ways {int(both.sum()):,}; excluded {int(exm.sum()):,}; usable {len(uk):,}")
cut_b, cut_n = nb - tail, nn - tail
CH = 1_000_000
n_diff = n_pk6 = bad_fields = bad_pk6 = miss = wrong_sign = stray = app_bad = 0
with open(base, 'rb') as fb, open(new, 'rb') as fn:
    i = 0
    while i < cut_n:
        c = min(CH, cut_n - i)
        a = np.fromfile(fn, dtype=REC, count=c)
        k = keys(a)
        j = np.searchsorted(uk, k).clip(0, max(len(uk) - 1, 0))
        inmap = (uk[j] == k) if len(uk) else np.zeros(c, bool)
        is6 = a['pk'] == 6
        n_pk6 += int(is6.sum())
        bad_pk6 += int((is6 & ((np.abs(a['cp']) < 98000) | (a['out'] != (a['cp'] > 0).astype(np.float32)))).sum())
        miss += int((inmap & ~is6).sum())
        stray += int((is6 & ~inmap).sum())
        wrong_sign += int((inmap & ((a['cp'] > 0) != uw[j])).sum())
        # base comparison only for rows that exist in the base head
        hb = max(0, min(c, cut_b - i))
        if hb > 0:
            b = np.fromfile(fb, dtype=REC, count=hb)
            x = a[:hb]
            same_rest = (b['o0'] == x['o0']) & (b['o1'] == x['o1']) & (b['w'] == x['w']) & (b['p3'] == x['p3']) & \
                        (b['cp5k'] == x['cp5k']) & (b['zx'] == x['zx']) & (b['stm'] == x['stm']) & (b['tk'] == x['tk']) & \
                        (b['rs'] == x['rs']) & (b['cl'] == x['cl']) & (b['pad'] == x['pad'])
            diff = ~((b['out'] == x['out']) & (b['cp'] == x['cp']) & (b['pk'] == x['pk']) & same_rest)
            n_diff += int(diff.sum())
            bad_fields += int((~same_rest).sum() + (diff & ~is6[:hb]).sum())
        if c > hb:  # appended rows
            y = a[hb:]
            app_bad += int(((np.abs(y['cp']) < 98000) | (y['out'] != (y['cp'] > 0).astype(np.float32)) | (y['pk'] != 6)).sum())
        i += c
    # tail
    fb.seek(cut_b * 48); fn.seek(cut_n * 48)
    same_tail = True
    left = tail
    while left > 0:
        c = min(CH, left) * 48
        if fb.read(c) != fn.read(c):
            same_tail = False
        left -= c // 48
check(same_tail, f"held-out tail ({tail:,} rows) byte-identical")
check(bad_fields == 0, f"base head rows changed only in outcome/cp/proof_kind and only on proof_kind 6 rows (changed rows {n_diff:,}, bad {bad_fields:,})")
check(app_bad == 0, f"appended rows all proven, outcome consistent, proof_kind 6 (bad {app_bad:,})")
check(bad_pk6 == 0, f"proof_kind-6 rows {n_pk6:,}: all |cp| >= 98000 with outcome = (cp > 0) (bad {bad_pk6:,})")
check(miss == 0 and stray == 0, f"usable proven positions <-> proof_kind 6 in the head (missed {miss:,}, stray {stray:,})")
check(wrong_sign == 0, f"proof_kind-6 rows carry the proof's sign (wrong {wrong_sign:,})")
print("ALL CHECKS PASS" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
