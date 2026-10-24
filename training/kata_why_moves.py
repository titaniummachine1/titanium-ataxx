"""Why does Kata pick its move? Compare Kata's 200-visit top move with alternatives on what the board looks like AFTER the
move and AFTER the enemy's best greedy reply (max conversions). Read-only, Kata raw self-play data."""
import sys, glob, numpy as np
sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P

KDIR = "C:/gitProjects/titanium-ataxx/data/datasets/KataGo_Ataxx_202604"
FULL = int(P.FULL)
R1 = [int(x) for x in P.R1]
R2 = [int(x) for x in P.R2]
pc = lambda x: bin(x).count("1")


def bb(v):
    return int(sum(int(v[s]) << s for s in range(49)))


def bits(x):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def reach(a, empty):
    c = 0
    j = 0
    for k in bits(a):
        c |= R1[k]
        j |= R2[k]
    return c & empty, j & empty & ~c


def play(u, e, frm, to):
    """mover u moves frm->to (frm None = clone). returns new (u, e)"""
    conv = e & R1[to]
    if frm is not None:
        u &= ~(1 << frm)
    return u | (1 << to) | conv, e & ~conv


def moves(u, e):
    empty = ~(u | e) & FULL
    out = []
    cl, _ = reach(u, empty)
    for t in bits(cl):
        out.append((None, t))
    for f in bits(u):
        for t in bits(R2[f] & empty):
            out.append((f, t))
    return out


def greedy_reply(u, e):
    """enemy e plays its max-conversion move (clone preferred on ties)"""
    best, bm = -1, None
    for f, t in moves(e, u):
        g = pc(u & R1[t]) * 2 + (1 if f is None else 0)
        if g > best:
            best, bm = g, (f, t)
    if bm is None:
        return u, e
    e2, u2 = play(e, u, *bm)
    return u2, e2


def describe(u, e):
    empty = ~(u | e) & FULL
    cu, ju = reach(u, empty)
    ce, je = reach(e, empty)
    lu, le = cu | ju, ce | je
    exposed_u = 0
    for k in bits(le):
        exposed_u |= R1[k]
    exposed_u &= u
    safe_u = sum(1 for k in bits(u) if not (R1[k] & empty))
    safe_e = sum(1 for k in bits(e) if not (R1[k] & empty))
    return dict(material=pc(u) - pc(e), only_me=pc(lu & ~le), only_en=pc(le & ~lu), my_clone_reach=pc(cu), en_clone_reach=pc(ce),
                my_exposed=pc(exposed_u), safe_diff=safe_u - safe_e)


rng = np.random.default_rng(13)
fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
cases = []
for f in [fs[i] for i in rng.choice(len(fs), 6, replace=False)]:
    z = np.load(f)
    pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
    gi = z["globalInputNC"]
    q = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
    emp = 49 - pl[:, 1].sum(1) - pl[:, 2].sum(1)
    st0 = {}
    for i in np.nonzero((gi[:, 0] == 0) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0) & (emp >= 12) & (emp <= 32))[0]:
        st0[(bb(pl[i, 1]), bb(pl[i, 2]))] = i
    # clone choices: stage-0 rows whose top square is empty
    for key, i in list(st0.items())[:600]:
        top = int(q[i, :49].argmax())
        if not ((key[0] >> top) & 1) and q[i, top] > 0.5:
            cases.append((key[0], key[1], (None, top)))
    for j in np.nonzero((gi[:, 0] == 1) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0))[0][:3000]:
        key = (bb(pl[j, 1]), bb(pl[j, 2]))
        if key in st0 and pl[j, 4].sum() == 1:
            i = st0[key]
            top0 = int(q[i, :49].argmax())
            org = int(np.nonzero(pl[j, 4])[0][0])
            if org == top0 and q[i, top0] > 0.5 and q[j, :49].max() > 0.5:
                cases.append((key[0], key[1], (org, int(q[j, :49].argmax()))))
rng.shuffle(cases)
cases = cases[:2500]
print("decisive Kata choices analysed (top move > 50%% search mass): %d (clones %d, jumps %d)" % (
    len(cases), sum(1 for c in cases if c[2][0] is None), sum(1 for c in cases if c[2][0] is not None)))

keys = ["material", "only_me", "only_en", "my_clone_reach", "en_clone_reach", "my_exposed", "safe_diff"]
wins = {k: [0, 0, 0] for k in keys}       # Kata better / equal / worse than the alternative, on this feature after the reply
diffs = {k: [] for k in keys}
alt_name = "best by material after the enemy's greedy reply (= what a 2-ply material search picks)"
agree = 0
for u, e, mv in cases:
    after = {}
    for m in moves(u, e):
        u1, e1 = play(u, e, *m)
        u2, e2 = greedy_reply(u1, e1)
        after[m] = describe(u2, e2)
    if mv not in after:
        continue
    alt = max(after, key=lambda m: (after[m]["material"], m[0] is None))
    if alt == mv:
        agree += 1
        continue
    a, b = after[mv], after[alt]
    for k in keys:
        d = a[k] - b[k]
        if k in ("only_en", "en_clone_reach", "my_exposed"):
            d = -d                                   # lower is better for these
        diffs[k].append(d)
        wins[k][0 if d > 0 else (1 if d == 0 else 2)] += 1
n_dis = len(cases) - agree
print("Kata's move == %s in %.0f%% of cases" % (alt_name, 100 * agree / len(cases)))
print("In the %d cases where they differ, after the enemy's greedy reply Kata's move is better/equal/worse on:" % n_dis)
for k in keys:
    w = wins[k]
    t = max(sum(w), 1)
    print("   %-15s better %3.0f%%  equal %3.0f%%  worse %3.0f%%   mean advantage %+.2f" % (k, 100 * w[0] / t, 100 * w[1] / t, 100 * w[2] / t, np.mean(diffs[k])))
