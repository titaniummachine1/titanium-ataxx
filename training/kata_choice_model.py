"""Why does KataGo pick its move? A conditional-logit CHOICE MODEL over all legal moves. READ-ONLY.

Data: Kata's own self-play (raw tdata), decisive decisions only (search top move > 50% of visits). Clones come from stage-0
rows; jumps from stage-0 rows paired with the stage-1 row of the same board (picked stone = origin, top destination).
For every legal move we compute what it does (side-to-move relative, exact bit operations):
  conv            stones converted immediately
  is_jump         jump (vacates the origin) vs clone
  org_attackable  jump only: the origin stone could be converted by the enemy next move
  org_support     jump only: own stones around the origin (what the vacated hole leaves behind)
  dest_filled     filled neighbours around the landing square (filling a hole vs landing in the open)
  recapture       the enemy's best conversion count right after the move
  my_exposed_a    my stones next to a square the enemy can land on, after the move
  en_exposed_a    enemy stones next to a square I can land on, after the move
  safe_a          (my stones with no empty neighbour) - (enemy's), after the move
  reach_a         (my clone-reach squares) - (enemy's), after the move
  only_a          (empties only I can reach) - (empties only the enemy can reach), after the move
  mat_2ply        material after the enemy's greedy (max-conversion) reply
P(move) is proportional to exp(w . x); w fitted by maximum likelihood on Kata's chosen moves (70% of positions),
scored by top-1 agreement on the other 30%. Coefficients are reported per natural unit so they read as exchange rates.
Usage: python training/kata_choice_model.py [n_cases]
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P  # noqa: E402

KDIR = "C:/gitProjects/titanium-ataxx/data/datasets/KataGo_Ataxx_202604"
FULL = int(P.FULL)
R1 = [int(x) for x in P.R1]
R2 = [int(x) for x in P.R2]
FEATS = ["conv", "is_jump", "org_attackable", "org_support", "dest_filled", "recapture", "my_exposed_a", "en_exposed_a",
         "safe_a", "reach_a", "only_a", "mat_2ply",
         # v2 (after the colour-flip finding: permanence is what the value counts)
         "my_perm_a", "en_perm_a", "conv_perm", "hole_exposes", "dest_wall", "my_perm_2ply", "my_exposed_2ply", "safe_2ply"]


def pc(x):
    return bin(x).count("1")


def bits(x):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def bb(v):
    return int(sum(int(v[s]) << s for s in range(49)))


def landing(a, empty):
    c = j = 0
    for k in bits(a):
        c |= R1[k]
        j |= R2[k]
    return c & empty, (c | j) & empty


def play(u, e, frm, to):
    conv = e & R1[to]
    if frm is not None:
        u &= ~(1 << frm)
    return u | (1 << to) | conv, e & ~conv


def moves(u, e):
    empty = ~(u | e) & FULL
    cl, _ = landing(u, empty)
    out = [(None, t) for t in bits(cl)]
    for f in bits(u):
        for t in bits(R2[f] & empty):
            out.append((f, t))
    return out


def best_capture(att, vic, empty):
    _, land = landing(att, empty)
    return max((pc(vic & R1[t]) for t in bits(land)), default=0)


def greedy_reply(u, e):
    best, bm = -1, None
    for f, t in moves(e, u):
        g = pc(u & R1[t]) * 2 + (1 if f is None else 0)
        if g > best:
            best, bm = g, (f, t)
    if bm is None:
        return u, e
    e2, u2 = play(e, u, *bm)
    return u2, e2


def feats(u, e, m):
    frm, to = m
    empty0 = ~(u | e) & FULL
    _, land_e0 = landing(e, empty0)
    att0 = 0
    for k in bits(land_e0):
        att0 |= R1[k]
    u1, e1 = play(u, e, frm, to)
    empty = ~(u1 | e1) & FULL
    cu, lu = landing(u1, empty)
    ce, le = landing(e1, empty)
    nle = nlu = 0
    for k in bits(le):
        nle |= R1[k]
    for k in bits(lu):
        nlu |= R1[k]
    safe_u = sum(1 for k in bits(u1) if not (R1[k] & empty))
    safe_e = sum(1 for k in bits(e1) if not (R1[k] & empty))
    u2, e2 = greedy_reply(u1, e1)
    # permanence: stones the other side cannot land next to (cannot be converted next move)
    my_perm = pc(u1 & ~nle)
    en_perm = pc(e1 & ~nlu)
    convd = e & R1[to]
    conv_perm = sum(1 for k in bits(convd | (1 << to)) if not (R1[k] & le))
    hole_exp = 0
    if frm is not None:                                   # my stones next to the vacated origin that the enemy can now reach
        hole_exp = pc(u1 & R1[frm]) if (le >> frm) & 1 else 0
    wall = 8 - pc(R1[to])
    empty2 = ~(u2 | e2) & FULL
    _, le2 = landing(e2, empty2)
    _, lu2 = landing(u2, empty2)
    nle2 = 0
    for k in bits(le2):
        nle2 |= R1[k]
    safe_u2 = sum(1 for k in bits(u2) if not (R1[k] & empty2))
    safe_e2 = sum(1 for k in bits(e2) if not (R1[k] & empty2))
    extra = [float(my_perm), float(en_perm), float(conv_perm), float(hole_exp), float(wall),
             float(pc(u2 & ~nle2)), float(pc(u2 & nle2)), float(safe_u2 - safe_e2)]
    return [pc(e & R1[to]), 1.0 if frm is not None else 0.0,
            1.0 if (frm is not None and (att0 >> frm) & 1) else 0.0,
            float(pc(u & R1[frm])) if frm is not None else 0.0,
            float(pc((u | e) & R1[to])),
            float(best_capture(e1, u1, empty)),
            float(pc(u1 & nle)), float(pc(e1 & nlu)), float(safe_u - safe_e),
            float(pc(cu) - pc(ce)), float(pc(lu & ~le) - pc(le & ~lu)),
            float(pc(u2) - pc(e2))] + extra


def load_cases(n_cases, seed=17):
    rng = np.random.default_rng(seed)
    fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
    clones, jumps = [], []
    for f in [fs[i] for i in rng.choice(len(fs), 12, replace=False)]:
        z = np.load(f)
        pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
        gi = z["globalInputNC"]
        q = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
        emp = 49 - pl[:, 1].sum(1) - pl[:, 2].sum(1)
        st0 = {}
        for i in np.nonzero((gi[:, 0] == 0) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0) & (emp >= 10) & (emp <= 36))[0]:
            st0[(bb(pl[i, 1]), bb(pl[i, 2]))] = (i, int(emp[i]))
        for key, (i, ne) in st0.items():
            top = int(q[i, :49].argmax())
            if not ((key[0] >> top) & 1) and q[i, top] / q[i].sum() > 0.5:
                clones.append((key[0], key[1], (None, top), ne))
        for j in np.nonzero((gi[:, 0] == 1) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0))[0]:
            key = (bb(pl[j, 1]), bb(pl[j, 2]))
            if key in st0 and pl[j, 4].sum() == 1:
                i, ne = st0[key]
                top0 = int(q[i, :49].argmax())
                org = int(np.nonzero(pl[j, 4])[0][0])
                if org == top0 and q[i, top0] / q[i].sum() > 0.5 and q[j, :49].max() / q[j].sum() > 0.5:
                    jumps.append((key[0], key[1], (org, int(q[j, :49].argmax())), ne))
    rng.shuffle(clones)
    rng.shuffle(jumps)
    # keep Kata's real clone/jump mix: ~40% of its decisive moves are clones (stage-0 search mass on clones 0.40)
    nc = min(len(clones), int(n_cases * 0.4))
    nj = min(len(jumps), n_cases - nc)
    return clones[:nc] + jumps[:nj]


def fit(X, ch, l2=1e-3):
    """conditional logit (maximum likelihood, L-BFGS): X list of (n_moves, d) arrays, ch list of chosen index"""
    from scipy.optimize import minimize
    A = np.concatenate(X)
    sizes = np.array([len(x) for x in X])
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    grp = np.repeat(np.arange(len(X)), sizes)
    cr = off + np.array(ch)
    xc = A[cr].sum(0)

    def nll(w):
        s = A @ w
        m = np.maximum.reduceat(s, off)
        e = np.exp(s - m[grp])
        z = np.add.reduceat(e, off)
        p = e / z[grp]
        f = -(s[cr].sum() - (m + np.log(z)).sum()) / len(X) + 0.5 * l2 * w @ w
        g = -(xc - (p[:, None] * A).sum(0)) / len(X) + l2 * w
        return f, g
    r = minimize(nll, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x


def top1(X, ch, w):
    return float(np.mean([int(np.argmax(x @ w) == c) for x, c in zip(X, ch)]))


def topk_scores(scores, ch, k):
    return float(np.mean([int((s > s[c]).sum() < k) for s, c in zip(scores, ch)]))


def tree_ranker(Xtr, Ctr, Xte):
    """nonlinear ceiling of the same features: classifier 'is this Kata's move', features centred per decision"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    A = np.concatenate([x - x.mean(0) for x in Xtr])
    A = np.column_stack([A, np.concatenate(Xtr)])
    yv = np.concatenate([(np.arange(len(x)) == c).astype(int) for x, c in zip(Xtr, Ctr)])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
    m.fit(A, yv)
    return [m.predict_proba(np.column_stack([x - x.mean(0), x]))[:, 1] for x in Xte]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cache = "C:/gitProjects/titanium-ataxx/data/nnue/kata_choice_v2_%d.npz" % n
    import os
    if os.path.exists(cache):
        zc = np.load(cache)
        A, sizes, ch, emp = zc["A"], zc["sizes"], list(zc["ch"]), zc["emp"]
        X = np.split(A, np.cumsum(sizes)[:-1])
        print("loaded cached features: %d decisions" % len(X))
    else:
        cases = load_cases(n)
        print("decisive Kata decisions: %d (clones %d, jumps %d)" % (len(cases), sum(c[2][0] is None for c in cases),
                                                                    sum(c[2][0] is not None for c in cases)), flush=True)
        X, ch, emp = [], [], []
        for k, (u, e, mv, ne) in enumerate(cases):
            ms = moves(u, e)
            if mv not in ms:
                continue
            X.append(np.array([feats(u, e, m) for m in ms], float))
            ch.append(ms.index(mv))
            emp.append(ne)
            if (k + 1) % 500 == 0:
                print("  features %d/%d" % (k + 1, len(cases)), flush=True)
        emp = np.array(emp)
        np.savez(cache, A=np.concatenate(X), sizes=np.array([len(x) for x in X]), ch=np.array(ch), emp=emp)
    # order is clones first then jumps: shuffle decisions before the train/test split
    perm = np.random.default_rng(1).permutation(len(X))
    X, ch, emp = [X[i] for i in perm], [ch[i] for i in perm], emp[perm]
    allx = np.concatenate(X)
    mu, sd = allx.mean(0), allx.std(0) + 1e-9
    Xs = [(x - mu) / sd for x in X]
    nmv = np.mean([len(x) for x in X])
    tr = np.arange(len(X)) < int(len(X) * 0.7)
    Xtr = [x for x, t in zip(Xs, tr) if t]
    Ctr = [c for c, t in zip(ch, tr) if t]
    Xte = [x for x, t in zip(Xs, tr) if not t]
    Cte = [c for c, t in zip(ch, tr) if not t]
    print("mean legal moves per decision %.1f (random top-1 = %.3f)" % (nmv, np.mean([1 / len(x) for x in Xte])))
    j = {f: i for i, f in enumerate(FEATS)}

    def single(col, sign=1):
        return float(np.mean([int(np.argmax(sign * x[:, col] + 1e-6 * x[:, j["is_jump"]] * -1) == c) for x, c in zip(Xte, Cte)]))
    print("\nBASELINE scorers, top-1 agreement with Kata on held-out decisions:")
    print("   max stones converted         %.3f" % single(j["conv"]))
    print("   max material after greedy reply (2-ply)   %.3f" % single(j["mat_2ply"]))
    print("   min my exposed stones after the move      %.3f" % single(j["my_exposed_a"], -1))
    w = fit(Xtr, Ctr)
    full = top1(Xte, Cte, w)
    print("\nCHOICE MODEL (all %d features): top-1 test %.3f, train %.3f" % (len(FEATS), full, top1(Xtr, Ctr, w)))
    lin_scores = [x @ w for x in Xte]
    print("   top-3 %.3f  top-5 %.3f" % (topk_scores(lin_scores, Cte, 3), topk_scores(lin_scores, Cte, 5)))
    w12 = fit([x[:, :12] for x in Xtr], Ctr)
    s12 = [x[:, :12] @ w12 for x in Xte]
    print("   (v1 12 features only: top-1 %.3f top-3 %.3f)" % (topk_scores(s12, Cte, 1), topk_scores(s12, Cte, 3)))
    mat = [x[:, j["mat_2ply"]] + 1e-3 * x[:, j["conv"]] for x in Xte]
    print("   (2-ply material: top-1 %.3f top-3 %.3f)" % (topk_scores(mat, Cte, 1), topk_scores(mat, Cte, 3)))
    ts = tree_ranker(Xtr, Ctr, Xte)
    print("TREE RANKER (same features, nonlinear): top-1 %.3f  top-3 %.3f  top-5 %.3f" % (
        topk_scores(ts, Cte, 1), topk_scores(ts, Cte, 3), topk_scores(ts, Cte, 5)))
    print("weights per natural unit (logit change per +1 of the feature; compare with 'conv' = value of one converted stone):")
    wn = w / sd
    for f in FEATS:
        print("   %-15s %+7.3f   (= %+.2f converted stones)" % (f, wn[j[f]], wn[j[f]] / wn[j["conv"]] if wn[j["conv"]] else np.nan))
    print("\nDROP-ONE (top-1 lost when the feature is removed and the model refitted):")
    for f in FEATS:
        keep = [i for i in range(len(FEATS)) if i != j[f]]
        w2 = fit([x[:, keep] for x in Xtr], Ctr)
        print("   %-15s %+.3f" % (f, top1([x[:, keep] for x in Xte], Cte, w2) - full))
    print("\nBY PHASE (model refitted per band; weights per natural unit, conv-normalised):")
    for lo, hi in ((10, 18), (19, 27), (28, 36)):
        m = (emp >= lo) & (emp <= hi)
        idx = np.nonzero(m)[0]
        if len(idx) < 300:
            continue
        cut = int(len(idx) * 0.7)
        a, b = idx[:cut], idx[cut:]
        wb = fit([Xs[i] for i in a], [ch[i] for i in a])
        acc = top1([Xs[i] for i in b], [ch[i] for i in b], wb)
        wbn = wb / sd
        print("   empties %d-%d (n=%d) top-1 %.3f | " % (lo, hi, len(idx), acc) +
              " ".join("%s %+.2f" % (f, wbn[j[f]] / wbn[j["conv"]]) for f in FEATS if f != "conv"))


if __name__ == "__main__":
    main()
