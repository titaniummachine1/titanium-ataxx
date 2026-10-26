"""Cheap eval (S57) vs NNUE gap study. Input: data/nnerr/out.txt (empties, stm piece diff, nmoves, s57, nnue)."""
import numpy as np
d = np.loadtxt("data/nnerr/out.txt")
emp, pd_, nm, s57, nn = d.T
e = nn - s57
a = np.abs(e)
q = lambda x, ps=(50, 75, 90, 95, 99): " ".join("p%d=%.0f" % (p, np.percentile(x, p)) for p in ps)
print("N", len(e), "| gap nnue-s57: mean %.1f sd %.1f | |gap| %s | max %.0f" % (e.mean(), e.std(), q(a), a.max()))
print("corr(s57,nnue) %.4f ; sd s57 %.0f sd nnue %.0f" % (np.corrcoef(s57, nn)[0, 1], s57.std(), nn.std()))
print("\nby empties bucket: n | sd(gap) | |gap| p50 p90 p99")
for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30), (30, 35), (35, 42), (42, 50)]:
    m = (emp >= lo) & (emp < hi)
    if m.sum() > 50:
        print("  %2d-%2d %6d | %5.0f | %4.0f %4.0f %5.0f" % (lo, hi, m.sum(), e[m].std(), *np.percentile(a[m], [50, 90, 99])))
print("\nby |s57| bucket: n | sd(gap) | |gap| p50 p90 p99 | frac sign(nnue)!=sign(s57)")
for lo, hi in [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1500), (1500, 99999)]:
    m = (np.abs(s57) >= lo) & (np.abs(s57) < hi)
    if m.sum() > 50:
        print("  %4d-%5d %6d | %5.0f | %4.0f %4.0f %5.0f | %.3f" % (lo, hi, m.sum(), e[m].std(), *np.percentile(a[m], [50, 90, 99]),
              (np.sign(nn[m]) != np.sign(s57[m])).mean()))
# predictability of |gap| from free features
try:
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    X = np.c_[emp, pd_, nm, s57, np.abs(s57), emp * np.abs(pd_)]
    n = len(e); tr = np.arange(n) % 5 != 0; te = ~tr
    out = {}
    for qq in (0.5, 0.9, 0.99):
        m = H(loss="quantile", quantile=qq, max_iter=150).fit(X[tr], a[tr])
        p = m.predict(X[te]); out[qq] = p
        print("quantile %.2f model: held-out coverage %.3f (target %.2f), mean predicted %.0f vs overall q %.0f" % (qq, (a[te] <= p).mean(), qq, p.mean(), np.percentile(a[tr], 100 * qq)))
    r = H(max_iter=150).fit(X[tr], a[tr]); p = r.predict(X[te])
    print("R2 of predicting |gap| from free features: %.3f" % (1 - ((a[te] - p) ** 2).sum() / ((a[te] - a[te].mean()) ** 2).sum()))
    # skip rule: given a window threshold T at distance dist from s57, skip when dist > c*bound(99%)
    b99 = out[0.99]; s, nt, aa = s57[te], nn[te], a[te]
    print("\nlazy rule: skip net if dist(s57,T) > bound99(features); share of evals skippable vs wrong-side rate")
    rng = np.random.default_rng(1)
    for scale in (1.0, 0.75, 0.5, 0.35):
        skip_tot = wrong = tot = 0
        for T in (-300, -100, 0, 100, 300):
            dist = np.abs(s - T); sk = dist > scale * b99
            wrong += (sk & (np.sign(nt - T) != np.sign(s - T))).sum(); skip_tot += sk.sum(); tot += len(s)
        print("  bound x%.2f: skippable %.1f%% | skipped-but-wrong-side %.3f%% of all evals" % (scale, 100 * skip_tot / tot, 100 * wrong / tot))
    # fixed margin baseline (no model)
    for M in (100, 200, 300, 500):
        skip_tot = wrong = tot = 0
        for T in (-300, -100, 0, 100, 300):
            dist = np.abs(s - T); sk = dist > M
            wrong += (sk & (np.sign(nt - T) != np.sign(s - T))).sum(); skip_tot += sk.sum(); tot += len(s)
        print("  fixed margin %d cp: skippable %.1f%% | wrong-side %.3f%%" % (M, 100 * skip_tot / tot, 100 * wrong / tot))
except ImportError:
    print("no sklearn")
