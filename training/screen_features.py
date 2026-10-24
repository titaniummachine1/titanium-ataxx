"""Feature screen: what decides how good a position is?  READ-ONLY, no net training, no engine changes.

Target: KataGo's own 200-visit TD value (win prob, side-to-move relative) on the 100k stratified calibration rows
(same rows as probe_concepts.py). Every idea from the strong engines is turned into cheap whole-board bit-operation
features (side-to-move relative) and grouped:
  material  : stone difference, empties, parity
  reach     : clone / jump reach, territory only one side reaches, stuck flags, number of moves
  attack    : best capture each side, attack sum, 2-ply exchange each side
  safety    : safe stones, exposed stones, frontier stones by empty-neighbour count, corner/edge stones, holes
  structure : empty regions (count, odd regions, largest)
  moonbird  : Moonbird 1.1.0's own three table families (pairs, env, mob) + turn bonus, exact re-implementation
Models are fitted on the KataGo logit and scored by MSE in win-probability space on held-out rows, per empties band:
  linear (what an HCE could be), gradient-boosted trees (nonlinear ceiling of the SAME features), and the other engines'
  own values (Moonbird eval, Ataxx_CPP value, Titanium 5k search) after a scale refit, for comparison.
Questions answered: which group removes the most error, which are redundant with each other, how far cheap features get vs
what only depth gives, and where Moonbird's tables add something the hand features do not.
Usage: python training/screen_features.py [n_rows]
"""
import glob
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import analyze_moonbird as AM  # noqa: E402
import probe_concepts as P  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
FULL = P.FULL
pc = P.pc
R1, R2 = P.R1, P.R2
BANDS = [(9, 15), (16, 24), (25, 35), (36, 49)]


def log(*a):
    print(*a, flush=True)


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def mask_of(squares):
    m = 0
    for s in squares:
        m |= 1 << s
    return np.uint64(m)


EDGE = mask_of([s for s in range(49) if s // 7 in (0, 6) or s % 7 in (0, 6)])
CORNER = mask_of([0, 6, 42, 48])


def net2_best(us, en):
    """best 2-ply exchange over clone squares for the mover `us`: 1 + converted - enemy's best recapture. -9 if no clone."""
    n = len(us)
    empty = ~(us | en) & FULL
    clone = P.spread(us, R1) & empty
    best = np.full(n, -9, np.int64)
    good = np.zeros(n, np.int64)
    for s in range(49):
        bit = np.uint64(1) << np.uint64(s)
        rows = np.nonzero((clone & bit) != 0)[0]
        if not len(rows):
            continue
        conv = en[rows] & R1[s]
        u2 = us[rows] | bit | conv
        e2 = en[rows] & ~conv
        emp2 = empty[rows] & ~bit
        le2 = (P.spread(e2, R1) | P.spread(e2, R2)) & emp2
        rec = np.zeros(len(rows), np.int64)
        for t in range(49):
            hit = ((le2 >> np.uint64(t)) & np.uint64(1)) == 1
            rec = np.maximum(rec, np.where(hit, pc(u2 & R1[t]).astype(np.int64), 0))
        v = 1 + pc(conv).astype(np.int64) - rec
        best[rows] = np.maximum(best[rows], v)
        good[rows] += (v >= 0)
    return best, good


def components_row(empty_bits):
    seen = np.zeros(49, bool)
    sizes = []
    for s in range(49):
        if empty_bits[s] and not seen[s]:
            st, k = [s], 0
            seen[s] = True
            while st:
                a = st.pop()
                k += 1
                r, c = divmod(a, 7)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < 7 and 0 <= cc < 7:
                            t = rr * 7 + cc
                            if empty_bits[t] and not seen[t]:
                                seen[t] = True
                                st.append(t)
            sizes.append(k)
    return sizes


def build_features(us, en):
    n = len(us)
    occ = us | en
    empty = ~occ & FULL
    C = P.concepts(us, en)
    clone_u, clone_e = P.spread(us, R1) & empty, P.spread(en, R1) & empty
    land_u = clone_u | (P.spread(us, R2) & empty)
    land_e = clone_e | (P.spread(en, R2) & empty)
    jmp_u = np.zeros(n, np.int64)
    jmp_e = np.zeros(n, np.int64)
    fr_u = np.zeros((n, 4), np.int64)       # my stones by empty neighbours: 0, 1-2, 3-4, 5+
    fr_e = np.zeros((n, 4), np.int64)
    holes = np.zeros(n, np.int64)           # empty squares with >= 6 filled neighbours
    holes_me = np.zeros(n, np.int64)        # ... that only I can reach
    holes_en = np.zeros(n, np.int64)
    for s in range(49):
        bit = np.uint64(1) << np.uint64(s)
        e_nb = pc(empty & R1[s]).astype(np.int64)
        for own, jmp, fr in ((us, jmp_u, fr_u), (en, jmp_e, fr_e)):
            has = (own & bit) != 0
            jmp += np.where(has, pc(R2[s] & empty).astype(np.int64), 0)
            for b, (lo, hi) in enumerate(((0, 0), (1, 2), (3, 4), (5, 8))):
                fr[:, b] += (has & (e_nb >= lo) & (e_nb <= hi))
        is_empty = (empty & bit) != 0
        filled_nb = pc(occ & R1[s]).astype(np.int64)
        # border squares have fewer than 8 neighbours: count them as full (border behaves like a wall)
        n_nb = bin(int(R1[s])).count("1")
        wall = 8 - n_nb
        h = is_empty & (filled_nb + wall >= 6)
        holes += h
        holes_me += h & ((land_u & bit) != 0) & ((land_e & bit) == 0)
        holes_en += h & ((land_e & bit) != 0) & ((land_u & bit) == 0)
    n2_me, good_me = net2_best(us, en)
    n2_en, good_en = net2_best(en, us)
    comp = [components_row(((empty[i] >> np.arange(49, dtype=np.uint64)) & np.uint64(1)).astype(bool)) for i in range(n)]
    F = {}
    F["material"] = C["material"]
    F["empties"] = pc(empty).astype(np.int64)
    F["parity"] = C["parity"]
    F["clone_reach"] = C["clone_reach"]
    F["jump_reach"] = C["jump_reach"]
    F["only_reach"] = C["only_reach"]
    F["stuck_me"] = C["stuck_me"]
    F["stuck_opp"] = C["stuck_opp"]
    F["moves_diff"] = (pc(clone_u).astype(np.int64) + jmp_u) - (pc(clone_e).astype(np.int64) + jmp_e)
    F["jump_moves_diff"] = jmp_u - jmp_e
    F["best_cap_me"] = C["best_cap_me"]
    F["best_cap_opp"] = C["best_cap_opp"]
    F["best_clone_cap_me"] = C["best_clone_cap_me"]
    F["attack_sum"] = C["attack_sum"]
    F["net2_me"] = n2_me
    F["net2_opp"] = n2_en
    F["good_clones_me"] = good_me
    F["good_clones_opp"] = good_en
    F["safe"] = C["safe"]
    F["exposed"] = C["exposed"]
    for b, nm in enumerate(("fr0", "fr1_2", "fr3_4", "fr5p")):
        F["stones_" + nm + "_diff"] = fr_u[:, b] - fr_e[:, b]
    F["edge_diff"] = pc(us & EDGE).astype(np.int64) - pc(en & EDGE).astype(np.int64)
    F["corner_diff"] = pc(us & CORNER).astype(np.int64) - pc(en & CORNER).astype(np.int64)
    F["holes"] = holes
    F["holes_me"] = holes_me
    F["holes_opp"] = holes_en
    F["n_regions"] = np.array([len(c) for c in comp])
    F["odd_regions"] = np.array([sum(1 for x in c if x % 2) for c in comp])
    F["largest_region"] = np.array([max(c) if c else 0 for c in comp])
    M = AM.families(AM.to_moon(us), AM.to_moon(en))
    for k in ("pairs", "env", "mob"):
        F["moon_" + k] = M[k]
    F["moon_turn"] = M["turn"]
    return {k: np.asarray(v, np.float64) for k, v in F.items()}


GROUPS = {
    "material": ["material", "empties", "parity"],
    "reach": ["clone_reach", "jump_reach", "only_reach", "stuck_me", "stuck_opp", "moves_diff", "jump_moves_diff"],
    "attack": ["best_cap_me", "best_cap_opp", "best_clone_cap_me", "attack_sum", "net2_me", "net2_opp", "good_clones_me", "good_clones_opp"],
    "safety": ["safe", "exposed", "stones_fr0_diff", "stones_fr1_2_diff", "stones_fr3_4_diff", "stones_fr5p_diff", "edge_diff",
               "corner_diff", "holes", "holes_me", "holes_opp"],
    "structure": ["n_regions", "odd_regions", "largest_region"],
    "moonbird": ["moon_pairs", "moon_env", "moon_mob", "moon_turn"],
}
HAND = ["material", "reach", "attack", "safety", "structure"]


def cols(F, groups):
    names = [k for g in groups for k in GROUPS[g]]
    return np.column_stack([F[k] for k in names]), names


def lin_fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    B = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ytr)
    return B @ w


def gbm_fit_predict(Xtr, ytr, Xte):
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def onehot_tables(F, names, rows_fit):
    """HCE knob tables: every feature value (clipped to its 1-99% range, integer bins, max 17 bins) gets its own knob"""
    cols, spec = [], []
    for k in names:
        v = F[k]
        if np.allclose(v, np.round(v)) and np.ptp(v[rows_fit]) <= 40:
            lo, hi = np.percentile(v[rows_fit], [1, 99])
            lo, hi = int(np.floor(lo)), int(np.ceil(hi))
            if hi - lo > 16:
                edges = np.linspace(lo, hi, 17)
                b = np.clip(np.digitize(v, edges[1:-1]), 0, 15)
                nb = 16
            else:
                b = np.clip(np.round(v).astype(int), lo, hi) - lo
                nb = hi - lo + 1
        else:                                          # continuous (Moonbird families): 16 quantile bins
            edges = np.percentile(v[rows_fit], np.linspace(0, 100, 17))
            b = np.clip(np.digitize(v, edges[1:-1]), 0, 15)
            nb = 16
        oh = np.zeros((len(v), nb), np.float32)
        oh[np.arange(len(v)), b] = 1
        cols.append(oh[:, 1:])                        # drop one bin per feature (intercept carries it)
        spec.append((k, nb))
    return np.concatenate(cols, 1), spec


def mse_p(pred_logit, p):
    return float(((sig(pred_logit) - p) ** 2).mean())


def scale_fit_mse(x_tr, p_tr, x_te, p_te):
    best = None
    for s in np.exp(np.linspace(np.log(0.05), np.log(10), 80)):
        e = ((sig(x_tr * s) - p_tr) ** 2).mean()
        if best is None or e < best[0]:
            best = (e, s)
    return float(((sig(x_te * best[1]) - p_te) ** 2).mean())


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    cache = ROOT + "/data/nnue/screen_feats_%d.npz" % n_rows
    fs = sorted(glob.glob(ROOT + "/data/nnue/final/moon1000/calib_c*.npz"))
    row = np.concatenate([np.load(f)["row"] for f in fs])
    rng = np.random.default_rng(3)
    row = np.sort(rng.choice(row, min(n_rows, len(row)), replace=False))
    z = np.load(ROOT + "/data/nnue/final/final_train_v3.npz")
    blk = z["blk"][row]
    print("rows with blocked squares dropped (features treat blocked as empty): %d of %d" % ((blk != 0).sum(), len(row)))
    row = row[blk == 0]
    stm, o0, o1 = z["stm"][row], z["occ0"][row], z["occ1"][row]
    us, en = np.where(stm == 0, o0, o1), np.where(stm == 0, o1, o0)
    p = z["kata_td_win"][row, 0].astype(np.float64) + 0.5 * z["kata_td_draw"][row, 0].astype(np.float64)
    ataxx = z["ataxx_p_win"][row].astype(np.float64) + 0.5 * z["ataxx_p_draw"][row].astype(np.float64)
    ti5k = np.clip(z["titanium_cp"][row].astype(np.float64), -1500, 1500) / 400.0
    emp = 49 - pc(us | en).astype(np.int64)
    log("rows", len(row), "mean Kata win prob %.3f, std %.3f" % (p.mean(), p.std()))
    t0 = time.time()
    if os.path.exists(cache):
        zz = np.load(cache)
        F = {k: zz[k] for k in zz.files}
    else:
        F = build_features(us, en)
        np.savez(cache, **F)
    log("features built (%.0fs): %d" % (time.time() - t0, len(F)))
    y = logit(p)
    allg = list(GROUPS)
    idx = np.arange(len(row))
    rng2 = np.random.default_rng(4)
    test = np.zeros(len(row), bool)
    test[rng2.choice(idx, len(idx) // 4, replace=False)] = True
    # ---- sanity: single feature rank correlations with Kata's value (all rows)
    log("\nSingle-feature |correlation| with KataGo's value (logit), all rows, top 20:")
    cs = sorted(((abs(np.corrcoef(F[k], y)[0, 1]) if F[k].std() > 0 else 0.0, k) for k in F), reverse=True)
    log("  " + ", ".join("%s %.2f" % (k, c) for c, k in cs[:20]))
    raw = np.concatenate([((us[:, None] >> np.arange(49, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float32),
                          ((en[:, None] >> np.arange(49, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float32)], 1)
    for lo, hi in BANDS:
        m = (emp >= lo) & (emp <= hi)
        tr, te = m & ~test, m & test
        log("\n================ empties %d-%d: train %d test %d | const-model MSE %.4f ================" % (lo, hi, tr.sum(), te.sum(),
                                                                                                         float(((p[te] - p[tr].mean()) ** 2).mean())))
        res = {}
        # other engines' own values
        mtot = (F["moon_pairs"] + F["moon_env"] + F["moon_mob"] + F["moon_turn"]) / AM.K * np.log(10)   # Moonbird units -> logit
        res["Moonbird eval (scale refit)"] = scale_fit_mse(mtot[tr], p[tr], mtot[te], p[te])
        res["Ataxx_CPP value (as is)"] = float(((ataxx[te] - p[te]) ** 2).mean())
        res["Titanium 5k search (scale refit)"] = scale_fit_mse(ti5k[tr], p[tr], ti5k[te], p[te])
        # linear
        for nm, gs in (("material only", ["material"]), ("HAND all (linear)", HAND), ("HAND + Moonbird (linear)", HAND + ["moonbird"]),
                       ("Moonbird families only (linear)", ["moonbird"] + ["material"])):
            X, _ = cols(F, gs)
            res[nm] = mse_p(lin_fit_predict(X[tr], y[tr], X[te]), p[te])
        # HCE knob tables (one knob per feature value, this phase band only): a pure lookup-table HCE
        for nm, gs in (("HCE knob tables, HAND", HAND), ("HCE knob tables, HAND + Moonbird", HAND + ["moonbird"])):
            names = [k for g in gs for k in GROUPS[g]]
            OH, _ = onehot_tables(F, names, tr)
            res[nm] = mse_p(lin_fit_predict(OH[tr], y[tr], OH[te]), p[te])
        # nonlinear
        for nm, gs in (("HAND all (trees)", HAND), ("HAND + Moonbird (trees)", HAND + ["moonbird"])):
            X, _ = cols(F, gs)
            res[nm] = mse_p(gbm_fit_predict(X[tr], y[tr], X[te]), p[te])
        res["raw 98 squares (trees)"] = mse_p(gbm_fit_predict(raw[tr], y[tr], raw[te]), p[te])
        Xa, _ = cols(F, allg)
        res["raw squares + ALL features (trees)"] = mse_p(gbm_fit_predict(np.column_stack([raw, Xa])[tr], y[tr], np.column_stack([raw, Xa])[te]), p[te])
        log("MSE vs Kata win prob on held-out rows (lower = better):")
        for k, v in res.items():
            log("   %-36s %.4f" % (k, v))
        # forward selection of groups (linear, then trees)
        for kind, fp in (("linear", lin_fit_predict), ("trees", gbm_fit_predict)):
            chosen = ["material"]
            Xc, _ = cols(F, chosen)
            cur = mse_p(fp(Xc[tr], y[tr], Xc[te]), p[te])
            line = "   greedy group order (%s): material %.4f" % (kind, cur)
            rest = [g for g in allg if g != "material"]
            while rest:
                sc = []
                for g in rest:
                    X, _ = cols(F, chosen + [g])
                    sc.append((mse_p(fp(X[tr], y[tr], X[te]), p[te]), g))
                sc.sort()
                cur = sc[0][0]
                chosen.append(sc[0][1])
                rest.remove(sc[0][1])
                line += " -> +%s %.4f" % (sc[0][1], cur)
            log(line)
        # drop-one-group from the full tree model
        Xall, _ = cols(F, allg)
        full = mse_p(gbm_fit_predict(Xall[tr], y[tr], Xall[te]), p[te])
        line = "   drop-one-group from ALL (trees; base %.4f): " % full
        for g in allg:
            X, _ = cols(F, [h for h in allg if h != g])
            line += "%s %+.4f  " % (g, mse_p(gbm_fit_predict(X[tr], y[tr], X[te]), p[te]) - full)
        log(line)


if __name__ == "__main__":
    main()
