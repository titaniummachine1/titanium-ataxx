"""Dataset v3: Kata-first targets, every row kept, every component kept (so lambda etc. stay tunable at train time).

Reads  data/nnue/final/final_{train,val,test}_v2.npz            (read-only; same rows, same order -> sidecars stay valid)
       data/nnue/final/kata_labels_{split}_nopol.npz              (restore_kata_labels.py --no-policy --out-suffix _nopol)
Writes data/nnue/final/final_{split}_v3.npz = ALL v2 columns unchanged + new columns:

  teacher_p3   float32  win-prob target for the side to move (the "teacher" half of the lambda mix)
                          kind 1  KataGo rows : mix of KataGo's own 200-visit search value (TD horizons, --horizons)
                          kind 2  own-game rows: sigmoid(clip(titanium_cp, +-1500) / 400)  (Titanium 5k search, Stockfish-style)
                          kind 3  terminal    : hard 0/1 game result read off the board (full board / a side wiped out)
                          kind 4  own-game rows relabeled by Moonbird (--moon-dir): sigmoid(cp / s_band), s_band fitted
                                  against KataGo TD h1 per empties band (calibrate_moonbird.py) -> same scale as kind 1
  teacher_cp3  float32  400*logit(teacher_p3), clipped +-1500 (real cp; the v2 `teacher` column is 1000*p_win, NOT cp)
  teacher_kind uint8    1/2/3 as above
  proven       int8     -1 unknown, 0 loss, 1 win for the side to move (terminal now; engine-proven can be written later)
  wsum3        float32  per-row sample weight = v2 wsum (sqrt multiplicity) * --src0-weight on kind-2 rows
  kata_td_win  float16 [N,4]   raw KataGo TD win prob per horizon   (0 for non-Kata rows)
  kata_td_draw float16 [N,4]
  kata_visits  float16  mean root visits of the KataGo search (0 for non-Kata rows)

Train with:  --wdl-loss --teacher-prob-col teacher_p3 --weight-col wsum3 --outcome-col outcome_p --outcome-weight <lambda_outcome>
Nothing here trains anything.  Re-run is idempotent (~minutes) -> change --horizons / --src0-weight freely.
"""
import argparse
import ctypes
import json
import os
import time
import zipfile

import numpy as np

ROOT = r"C:/gitProjects/titanium-ataxx"
FINAL = ROOT + "/data/nnue/final"
SPLITS = ("val", "test", "train")


def popcount64(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


class _MEM(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram_guard(where, min_gb):
    """Abort instead of pushing Windows into swap (the 2026-10-24 freeze/crash)."""
    m = _MEM()
    m.dwLength = ctypes.sizeof(_MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    free = m.ullAvailPhys / 2 ** 30
    if free < min_gb:
        raise SystemExit("RAM GUARD at %s: %.1f GB free < %.1f GB -> abort (nothing written)" % (where, free, min_gb))
    return free


def save_streaming(path, z, new_cols):
    """Write the npz one column at a time: all v2 columns copied from the open source npz, then the new ones.
    Peak RAM = one column, not the whole 3.7 GB file (np.savez(**cols) held everything at once)."""
    tmp = path + ".tmp.npz"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for n in list(z.files) + list(new_cols):
            arr = new_cols[n] if n in new_cols else z[n]
            with zf.open(n + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)
            del arr
    os.replace(tmp, path)


NEED = ("occ0", "occ1", "blk", "stm", "teacher_src", "titanium_cp", "wsum", "outcome_p", "n_outcomes")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,0,0,0",
                    help="weights over the 4 KataGo TD horizons (h1 was best calibrated + best vs result)")
    ap.add_argument("--src0-weight", type=float, default=0.5,
                    help="sample-weight multiplier for own-game rows (Titanium self-labels, single-game outcome)")
    ap.add_argument("--cp-scale", type=float, default=400.0, help="titanium_cp -> p scale (= trainer --wdl-scale)")
    ap.add_argument("--moon-dir", default="", help="e.g. data/nnue/final/moon1000 (relabel_moonbird.py output)")
    ap.add_argument("--moon-weight", type=float, default=1.0, help="sample-weight multiplier for Moonbird rows")
    ap.add_argument("--ow-floor", type=float, default=0.05, help="min outcome weight per row (unbiased anchor)")
    ap.add_argument("--ow-cap", type=float, default=0.5)
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="RAM guard: abort if free RAM drops below this")
    a = ap.parse_args()
    hw = np.array([float(x) for x in a.horizons.split(",")], np.float64)
    hw = hw / hw.sum()
    report = dict(horizons=hw.tolist(), src0_weight=a.src0_weight, cp_scale=a.cp_scale, moon_dir=a.moon_dir,
                  moon_weight=a.moon_weight, splits={})
    cal = None
    if a.moon_dir:
        with open(os.path.join(a.moon_dir, "calibration.json")) as f:
            cal = json.load(f)
        report["moon_calibration"] = cal

    for sp in a.splits.split(","):
        t0 = time.time()
        ram_guard("start " + sp, a.min_free_gb + 3.0)
        z = np.load("%s/final_%s_v2.npz" % (FINAL, sp))
        cols = {n: z[n] for n in NEED}          # only what the targets need; the rest is streamed at save time
        N = len(cols["occ0"])
        k = np.load("%s/kata_labels_%s_nopol.npz" % (FINAL, sp))
        src = cols["teacher_src"]
        stm = cols["stm"]

        kind = np.full(N, 2, np.uint8)
        p3 = np.zeros(N, np.float64)
        td_w = np.zeros((N, 4), np.float16)
        td_d = np.zeros((N, 4), np.float16)
        vis = np.zeros(N, np.float16)

        # ---- own-game / other rows: Titanium 5k search value as a probability
        cp = np.clip(cols["titanium_cp"].astype(np.float64), -1500.0, 1500.0)
        p3[:] = sigmoid(cp / a.cp_scale)
        del cp

        # ---- KataGo rows
        rows = k["row"][k["has"]]
        sel = np.nonzero(k["has"])[0]
        assert (src[rows] == 1).all(), "kata sidecar rows must be src==1"
        W = k["kata_td_win"][sel].astype(np.float64)
        D = k["kata_td_draw"][sel].astype(np.float64)
        pk = ((W + 0.5 * D) * hw[None, :]).sum(1)
        p3[rows] = pk
        kind[rows] = 1
        td_w[rows] = W.astype(np.float16)
        td_d[rows] = D.astype(np.float16)
        vis[rows] = k["kata_visits"][sel].astype(np.float16)
        del W, D, pk
        ram_guard("after kata " + sp, a.min_free_gb)
        n_k_missing = int(((src == 1) & (kind != 1)).sum())

        # ---- Moonbird relabel of own-game rows (kind 2 -> 4), scale per empties band from the Kata calibration
        n_moon_nan = 0
        if cal is not None:
            import glob
            fs = sorted(glob.glob(os.path.join(a.moon_dir, "%s_c[0-9]*.npz" % sp)))
            if fs:
                mr = np.concatenate([np.load(f)["row"] for f in fs])
                mc = np.concatenate([np.load(f)["cp"] for f in fs]).astype(np.float64)
                assert len(np.unique(mr)) == len(mr), "duplicate Moonbird rows"
                assert (kind[mr] == 2).all(), "Moonbird rows must be own-game rows"
                emp_m = 49 - popcount64(cols["occ0"][mr] | cols["occ1"][mr])
                scale = np.full(len(mr), np.nan)
                for b in cal["bands"]:
                    scale[(emp_m >= b["lo"]) & (emp_m <= b["hi"])] = b["scale"]
                assert np.isfinite(scale).all(), "empties outside calibration bands"
                okm = np.isfinite(mc)
                n_moon_nan = int((~okm).sum())
                p3[mr[okm]] = sigmoid(mc[okm] / scale[okm])
                kind[mr[okm]] = 4
                report.setdefault("moon_chunks", {})[sp] = len(fs)

        # ---- terminal positions: board full or a side has no stones -> result is on the board
        o0, o1 = cols["occ0"], cols["occ1"]
        c0, c1 = popcount64(o0), popcount64(o1)
        full = (c0 + c1) == 49
        # blocked squares do not occur in this dataset (blk==0); assert instead of assuming
        assert not cols["blk"].any(), "blocked squares present: terminal rule invalid"
        term = full | (c0 == 0) | (c1 == 0)
        mover = np.where(stm == 0, c0, c1)
        other = np.where(stm == 0, c1, c0)
        res = np.where(mover > other, 1.0, np.where(mover < other, 0.0, 0.5))
        proven = np.full(N, -1, np.int8)
        proven[term & (res == 1.0)] = 1
        proven[term & (res == 0.0)] = 0
        pr = proven >= 0
        agree = float(np.mean(np.abs(cols["outcome_p"][pr] - res[pr]) < 1e-6)) if pr.any() else 1.0
        kata_term_err = float(np.mean(np.abs(p3[pr & (kind == 1)] - res[pr & (kind == 1)]))) if (pr & (kind == 1)).any() else 0.0
        p3[pr] = res[pr]
        kind[pr] = 3

        p3 = np.clip(p3, 0.0, 1.0)
        pc = np.clip(p3, 1e-4, 1 - 1e-4)
        cp3 = np.clip(400.0 * np.log(pc / (1 - pc)), -1500.0, 1500.0)
        w3 = cols["wsum"].astype(np.float64).copy()
        w3[kind == 2] *= a.src0_weight
        w3[kind == 4] *= a.moon_weight

        assert np.isfinite(p3).all() and np.isfinite(w3).all() and (w3 >= 0).all()
        wk = np.where(kind == 1, cols["wsum"].astype(np.float64), 0.0)   # Kata-only variant: own rows weight 0

        # ---- per-row outcome weight (lambda), per teacher kind x empties band, from the data:
        # T = teacher, O = averaged result of n games. E(T-O)^2 = teacherErr + VarO, VarO ~ T(1-T)/n.
        # Best blend weight on O = teacherErr / (teacherErr + VarO), floored (anchor) and capped.
        O = cols["outcome_p"].astype(np.float64)
        nO = np.maximum(cols["n_outcomes"], 1).astype(np.float64)
        wO = cols["wsum"].astype(np.float64)
        emp_all = 49 - (c0 + c1)
        ow_row = np.full(N, a.ow_floor, np.float64)
        ow_tab = []
        for c in (1, 2, 4):
            for lo, hi in ((0, 8), (9, 15), (16, 24), (25, 35), (36, 49)):
                m = (kind == c) & (emp_all >= lo) & (emp_all <= hi)
                if m.sum() < 1000:
                    continue
                d2 = np.average((p3[m] - O[m]) ** 2, weights=wO[m])
                vo = np.average(p3[m] * (1 - p3[m]) / nO[m], weights=wO[m])
                et = max(d2 - vo, 0.0)
                ow = min(max(et / max(et + vo, 1e-9), a.ow_floor), a.ow_cap)
                ow_row[m] = ow
                ow_tab.append(dict(kind=c, lo=lo, hi=hi, n=int(m.sum()), d2=float(d2), var_o=float(vo), ow=float(ow)))
        del O, nO, wO, emp_all, mover, other, res, term, full
        ram_guard("before save " + sp, a.min_free_gb)
        new = dict(teacher_p3=p3.astype(np.float32), teacher_cp3=cp3.astype(np.float32), teacher_kind=kind,
                   proven=proven, wsum3=w3.astype(np.float32), wsum_kataonly=wk.astype(np.float32),
                   ow_row=ow_row.astype(np.float32), kata_td_win=td_w, kata_td_draw=td_d, kata_visits=vis)
        path = "%s/final_%s%s.npz" % (FINAL, sp, a.suffix)
        save_streaming(path, z, new)
        del new, cp3, wk

        # ---- audit numbers
        r = dict(rows=N, kata_rows=int((kind == 1).sum()), own_rows=int((kind == 2).sum()),
                 moon_rows=int((kind == 4).sum()), moon_nan_fallback=n_moon_nan,
                 terminal_rows=int(pr.sum()), src1_without_kata=n_k_missing,
                 terminal_outcome_agree=agree, kata_vs_terminal_mean_abs_err=kata_term_err,
                 p3_mean=float(p3.mean()),
                 p3_std_by_kind={int(c): float(p3[kind == c].std()) for c in (1, 2, 3, 4) if (kind == c).any()},
                 corr_p3_outcome_by_kind={int(c): float(np.corrcoef(p3[kind == c], cols["outcome_p"][kind == c])[0, 1])
                                          for c in (1, 2, 4) if (kind == c).any()},
                 visits_lt100_frac=float((vis[kind == 1] < 100).mean()) if (kind == 1).any() else 0.0,
                 weight_share={int(c): float(w3[kind == c].sum() / w3.sum()) for c in (1, 2, 3, 4) if (kind == c).any()},
                 ow_table=ow_tab, bytes=os.path.getsize(path), secs=round(time.time() - t0))
        report["splits"][sp] = r
        print(sp, json.dumps(r), flush=True)
        del cols, z, k

    with open("%s/final%s_report.json" % (FINAL, a.suffix), "w") as f:
        json.dump(report, f, indent=1)
    print("wrote report")


if __name__ == "__main__":
    main()
