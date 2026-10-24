"""Restore KataGo's own search-based labels onto the v2 dataset rows (teacher_src == 1).

Why: the v1/v2 extraction kept only the final game result (col 0/2 of globalTargetsNC) from the raw KataGo
self-play files. The raw files also hold, per position, from a ~200-visit search:
  * TD-value targets at 4 horizons (win/loss/draw triples at cols 4-6, 8-10, 12-14, 16-18)
  * root visit count (col 60)
  * the full MCTS visit distribution as policy target (policyTargetsNCMove[:, 0, :], 50 slots)
Measured on 575,637 matched positions: Kata TD-h1 predicts the real result with corr 0.72 / MSE 0.115;
the Ataxx raw relabel only 0.44 / 0.188 (docs/AXNET_PROBE.md section 8).

This script is read-only on the dataset: it writes NEW sidecar files aligned to final_{train,val,test}_v2.npz
rows, nothing is overwritten. Positions are keyed by (occ0, occ1) with stm == 0 (mover = occ0), as in
convert_katago_ataxx.py. Duplicates in the raw data are averaged (same as outcome_p aggregation).

Usage:  python training/restore_kata_labels.py [--max-files N] [--no-policy]
Out:    data/nnue/final/kata_labels_{train,val,test}.npz
"""
import argparse
import glob
import os
import time

import numpy as np

ROOT = r"C:/gitProjects/titanium-ataxx"
RAW = ROOT + "/data/datasets/KataGo_Ataxx_202604"
FINAL = ROOT + "/data/nnue/final"
BW = (1 << np.arange(49, dtype=np.uint64))
SPLITS = ("train", "val", "test")
# globalTargetsNC columns kept: outcome win/draw/none, 4x TD (win, draw), root visits
COLS = dict(win=0, draw=2, none=3, td_win=[4, 8, 12, 16], td_draw=[6, 10, 14, 18], visits=60)
M1 = np.uint64(0x9E3779B97F4A7C15)
M2 = np.uint64(0xC2B2AE3D27D4EB4F)


def mix(a, b):
    return (a * M1) ^ (((b << np.uint64(23)) | (b >> np.uint64(41))) * M2)


def decode(packed):
    bits = np.unpackbits(packed, axis=2)[:, :, :49]
    return (bits[:, 1].astype(np.uint64) * BW).sum(1), (bits[:, 2].astype(np.uint64) * BW).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    t0 = time.time()

    # ---- keys of all src==1 rows across splits (train first), remembering (split, row)
    occ0, occ1, split_id, row_id = [], [], [], []
    meta = {}
    for si, sp in enumerate(SPLITS):
        z = np.load("%s/final_%s_v2.npz" % (FINAL, sp))
        src = z["teacher_src"]
        idx = np.nonzero(src == 1)[0]
        assert (z["stm"][idx] == 0).all(), "src1 rows must all have stm==0"
        occ0.append(z["occ0"][idx]); occ1.append(z["occ1"][idx])
        split_id.append(np.full(len(idx), si, np.uint8)); row_id.append(idx.astype(np.int64))
        meta[sp] = dict(n_rows=len(src), outcome_p=z["outcome_p"][idx], n_outcomes=z["n_outcomes"][idx])
        print("split %-5s src1 rows %d" % (sp, len(idx)), flush=True)
    occ0 = np.concatenate(occ0); occ1 = np.concatenate(occ1)
    split_id = np.concatenate(split_id); row_id = np.concatenate(row_id)
    N = len(occ0)
    H = mix(occ0, occ1)
    order = np.argsort(H)
    Hs = H[order]
    print("keys built: %d rows, %.0fs" % (N, time.time() - t0), flush=True)

    cnt = np.zeros(N, np.float32)
    out_s = np.zeros(N, np.float32)
    vis_s = np.zeros(N, np.float32)
    tdw = np.zeros((N, 4), np.float32)
    tdd = np.zeros((N, 4), np.float32)
    pol = None if a.no_policy else np.zeros((N, 50), np.float32)

    files = sorted(glob.glob(RAW + "/*/tdata/*.npz"))
    if a.max_files:
        files = files[:a.max_files]
    print("raw files", len(files), flush=True)
    matched = total = 0
    for fi, f in enumerate(files):
        try:
            z = np.load(f)
            own, opp = decode(z["binaryInputNCHWPacked"])
            G = z["globalTargetsNC"].astype(np.float32)
            P = z["policyTargetsNCMove"][:, 0, :].astype(np.float32) if pol is not None else None
        except Exception as e:  # corrupt/partial file: skip, report
            print("SKIP %s: %s" % (os.path.basename(f), e), flush=True)
            continue
        total += len(own)
        hs = mix(own, opp)
        pos = np.clip(np.searchsorted(Hs, hs), 0, N - 1)
        r = order[pos]
        ok = (Hs[pos] == hs) & (occ0[r] == own) & (occ1[r] == opp)
        if not ok.any():
            continue
        r = r[ok]; g = G[ok]
        matched += len(r)
        np.add.at(cnt, r, 1.0)
        np.add.at(out_s, r, g[:, COLS["win"]] + 0.5 * g[:, COLS["draw"]])
        np.add.at(vis_s, r, g[:, COLS["visits"]])
        for h in range(4):
            np.add.at(tdw[:, h], r, g[:, COLS["td_win"][h]])
            np.add.at(tdd[:, h], r, g[:, COLS["td_draw"][h]])
        if pol is not None:
            p = P[ok]
            p = p / np.maximum(p.sum(1, keepdims=True), 1.0)
            np.add.at(pol, r, p)
        if fi % 50 == 0:
            print("file %d/%d matched %d/%d rows  %.0fs" % (fi, len(files), matched, total, time.time() - t0), flush=True)

    have = cnt > 0
    print("positions with Kata labels: %d / %d (%.2f%%)" % (have.sum(), N, 100 * have.mean()), flush=True)
    c = np.maximum(cnt, 1.0)
    res = dict(
        kata_count=cnt, kata_outcome=out_s / c, kata_visits=vis_s / c,
        kata_td_win=tdw / c[:, None], kata_td_draw=tdd / c[:, None],
    )
    if pol is not None:
        res["kata_policy"] = (pol / np.maximum(pol.sum(1, keepdims=True), 1e-9)).astype(np.float16)

    # ---- integrity: aggregated raw outcome must reproduce v2 outcome_p; count must match n_outcomes
    base = 0
    for si, sp in enumerate(SPLITS):
        sel = np.nonzero(split_id == si)[0]
        sel = sel[np.argsort(row_id[sel])]
        o_p = meta[sp]["outcome_p"]; n_o = meta[sp]["n_outcomes"]
        hv = have[sel]
        d = np.abs(res["kata_outcome"][sel][hv] - o_p[hv])
        print("[%s] rows %d labeled %d | outcome match: mean|diff| %.5f max %.4f | count==n_outcomes on %.1f%%"
              % (sp, len(sel), hv.sum(), d.mean() if hv.any() else -1, d.max() if hv.any() else -1,
                 100 * np.mean(res["kata_count"][sel][hv] == n_o[hv]) if hv.any() else 0), flush=True)
        out = {k: v[sel] for k, v in res.items()}
        out["row"] = row_id[sel]
        out["has"] = hv
        path = "%s/kata_labels_%s%s.npz" % (FINAL, sp, a.out_suffix)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        print("wrote", path, os.path.getsize(path), "bytes", flush=True)
    print("done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
