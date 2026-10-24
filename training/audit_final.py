import argparse
import json
import os
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
FINAL = os.path.join(ROOT, "data", "nnue", "final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    OUT = os.path.join(FINAL, "final_audit%s.json" % args.suffix)
    result = {"created_at": time.time(), "suffix": args.suffix,
              "splits": {}, "ok": True, "problems": []}
    total = 0
    for split in ("train", "val", "test"):
        p = os.path.join(FINAL, "final_%s%s.npz" % (split, args.suffix))
        if not os.path.exists(p):
            result["ok"] = False
            result["problems"].append("missing %s" % p)
            continue
        z = np.load(p, mmap_mode="r")
        keys = list(z.keys())
        need = ["occ0", "occ1", "blk", "stm", "out", "outcome_p", "wins",
                "losses", "draws", "n_outcomes", "teacher", "teacher_p", "titanium_cp",
                "sc", "wsum", "teacher_src"]
        miss = [k for k in need if k not in keys]
        if miss:
            result["ok"] = False
            result["problems"].append("%s missing %s" % (split, miss))
        n = len(z["occ0"]) if "occ0" in z else 0
        total += n
        info = {"rows": int(n), "keys": keys}
        for k in ("out", "outcome_p", "teacher", "teacher_p", "titanium_cp", "wins", "losses", "draws"):
            if k in z:
                a = np.asarray(z[k], dtype=np.float64)
                info[k + "_min"] = float(np.nanmin(a)) if len(a) else None
                info[k + "_max"] = float(np.nanmax(a)) if len(a) else None
                info[k + "_nan"] = int(np.isnan(a).sum())
                if info[k + "_nan"]:
                    result["ok"] = False
                    result["problems"].append("%s %s has NaNs" % (split, k))
        result["splits"][split] = info
        if all(k in z for k in ("ataxx_p_win", "ataxx_p_draw", "ataxx_p_loss", "ataxx_ply")):
            pw = np.asarray(z["ataxx_p_win"], dtype=np.float64)
            pdw = np.asarray(z["ataxx_p_draw"], dtype=np.float64)
            pl = np.asarray(z["ataxx_p_loss"], dtype=np.float64)
            rel = ~np.isnan(pw)
            info["relabeled_rows"] = int(rel.sum())
            info["relabeled_p_sum_min"] = float((pw + pdw + pl)[rel].min()) if rel.any() else None
            info["relabeled_p_sum_max"] = float((pw + pdw + pl)[rel].max()) if rel.any() else None
            bad = rel & (np.abs(pw + pdw + pl - 1.0) > 1e-3)
            if bad.any():
                result["ok"] = False
                result["problems"].append("%s %d relabeled rows p_win+p_draw+p_loss != 1" % (split, int(bad.sum())))
            src_mix = {int(k2): int((np.asarray(z["teacher_src"]) == k2).sum())
                       for k2 in np.unique(np.asarray(z["teacher_src"]))}
            info["teacher_src_mix"] = src_mix
            tp = np.asarray(z["teacher_p"], dtype=np.float64)
            if rel.any() and (tp[rel].min() < -1e-6 or tp[rel].max() > 1 + 1e-6):
                result["ok"] = False
                result["problems"].append("%s relabeled teacher_p outside [0,1]" % split)
        del z
    result["total_rows"] = total
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
