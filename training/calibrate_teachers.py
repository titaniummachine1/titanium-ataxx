import json
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
FINAL = ROOT + r"\data\nnue\final"


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def fit_ab(x, y):
    best = (1e9, 0.0, 0.0)
    for a in (1 / 2000.0, 1 / 1000.0, 1 / 400.0, 1 / 200.0, 1 / 100.0, 1 / 50.0):
        for b in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            m = float(np.mean((sig(a * x + b) - y) ** 2))
            if m < best[0]:
                best = (m, a, b)
    mse, a, b = best
    for _ in range(3):
        improved = False
        for da, db in ((a * 0.2, 0.0), (-a * 0.2, 0.0), (0.0, 0.2), (0.0, -0.2)):
            na, nb = a + da, b + db
            m = float(np.mean((sig(na * x + nb) - y) ** 2))
            if m < mse:
                mse, a, b = m, na, nb
                improved = True
        if not improved:
            break
    return {"mse": mse, "a": a, "b": b}


def main():
    t0 = time.time()
    out = {}
    for split in ("train", "val", "test"):
        z = np.load(FINAL + "\\final_%s.npz" % split, mmap_mode="r")
        src = np.asarray(z["teacher_src"])
        tcp = np.asarray(z["teacher"], dtype=np.float64)
        y = np.asarray(z["outcome_p"], dtype=np.float64)
        res = {}
        for sname, s in (("ataxx", 0), ("katago", 1), ("deep", 2), ("engine", 3)):
            m = src == s
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                res[sname] = {"n": 0}
                continue
            if len(idx) > 400000:
                rng = np.random.default_rng(7)
                idx = idx[rng.choice(len(idx), 400000, replace=False)]
            x, yy = tcp[idx], y[idx]
            if sname == "katago":
                res[sname] = {"n": int(m.sum()), "mse": 0.0, "a": None, "b": None,
                              "note": "teacher is outcome itself"}
                continue
            r = fit_ab(x, yy)
            r["n"] = int(m.sum())
            r["mse_sigmoid_x_over_400"] = float(np.mean((sig(x / 400.0) - yy) ** 2))
            res[sname] = r
            print(split, sname, r, flush=True)
        out[split] = res
        del z
    with open(FINAL + "\\teacher_calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("elapsed %.0fs" % (time.time() - t0), flush=True)
    print(json.dumps(out["train"], indent=2))


if __name__ == "__main__":
    main()
