import json
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
FINAL = ROOT + r"\data\nnue\final"

FITS = {
    0: (0.001, -0.5),   # ataxx raw q*1000
    2: (0.003, 0.0),    # deep cp
    3: (0.001, 0.0),    # engine cp
}


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def main():
    t0 = time.time()
    with open(FINAL + "\\teacher_calibration.json", encoding="utf-8") as f:
        cal = json.load(f)
    for split in ("train", "val", "test"):
        path = FINAL + "\\final_%s.npz" % split
        z = np.load(path, mmap_mode="r")
        d = {k: np.asarray(z[k]) for k in z.files}
        del z
        src = d["teacher_src"].astype(np.int64)
        tp = np.empty(len(src), dtype=np.float32)
        for s, (a, b) in FITS.items():
            m = src == s
            tp[m] = sig(a * d["teacher"][m].astype(np.float64) + b).astype(np.float32)
        m = src == 1
        tp[m] = d["outcome_p"][m].astype(np.float32)
        d["teacher_p"] = tp
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **d)
        del d
        import os
        os.replace(tmp, path)
        print(split, "teacher_p mean %.4f min %.4f max %.4f (%.0fs)" % (
            float(tp.mean()), float(tp.min()), float(tp.max()), time.time() - t0),
            flush=True)
    print("elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
