"""Extract the 10 training columns for Modal upload into data/nnue/modal_up/final_{train,val}_v5.npz.

Columns read by train_nnue.Split:
  occ0, occ1, stm, outcome_p, teacher_kind, kata_td_win, kata_td_draw, titanium_cp, teacher_p3, wsum
"""
import os
import time
import zipfile
import numpy as np
import numpy.lib.format as F

ROOT = "C:/gitProjects/titanium-ataxx"
FINAL = os.path.join(ROOT, "data", "nnue", "final")
MODAL_UP = os.path.join(ROOT, "data", "nnue", "modal_up")
COLS = [
    "occ0", "occ1", "stm", "outcome_p", "teacher_kind",
    "kata_td_win", "kata_td_draw", "titanium_cp", "teacher_p3", "wsum"
]

def main():
    os.makedirs(MODAL_UP, exist_ok=True)
    for sp in ("val", "train"):
        t0 = time.time()
        src_p = os.path.join(FINAL, f"final_{sp}_v5.npz")
        dst_p = os.path.join(MODAL_UP, f"final_{sp}_v5.npz")
        print(f"Reading {src_p} and extracting {len(COLS)} columns...", flush=True)
        data = {}
        with zipfile.ZipFile(src_p) as z:
            for c in COLS:
                with z.open(f"{c}.npy") as f:
                    data[c] = F.read_array(f)
        print(f"Loaded {sp} columns in {time.time() - t0:.1f}s. Saving compressed to {dst_p}...", flush=True)
        t_save = time.time()
        np.savez_compressed(dst_p, **data)
        size_mb = os.path.getsize(dst_p) / 1_048_576.0
        print(f"Wrote {dst_p} ({size_mb:.1f} MB) in {time.time() - t_save:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
