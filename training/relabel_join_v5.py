"""Join v5 relabels (data/nnue/relabel/labels_v5.npz) into final_{train,val,test}_v5.npz.

Keeps every column from v4, with:
  titanium_cp      <- new 5k relabel score
  titanium_cp_old  <- the v4 titanium_cp (kept for auditing)
  relabel_depth    <- new 5k relabel depth

Usage:
  python training/relabel_join_v5.py            # creates final_{train,val,test}_v5.npz
  python training/relabel_join_v5.py --verify   # verify existing files
"""
import argparse
import sys
import zipfile

import numpy as np
import numpy.lib.format as F

ROOT = "C:/gitProjects/titanium-ataxx"
FINAL = ROOT + "/data/nnue/final"
REL = ROOT + "/data/nnue/relabel"
SPLITS = ("train", "val", "test")
REC = np.dtype([("o0", "<u8"), ("o1", "<u8"), ("blk", "<u8"), ("stm", "u1")])


def read(z, name):
    with z.open(name + ".npy") as f:
        return F.read_array(f)


def write(z, name, arr):
    with z.open(name + ".npy", "w", force_zip64=True) as f:
        F.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)


def load_maps():
    L = np.load(REL + "/labels_v5.npz")
    rows, have, score, depth = L["rows"], L["have"], L["score"], L["depth"]
    assert have.all(), "labels_v5.npz is not complete"
    n = len(rows)
    assert len(np.unique(rows)) == n and rows.min() == 0 and rows.max() == n - 1, "rows is not a valid permutation"
    inv = np.empty(n, np.int64)          # global dataset row -> packed record index
    inv[rows] = np.arange(n)
    return inv, score, depth


def split_sizes():
    out = []
    for sp in SPLITS:
        with zipfile.ZipFile("%s/final_%s_v4.npz" % (FINAL, sp)) as z:
            out.append(len(read(z, "stm")))
    return out


def join():
    print("Loading labels_v5...", flush=True)
    inv, score, depth = load_maps()
    sizes = split_sizes()
    assert sum(sizes) == len(inv), "dataset rows %d != labelled rows %d" % (sum(sizes), len(inv))
    base = 0
    for sp, n in zip(SPLITS, sizes):
        print(f"Joining {sp} ({n:,} rows)...", flush=True)
        idx = inv[base + np.arange(n)]
        new_cp = score[idx].astype(np.float32)
        new_depth = depth[idx].astype(np.int16)
        src, dst = "%s/final_%s_v4.npz" % (FINAL, sp), "%s/final_%s_v5.npz" % (FINAL, sp)
        with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as zo:
            for m in zi.namelist():
                name = m[:-4]
                a = read(zi, name)
                if name == "titanium_cp":
                    write(zo, "titanium_cp_old", a)
                    write(zo, "titanium_cp", new_cp)
                elif name == "titanium_cp_old":
                    # skip older old column
                    pass
                elif name == "relabel_depth":
                    write(zo, "relabel_depth", new_depth)
                else:
                    write(zo, name, a)
                del a
        print(f"{sp}: {n:,} rows -> {dst}", flush=True)
        base += n
    print("Done joining all splits into v5.", flush=True)


def verify():
    inv, score, depth = load_maps()
    sizes = split_sizes()
    base, bad = 0, 0
    for sp, n in zip(SPLITS, sizes):
        with zipfile.ZipFile("%s/final_%s_v4.npz" % (FINAL, sp)) as z4, zipfile.ZipFile("%s/final_%s_v5.npz" % (FINAL, sp)) as z5:
            names4, names5 = [m[:-4] for m in z4.namelist()], [m[:-4] for m in z5.namelist()]
            for name in names5:
                if name in ("titanium_cp", "titanium_cp_old", "relabel_depth"):
                    continue
                a4, a5 = read(z4, name), read(z5, name)
                ok = a4.shape == a5.shape and a4.dtype == a5.dtype and np.array_equal(a4, a5, equal_nan=a4.dtype.kind == "f")
                if not ok:
                    bad += 1
                    print("MISMATCH", sp, name)
                del a4, a5
            idx = inv[base + np.arange(n)]
            new_cp = read(z5, "titanium_cp")
            new_dep = read(z5, "relabel_depth")
            ok = np.array_equal(new_cp, score[idx].astype(np.float32)) and np.array_equal(new_dep, depth[idx].astype(np.int16))
            print(sp, "titanium_cp and relabel_depth match labels_v5 exactly:", ok)
            bad += 0 if ok else 1
            print(sp, "titanium_cp finite:", bool(np.isfinite(new_cp).all()), " min %.0f max %.0f" % (new_cp.min(), new_cp.max()))
        base += n
    print("VERIFIED OK" if not bad else "VERIFICATION FAILED (%d problems)" % bad)
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if not a.verify:
        join()
    sys.exit(1 if verify() else 0)
