"""Put the relabel (data/nnue/relabel/labels.npz) into the training files as the NEW `titanium_cp`.

    python training/relabel_join.py            # writes final_{train,val,test}_v4.npz, then verifies them
    python training/relabel_join.py --verify   # verification only

final_{split}_v4.npz = every v3 column unchanged EXCEPT
  titanium_cp      <- relabel score (big NNUE c4829350, 5000 nodes, stm-relative like the old column; proven wins/losses stay +-99999-ply, the trainer clips to +-1500)
  titanium_cp_old  <- the previous column (kept for comparison)
  relabel_depth    int16, depth the 5000-node search reached
v3 files are never touched. Streams one column at a time (RAM peak ~1.5 GB), members stored uncompressed like the v3 files.
Row order: global row = train rows, then val, then test; labels.rows[i] is the global row of packed record i (relabel_pack.py).
Verification (all rows, not a sample): every non-label column identical to v3, titanium_cp == the mapped score, the packed position of each row equals the dataset position.
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
    L = np.load(REL + "/labels.npz")
    rows, have, score, depth = L["rows"], L["have"], L["score"], L["depth"]
    assert have.all(), "labels.npz is not complete"
    n = len(rows)
    assert len(np.unique(rows)) == n and rows.min() == 0 and rows.max() == n - 1, "rows.npy is not a permutation"
    inv = np.empty(n, np.int64)          # global dataset row -> packed record index
    inv[rows] = np.arange(n)
    return inv, score, depth


def split_sizes():
    out = []
    for sp in SPLITS:
        with zipfile.ZipFile("%s/final_%s_v3.npz" % (FINAL, sp)) as z:
            out.append(len(read(z, "stm")))
    return out


def join():
    inv, score, depth = load_maps()
    sizes = split_sizes()
    assert sum(sizes) == len(inv), "dataset rows %d != labelled rows %d" % (sum(sizes), len(inv))
    base = 0
    for sp, n in zip(SPLITS, sizes):
        idx = inv[base + np.arange(n)]
        new_cp = score[idx].astype(np.float32)
        new_depth = depth[idx].astype(np.int16)
        src, dst = "%s/final_%s_v3.npz" % (FINAL, sp), "%s/final_%s_v4.npz" % (FINAL, sp)
        with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as zo:
            for m in zi.namelist():
                name = m[:-4]
                a = read(zi, name)
                if name == "titanium_cp":
                    write(zo, "titanium_cp_old", a)
                    write(zo, "titanium_cp", new_cp)
                else:
                    write(zo, name, a)
                del a
            write(zo, "relabel_depth", new_depth)
        print(sp, n, "rows ->", dst, flush=True)
        base += n


def verify():
    inv, score, depth = load_maps()
    P = np.memmap(REL + "/positions.bin", dtype=REC, mode="r")
    sizes = split_sizes()
    base, bad = 0, 0
    for sp, n in zip(SPLITS, sizes):
        with zipfile.ZipFile("%s/final_%s_v3.npz" % (FINAL, sp)) as z3, zipfile.ZipFile("%s/final_%s_v4.npz" % (FINAL, sp)) as z4:
            names3, names4 = [m[:-4] for m in z3.namelist()], [m[:-4] for m in z4.namelist()]
            extra = sorted(set(names4) - set(names3))
            assert sorted(set(names3) - set(names4)) == [], "v4 lost columns"
            assert extra == ["relabel_depth", "titanium_cp_old"], extra
            for name in names3:
                a3, a4 = read(z3, name), read(z4, name)
                if name == "titanium_cp":
                    ok = np.array_equal(a3, read(z4, "titanium_cp_old")) and a4.shape == a3.shape
                    new = a4
                else:
                    ok = a3.shape == a4.shape and a3.dtype == a4.dtype and np.array_equal(a3, a4, equal_nan=a3.dtype.kind == "f")
                if not ok:
                    bad += 1
                    print("MISMATCH", sp, name)
                del a3, a4
            idx = inv[base + np.arange(n)]
            ok = np.array_equal(new, score[idx].astype(np.float32)) and np.array_equal(read(z4, "relabel_depth"), depth[idx].astype(np.int16))
            print(sp, "titanium_cp == mapped relabel score and depth:", ok)
            bad += 0 if ok else 1
            # position of every row == packed record
            r = P[idx]
            for k, col in (("o0", "occ0"), ("o1", "occ1"), ("blk", "blk"), ("stm", "stm")):
                ok = np.array_equal(r[k], read(z4, col))
                print(sp, "position", col, "matches packed record for ALL rows:", ok)
                bad += 0 if ok else 1
            print(sp, "titanium_cp finite:", bool(np.isfinite(new).all()), " min %.0f max %.0f" % (new.min(), new.max()))
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
