"""Pack dataset positions (final_{train,val,test}_v3.npz, in that order) into 25-byte records for the relabel.

Record = occ0 u64 | occ1 u64 | blk u64 | stm u8 (little endian). Writes data/nnue/relabel/positions.bin and rows.npy
(rows.npy[i] = GLOBAL dataset row of packed record i; global row order = train rows, then val, then test).
--subset N keeps N random train rows (seed 7, sorted) PLUS all val and test rows; default = everything.
Reads only the four needed columns, one split at a time (RAM-safe, law 21).
Usage: python training/relabel_pack.py --subset 8000000
"""
import argparse
import os
import zipfile

import numpy as np
import numpy.lib.format as F

ROOT = "C:/gitProjects/titanium-ataxx"
FINAL = ROOT + "/data/nnue/final"
OUT = ROOT + "/data/nnue/relabel"
REC = np.dtype([("o0", "<u8"), ("o1", "<u8"), ("blk", "<u8"), ("stm", "u1")])


def col(z, name):
    with z.open(name + ".npy") as f:
        return F.read_array(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="random train rows to keep (0 = all); val/test always all")
    ap.add_argument("--shuffle", action="store_true", help="write the records in a random order (seed 11), so any prefix "
                    "or any contiguous range of positions.bin is a random sample of the packed rows")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    base, ids = 0, []
    with open(OUT + "/positions.bin", "wb") as fo:
        for sp in ("train", "val", "test"):
            z = zipfile.ZipFile("%s/final_%s_v3.npz" % (FINAL, sp))
            o0, o1, blk, stm = col(z, "occ0"), col(z, "occ1"), col(z, "blk"), col(z, "stm")
            n = len(o0)
            sel = np.arange(n)
            if sp == "train" and 0 < a.subset < n:
                sel = np.sort(np.random.default_rng(7).choice(n, a.subset, replace=False))
            for lo in range(0, len(sel), 2_000_000):
                s = sel[lo:lo + 2_000_000]
                r = np.empty(len(s), REC)
                r["o0"], r["o1"], r["blk"], r["stm"] = o0[s], o1[s], blk[s], stm[s]
                fo.write(r.tobytes())
            ids.append(base + sel)
            print(sp, "dataset rows", n, "packed", len(sel), flush=True)
            base += n
            del o0, o1, blk, stm
    ids = np.concatenate(ids).astype(np.int64)
    if a.shuffle:
        perm = np.random.default_rng(11).permutation(len(ids))
        recs = np.fromfile(OUT + "/positions.bin", dtype=REC)
        recs[perm].tofile(OUT + "/positions.bin")
        del recs
        ids = ids[perm]
    np.save(OUT + "/rows.npy", ids)
    print("packed", len(ids), "records ->", OUT + "/positions.bin")


if __name__ == "__main__":
    main()
