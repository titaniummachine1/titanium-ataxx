"""Pack exp/fut-probe v2 records (64 B: occ0, occ1, blockers as u64; then i32 turn, alpha, child NNUE eval, cached,
new_depth before the futility step, child hand eval, parent static eval, ply, root ply, 0) into training arrays for the
futility small net (D109). Positions are made side-to-move relative: own = stones of the child's side to move.
Train keeps one row per distinct position (first label; the big net is deterministic so repeats agree); eval keeps
every record so the futility simulation sees the real call mix.
Usage: python training/smallnet_pack.py
"""
import glob
import numpy as np

D = "C:/gitProjects/titanium-ataxx/data/nnue/smallnet"
REC = np.dtype([("occ0", "<u8"), ("occ1", "<u8"), ("blk", "<u8"), ("turn", "<i4"), ("alpha", "<i4"), ("nn", "<i4"),
                ("cached", "<i4"), ("nd0", "<i4"), ("hand", "<i4"), ("pst", "<i4"), ("ply", "<i4"), ("rply", "<i4"), ("z", "<i4")])
assert REC.itemsize == 64


def load(files):
    r = np.concatenate([np.fromfile(f, dtype=REC) for f in files])
    own = np.where(r["turn"] == 0, r["occ0"], r["occ1"])
    opp = np.where(r["turn"] == 0, r["occ1"], r["occ0"])
    return r, own, opp


for name, files, dedup in (("train", sorted(glob.glob(D + "/rec_tr*.bin")), True), ("eval", [D + "/rec_ev.bin"], False)):
    r, own, opp = load(files)
    ok = np.abs(r["nn"]) < 20000
    n0 = len(r)
    if dedup:
        key = np.stack([own, opp, r["blk"]], 1)
        _, first = np.unique(key, axis=0, return_index=True)
        keep = np.zeros(len(r), bool)
        keep[first] = True
        ok &= keep
    r, own, opp = r[ok], own[ok], opp[ok]
    np.savez(D + "/sn_%s.npz" % name, own=own, opp=opp, blk=r["blk"], nn=r["nn"], hand=r["hand"], pst=r["pst"],
             alpha=r["alpha"], nd0=r["nd0"], cached=r["cached"])
    print("%s: %d records -> %d rows (%s)" % (name, n0, len(r), "distinct positions" if dedup else "all calls"))
