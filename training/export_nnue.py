"""Export a train_nnue.py checkpoint (.pt) to the engine's TNN1 float32 format (crates/titanium/src/nnue.rs).
Also writes <out>.ref.txt: board line + expected cp for 1000 val positions, for the Rust parity test
(`titanium-cli nneval --nnue <out> --boards <out>.ref.txt`).
Usage: python training/export_nnue.py --pt data/nnue/nnue_pbt/best_base.pt --out data/nnue/nnue_base.tnn [--drop G5]
v2 checkpoints (s57w present) -> TNN2; parity then needs `nneval ... --tuple data/nnue/final_v2_wdl_ep8.tup`.
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nnue as T  # noqa: E402
from ataxx_move import board_str  # noqa: E402

GROUP_BITS = {"G1": 1, "G2": 2, "G3": 4, "G5": 16, "G6": 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="")
    a = ap.parse_args()
    ck = torch.load(a.pt, weights_only=False)
    sd = ck["model"]
    W = sd["ft.weight"].shape[1]
    L1 = sd["l2.weight"].shape[1]
    L2 = sd["out.weight"].shape[1]
    NB = T.NB
    groups = 1 | 2 | 4 | 16 | 32
    for g in filter(None, a.drop.split(",")):
        groups &= ~GROUP_BITS[g.strip()]
    f32 = lambda t: t.detach().cpu().numpy().astype("<f4").ravel().tobytes()
    v2 = "s57w" in sd
    if v2:
        L1 = sd["l1.weight"].shape[0] // NB
        L2 = sd["l2.weight"].shape[0] // NB
        pairwise = sd["l1.weight"].shape[1] == W
    with open(a.out, "wb") as f:
        if v2:
            f.write(b"TNN2" + struct.pack("<7I", T.NUM_FEATS, W, L1, L2, NB, groups, int(pairwise)))
        else:
            f.write(b"TNN1" + struct.pack("<6I", T.NUM_FEATS, W, L1, L2, NB, groups))
        for k in ("ft.weight", "ftb", "psqt.weight", "l1.weight", "l1.bias", "l2.weight", "l2.bias",
                  "out.weight", "out.bias") + (("s57w",) if v2 else ()):
            f.write(f32(sd[k]))
    print("wrote %s (%d bytes) %s width %d l1 %d l2 %d groups %d" % (a.out, os.path.getsize(a.out), "TNN2" if v2 else "TNN1",
                                                                    W, L1, L2, groups))
    # reference evals for the Rust parity test
    model = T.NNUE2(W=W, L1=L1, L2=L2, pairwise=pairwise) if v2 else T.NNUE(W=W, L1=L1, L2=L2)
    model.load_state_dict(sd)
    model.eval()
    keep = None
    if a.drop:
        keep = np.ones(T.NUM_FEATS, bool)
        for g in a.drop.split(","):
            lo, hi = T.GROUPS[g.strip()]
            keep[lo:hi] = False
    va = T.Split("val", "base", keep)
    rows = np.random.default_rng(3).choice(len(va.target), 1000, replace=False)
    z = np.load(T.FINAL + "/final_val_v3.npz")
    ix, off, bk, _, _ = va.batch(rows)
    s57 = va.extra(rows)[2]
    if v2:
        assert float(s57.abs().sum()) > 0, "v2 export needs data/nnue/s57/val.npy (training/build_s57_sidecar.py)"
    with torch.no_grad():
        y = model(ix, off, bk, s57).numpy()
    with open(a.out + ".ref.txt", "w") as f:
        for r, v in zip(rows, y):
            f.write("%s %d\n" % (board_str(int(z["occ0"][r]), int(z["occ1"][r]), 0, int(z["stm"][r])),
                                 int(np.clip(np.round(400 * v), -30000, 30000))))
    print("wrote", a.out + ".ref.txt")


if __name__ == "__main__":
    main()
