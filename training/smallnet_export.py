"""Export a futility small net (smallnet_train.py) for the engine (D110): magic "TSN1", u32 W, H; then f32 LE arrays
ft[147][W] (feature-major), ftb[W], l1_w[NB*H][2W] (row = bucket*H + i), l1_b[NB*H], out_w[NB][H], out_b[NB],
skip[2][NB] (parent static eval, child hand eval; both in cp/400). Bound nets export ONE head (--tau).
Also writes <out>.ref.txt: own opp blk (hex u64) parent_cp hand_cp expected_cp for 2000 held-out calls (Rust parity).
Usage: python training/smallnet_export.py --pt sn_q32.pt --tau 0.01 --out data/nnue/fut_small.tsn
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import smallnet_train as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--tau", type=float, default=None, help="head of a bound net (omit for an eval net)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
st = torch.load(a.pt, map_location="cpu", weights_only=False)
ar = st["args"]
taus = [float(x) for x in ar["quantiles"].split(",")] if ar.get("quantiles") else []
Q = max(1, len(taus))
q = taus.index(a.tau) if taus else 0
assert ar.get("no_hand") is False or ar.get("no_hand") is None or not ar["no_hand"], "export expects the hand-eval skip"
m = S.SmallNet(ar["width"], ar["hidden"], True, Q)
m.load_state_dict(st["model"])
W, H, NB = m.W, m.H, S.NB
sd = {k: v.detach().numpy().astype("<f4") for k, v in m.state_dict().items()}
out_w = sd["out.weight"].reshape(NB, Q, H)[:, q, :]
out_b = sd["out.bias"].reshape(NB, Q)[:, q]
blob = b"TSN1" + struct.pack("<2I", W, H)
for arr in (sd["ft.weight"].T, sd["ft.bias"], sd["l1.weight"], sd["l1.bias"], out_w, out_b, sd["skip"]):
    blob += np.ascontiguousarray(arr, dtype="<f4").tobytes()
open(a.out, "wb").write(blob)
z = dict(np.load(S.D + "/sn_eval.npz"))
idx = np.random.default_rng(5).choice(len(z["nn"]), 2000, replace=False)
rows = S.Rows(z, idx)
m.eval()
with torch.no_grad():
    y = m(*rows.batch(np.arange(len(idx)))[:5])
    y = (y[:, q] if Q > 1 else y).numpy()
with open(a.out + ".ref.txt", "w") as f:
    for i, k in enumerate(idx):
        f.write("%x %x %x %d %d %.3f\n" % (z["own"][k], z["opp"][k], z["blk"][k], z["pst"][k], z["hand"][k], 400 * y[i]))
print("wrote %s (%d bytes) W %d H %d head %s, ref %s.ref.txt" % (a.out, len(blob), W, H, a.tau, a.out))
