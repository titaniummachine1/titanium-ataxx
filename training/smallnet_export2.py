"""Export a small net v2 (smallnet_train.py --extra --alpha-in, bound mode) for the engine (D115): magic "TSN2", u32 W, H, K
(K = 11 extra inputs: the 10 tactical scalars of smallnet::tactical(), then alpha / 400 clipped to +-5), then f32 LE arrays:
ft[147][W] (feature-major), ftb[W], l1_w[NB*H][2W+K] (row = bucket*H + i; columns = 2W activations then K extras),
l1_b[NB*H], out_w[NB][H], out_b[NB], skip[2][NB] (parent static eval, child hand eval; cp/400), exskip[NB][K].
ONE head (--tau). The calibration shift (cp) that gives the target wrong-prune rate on the held-out calls (default 1.43% =
the deployed v1 net) is added to out_b, so the engine uses the bound as is.
Also writes <out>.ref.txt: own opp blk (hex u64) parent_cp hand_cp alpha_cp expected_cp for 2000 held-out calls (Rust parity).
Usage: python training/smallnet_export2.py --pt data/nnue/smallnet/v2/opt_002.pt --tau 0.05 --out data/nnue/fut_small2.tsn
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import smallnet_train as S  # noqa: E402
import smallnet_eval2 as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--tau", type=float, required=True)
ap.add_argument("--rate", type=float, default=1.43, help="target wrong-prune rate in %% of the moves the big net keeps")
ap.add_argument("--out", required=True)
a = ap.parse_args()
m, ar, val = E.load_model(a.pt)
assert ar.get("extra") and ar.get("alpha_in"), "export2 expects --extra --alpha-in nets"
taus = [float(x) for x in ar["quantiles"].split(",")]
q = taus.index(a.tau)
W, H, K, NB = m.W, m.H, m.K, S.NB
assert K == S.NEX + 1

z = dict(np.load(S.D + "/sn_eval.npz"))
exf = S.D + "/sn_eval_ex.npy"
if not os.path.exists(exf):
    own, opp, blk = z["own"], z["opp"], z["blk"]
    ex = np.empty((len(own), S.NEX), np.float16)
    for i in range(0, len(own), 1 << 20):
        ex[i:i + (1 << 20)] = S.extras(own[i:i + (1 << 20)], opp[i:i + (1 << 20)], blk[i:i + (1 << 20)])
    np.save(exf, ex)
ex = np.load(exf)
rows = S.Rows(z, np.arange(len(z["nn"])), ex=ex, alpha_in=True)
out = np.round(S.evaluate(m, rows) * 400).astype(np.int64)[:, q]
alpha, nn, nd0 = z["alpha"].astype(np.int64), z["nn"].astype(np.int64), z["nd0"].astype(np.int64)
keep = ~(nd0 - E.red(alpha, nn) <= 0)
unc = z["cached"] == 0
shifts = np.arange(-300, 301)
wr = np.array([100 * ((nd0 - E.red(alpha, out + s) <= 0) & keep).sum() / keep.sum() for s in shifts])
sk = np.array([100 * ((nd0 - E.red(alpha, out + s) <= 0) & unc).sum() / unc.sum() for s in shifts])
i = int(np.argmin(np.abs(wr - a.rate)))
shift = int(shifts[i])
print("tau %g: shift %+d cp gives wrong prunes %.2f%% (target %.2f), skipped big-net evals %.1f%%" % (a.tau, shift, wr[i], a.rate, sk[i]))

sd = {k: v.detach().numpy().astype("<f4") for k, v in m.state_dict().items()}
out_w = sd["out.weight"].reshape(NB, len(taus), H)[:, q, :]
out_b = sd["out.bias"].reshape(NB, len(taus))[:, q] + shift / 400.0
blob = b"TSN2" + struct.pack("<3I", W, H, K)
for arr in (sd["ft.weight"].T, sd["ft.bias"], sd["l1.weight"], sd["l1.bias"], out_w, out_b, sd["skip"], sd["exskip"]):
    blob += np.ascontiguousarray(arr, dtype="<f4").tobytes()
open(a.out, "wb").write(blob)
idx = np.random.default_rng(5).choice(len(nn), 2000, replace=False)
r2 = S.Rows(z, idx, ex=ex, alpha_in=True)
m.eval()
with torch.no_grad():
    y = m(*r2.batch(np.arange(len(idx)))[:6])[:, q].numpy() + shift / 400.0
with open(a.out + ".ref.txt", "w") as f:
    for j, k in enumerate(idx):
        f.write("%x %x %x %d %d %d %.3f\n" % (z["own"][k], z["opp"][k], z["blk"][k], z["pst"][k], z["hand"][k], z["alpha"][k], 400 * y[j]))
print("wrote %s (%d bytes) W %d H %d K %d head %g shift %+d, ref %s.ref.txt" % (a.out, len(blob), W, H, K, a.tau, shift, a.out))
