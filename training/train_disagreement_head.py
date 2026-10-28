"""Train the zero-cost Expected Disagreement Head (TNN3) via closed-form Ridge Regression.

Target: Delta = min(2000, |StaticEval - Search5k|) / 400.0.
Inputs: Hidden layer activations h = (h1 || h2) in [0, 1]^48 for each position.
For each phase bucket b in [0..7]:
  Solves (X_b^T X_b + lambda * I) w_b = X_b^T Delta_b
Exported to TNN3 format with zero changes to static evaluation weights.
"""

import argparse
import os
import struct
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nnue as T
from ataxx_move import board_str


ROOT = os.environ.get("TI_ROOT", "C:/gitProjects/titanium-ataxx")

def main():
    parser = argparse.ArgumentParser(description="Train Disagreement Head (TNN3)")
    parser.add_argument("--pt", default=ROOT + "/data/nnue/modal_ckpt/s1c_hg/net.pt", help="Path to champion .pt checkpoint")
    parser.add_argument("--out", default=ROOT + "/data/nnue/champion_disagree.tnn", help="Output TNN3 network file")
    parser.add_argument("--rows", type=int, default=200000, help="Number of training rows to fit ridge regression")
    parser.add_argument("--l2", type=float, default=1.0, help="Ridge regularization parameter lambda")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size for forward activation extraction")
    args = parser.parse_args()

    print(f"[1/4] Loading champion model from {args.pt}...")
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    W = sd["ft.weight"].shape[1]
    L1 = sd["l1.weight"].shape[0] // T.NB
    L2 = sd["l2.weight"].shape[0] // T.NB
    pairwise = sd["l1.weight"].shape[1] == W if "pairwise" in sd else False
    v2 = "s57w" in sd

    model = T.NNUE2(W=W, L1=L1, L2=L2, pairwise=pairwise) if v2 else T.NNUE(W=W, L1=L1, L2=L2)
    model.load_state_dict(sd)
    model.eval()

    print(f"[2/4] Loading dataset and extracting activations for {args.rows} rows...")
    tr_split = T.Split("train", "base", None)
    tr_split.rot_aug = True
    n_rows = min(args.rows, len(tr_split.target))
    rows = np.random.default_rng(42).choice(len(tr_split.target), n_rows, replace=False)

    # Load search targets from final_train_v5.npz
    train_npz = np.load(T.FINAL + "/final_train_v5.npz")

    # Accumulate ridge statistics per bucket: A_b = X_b^T X_b, v_b = X_b^T y_b
    dim_in = L1 + L2  # 16 + 32 = 48
    A = [np.zeros((dim_in + 1, dim_in + 1), dtype=np.float64) for _ in range(T.NB)]
    V = [np.zeros(dim_in + 1, dtype=np.float64) for _ in range(T.NB)]
    counts = [0 for _ in range(T.NB)]
    sum_y2 = [0.0 for _ in range(T.NB)]

    bs = args.batch_size
    for i in range(0, n_rows, bs):
        batch_idx = rows[i : i + bs]
        ix, off, bk, _, _ = tr_split.batch(batch_idx)
        
        with torch.no_grad():
            a = model.ft(ix, off) + model.ftb
            if getattr(model, "pairwise", False):
                h = model.W // 2
                a = a[:, :h] * a[:, h:]
            x = torch.cat([a[0::2], a[1::2]], 1).clamp(0, 1)
            n = len(bk)
            ar = torch.arange(n)
            h1 = model.l1(x).view(n, T.NB, model.L1)[ar, bk].clamp(0, 1)
            h2 = model.l2(h1).view(n, T.NB, model.L2)[ar, bk].clamp(0, 1)
            feat = torch.cat([h1, h2], dim=1).numpy()

            if v2:
                o = model.out(torch.cat([h1, h2], dim=1))[ar, bk]
            else:
                o = model.out(h2)[ar, bk]
            ps = model.psqt(ix, off)
            static_logit = o + 0.5 * (ps[0::2] - ps[1::2])[ar, bk]
            static_cp = (400.0 * static_logit).numpy()

        search_cp = train_npz["titanium_cp"][batch_idx]
        delta_batch = np.clip(np.abs(static_cp - search_cp), 0.0, 2000.0).astype(np.float32) / 400.0

        bk_np = bk.numpy()
        for b in range(T.NB):
            mask = (bk_np == b)
            cnt = mask.sum()
            if cnt == 0:
                continue
            counts[b] += cnt
            X_b = np.hstack([feat[mask], np.ones((cnt, 1), dtype=np.float32)])
            y_b = delta_batch[mask]
            A[b] += X_b.T @ X_b
            V[b] += X_b.T @ y_b
            sum_y2[b] += float(np.sum(y_b**2))

    print("[3/4] Solving closed-form ridge regressions per phase bucket...")
    dis_w = np.zeros((T.NB, dim_in), dtype=np.float32)
    dis_b = np.zeros(T.NB, dtype=np.float32)

    for b in range(T.NB):
        if counts[b] < 50:
            print(f"  Bucket {b}: insufficient samples ({counts[b]}), using baseline uncertainty 0.5")
            dis_b[b] = 0.5
            continue
        # Solve (A_b + lambda * I) w = V_b
        reg = args.l2 * np.eye(dim_in + 1, dtype=np.float64)
        w_sol = np.linalg.solve(A[b] + reg, V[b]).astype(np.float32)
        dis_w[b] = w_sol[:dim_in]
        dis_b[b] = w_sol[dim_in]
        # Evaluate training MAE on bucket summary
        mean_err = np.sqrt(np.maximum(0.0, (w_sol @ A[b] @ w_sol - 2 * w_sol @ V[b] + sum_y2[b]) / counts[b])) * 400.0
        print(f"  Bucket {b} (N={counts[b]}): solved w in R^{dim_in}, bias={dis_b[b]:.4f}, approx RMSE={mean_err:.1f} cp")

    print(f"[4/4] Exporting to TNN3 format: {args.out}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    groups = 1 | 2 | 4 | 16 | 32
    f32 = lambda t: t.detach().cpu().numpy().astype("<f4").ravel().tobytes()

    with open(args.out, "wb") as f:
        # TNN3 magic header: includes has_dis=True
        # Format: magic(4), NUM_FEATS, W, L1, L2, NB, groups, pairwise(1)
        pairwise_int = int(pairwise)
        f.write(b"TNN3" + struct.pack("<7I", T.NUM_FEATS, W, L1, L2, T.NB, groups, pairwise_int))
        # Standard weights
        for k in ("ft.weight", "ftb", "psqt.weight", "l1.weight", "l1.bias", "l2.weight", "l2.bias", "out.weight", "out.bias"):
            f.write(f32(sd[k]))
        if v2:
            f.write(f32(sd["s57w"]))
        # TNN3 auxiliary Disagreement Head weights
        f.write(dis_w.astype("<f4").tobytes())
        f.write(dis_b.astype("<f4").tobytes())

    file_size = os.path.getsize(args.out)
    print(f"Exported TNN3 successfully: {args.out} ({file_size} bytes, +{dis_w.nbytes + dis_b.nbytes} bytes dis head)")

    # Parity reference generation for nneval
    print("Generating validation reference evals for parity tripwire...")
    va = T.Split("val", "base", None)
    ref_rows = np.random.default_rng(3).choice(len(va.target), 1000, replace=False)
    z_val = np.load(T.FINAL + "/final_val_v5.npz")
    ix_val, off_val, bk_val, _, _ = va.batch(ref_rows)
    with torch.no_grad():
        y_val = model(ix_val, off_val, bk_val).numpy()
    ref_out = args.out + ".ref.txt"
    with open(ref_out, "w") as fo:
        for r, v in zip(ref_rows, y_val):
            fo.write("%s %d\n" % (board_str(int(z_val["occ0"][r]), int(z_val["occ1"][r]), 0, int(z_val["stm"][r])),
                                  int(np.clip(np.round(400 * v), -30000, 30000))))
    print(f"Parity reference written to {ref_out} (1,000 boards)")


if __name__ == "__main__":
    main()
