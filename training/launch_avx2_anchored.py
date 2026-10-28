"""Launcher for the 8-epoch AVX2 (W=256) training run with supplied Botzone anchor positions.

Follows user directive:
"when it saturated you do supply missing positions and do another 8 epochs"

Pipeline:
  1. Base saturation run completed on 32.53M positions with best MSE 0.021357 (net_ep3_ema.pt).
  2. Supplied missing positions: 88,295 Botzone >= 2200 Elo anchor positions labeled by Titanium 5k teacher.
  3. Combined dataset v6 (32,622,551 rows) with verified cryptographic SHA-256 features.
  4. Launch 8-epoch continuation on Modal GPU starting from /vol/ckpt/avx2_w256_saturated/net.pt.
"""
import os
import sys
import subprocess

ROOT = "C:/gitProjects/titanium-ataxx"
MODAL_SCRIPT = os.path.join(ROOT, "training/modal_train.py")

def main():
    print("=" * 65)
    print("=== LAUNCHING AVX2 (W=256) 8-EPOCH ANCHORED SATURATION RUN ===")
    print("=" * 65)

    tag = "avx2_w256_anchored"
    data_tag = "v6"
    train_args = (
        "--stage base "
        "--init /vol/ckpt/avx2_w256_saturated/net.pt "
        "--out /vol/ckpt/avx2_w256_anchored "
        "--epochs 8 "
        "--bs 8192 "
        "--lr 1.5e-4 "
        "--lr-min 1e-5 "
        "--opt ranger "
        "--la-k 8 "
        "--la-alpha 0.5 "
        "--ema 0.9995 "
        "--width 256 "
        "--diff-nudge 0.0 "
        "--threads 4"
    )

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    cmd_modal = [
        "python", "-m", "modal", "run", "--detach", MODAL_SCRIPT,
        "--tag", tag,
        "--data-tag", data_tag,
        "--args", train_args,
    ]

    print(f"\nModal Run Command:\n{' '.join(cmd_modal)}\n")
    r_modal = subprocess.run(cmd_modal, env=env, text=True)
    if r_modal.returncode == 0:
        print("\n[SUCCESS] Successfully launched AVX2 8-epoch anchored training on Modal GPU!")
    else:
        print(f"\n[ERROR] Modal launch returned exit code {r_modal.returncode}")
        sys.exit(r_modal.returncode)

if __name__ == "__main__":
    main()
