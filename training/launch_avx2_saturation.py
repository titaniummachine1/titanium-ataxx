"""Orchestrator to finalize Botzone anchors and launch AVX2 net saturation on Modal GPU.

Follows Decisions D130, D134, D135, D136:
  1. Merge all 88,295 Botzone >= 2200 Elo anchor positions with Titanium 5k teacher evaluations.
  2. Verify 100% label completeness across all 18 chunks.
  3. Launch Modal GPU saturation training of downscaled_w256.pt (W=256, 6 epochs, Ranger, uniform loss).
  4. Track training progress and export champion .tnn.
"""
import os
import sys
import subprocess
import time

ROOT = "C:/gitProjects/titanium-ataxx"
DATA_DIR = os.path.join(ROOT, "data/nnue/botzone_anchors")
MERGE_SCRIPT = os.path.join(ROOT, "training/merge_botzone_anchors.py")
MODAL_SCRIPT = os.path.join(ROOT, "training/modal_train.py")

def main():
    print("=" * 60)
    print("=== LAUNCHING AVX2 NET SATURATION PIPELINE ===")
    print("=" * 60)

    # 1. Merge labels
    print("\nStep 1: Merging Titanium 5k teacher labels with Botzone WDL anchors...")
    cmd_merge = [sys.executable, MERGE_SCRIPT]
    r = subprocess.run(cmd_merge, text=True)
    if r.returncode != 0:
        print("Warning: Full merge not ready yet. Running partial check...")
        subprocess.run([sys.executable, MERGE_SCRIPT, "--partial"], text=True)
        return

    # 2. Launch Modal GPU run
    print("\nStep 2: Launching AVX2 (W=256) saturation on Modal GPU...")
    tag = "avx2_w256_saturated"
    train_args = (
        "--stage base "
        "--init /vol/up/downscaled_w256.pt "
        "--out /vol/ckpt/avx2_w256_saturated "
        "--epochs 6 "
        "--bs 8192 "
        "--lr 3e-4 "
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
        "--args", train_args,
    ]
    print(f"Command: {' '.join(cmd_modal)}")
    r_modal = subprocess.run(cmd_modal, env=env, text=True)
    if r_modal.returncode == 0:
        print("\n[SUCCESS] Successfully launched AVX2 saturation training on Modal GPU!")
    else:
        print(f"\nModal launch returned code {r_modal.returncode}")

if __name__ == "__main__":
    main()
