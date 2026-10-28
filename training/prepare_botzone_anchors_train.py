"""Prepare Botzone anchor positions into binary feature records and training NPZ.

Produces:
  1. Raw binary board inputs (anchors.occ0, anchors.occ1, anchors.stm)
  2. Runs `titanium-cli nndump` to generate anchors.cnt and anchors.idx
  3. Builds data/nnue/botzone_anchors/botzone_anchors_train.npz with:
     - occ0, occ1, stm
     - titanium_cp, outcome_p, target, wsum, bucket
"""
import os
import sys
import json
import subprocess
import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
DATA_DIR = os.path.join(ROOT, "data/nnue/botzone_anchors")
IN_JSONL = os.path.join(DATA_DIR, "botzone_2200_anchors_labeled.jsonl")
OUT_NPZ = os.path.join(DATA_DIR, "botzone_anchors_train.npz")
CLI_EXE = "C:/gitProjects/main-ataxx/target/release/titanium-cli.exe"

def main():
    print(f"Loading labeled anchors from {IN_JSONL}...")
    recs = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    n = len(recs)
    print(f"Loaded {n:,} anchor records.")

    occ0 = np.empty(n, dtype=np.uint64)
    occ1 = np.empty(n, dtype=np.uint64)
    stm = np.empty(n, dtype=np.uint8)
    ti_cp = np.empty(n, dtype=np.int32)
    out_p = np.empty(n, dtype=np.float32)
    target = np.empty(n, dtype=np.float32)
    wsum = np.empty(n, dtype=np.float32)
    bucket = np.empty(n, dtype=np.uint8)

    for i, r in enumerate(recs):
        o0 = int(r["occ0"])
        o1 = int(r["occ1"])
        s = int(r.get("turn", 0))
        cp = int(r.get("titanium_cp", 0))
        op = float(r.get("outcome_p", 0.5))
        visits = int(r.get("n", 1))

        occ0[i] = o0
        occ1[i] = o1
        stm[i] = s
        ti_cp[i] = cp
        out_p[i] = op

        # Stockfish sigmoid target
        ti_prob = 1.0 / (1.0 + np.exp(-np.clip(cp, -1500, 1500) / 400.0))
        lam = 0.35 if visits >= 2 else 0.15
        target[i] = lam * op + (1.0 - lam) * ti_prob

        # Multi-visit confidence boost
        wsum[i] = min(4.0, 1.0 + 0.3 * (visits - 1)) if visits >= 2 else 1.0

        # Piece count bucket (0..3)
        stones = bin(o0).count("1") + bin(o1).count("1")
        if stones <= 14:
            bk = 0
        elif stones <= 24:
            bk = 1
        elif stones <= 34:
            bk = 2
        else:
            bk = 3
        bucket[i] = bk

    # Write raw binary files for nndump
    raw_prefix = os.path.join(DATA_DIR, "anchors_raw")
    occ0.tofile(f"{raw_prefix}.occ0")
    occ1.tofile(f"{raw_prefix}.occ1")
    stm.tofile(f"{raw_prefix}.stm")
    print(f"Wrote raw board inputs to {raw_prefix}.*")

    # Run titanium-cli nndump
    feat_prefix = os.path.join(DATA_DIR, "anchors_feat")
    if os.path.exists(CLI_EXE):
        print(f"Running nndump via {CLI_EXE}...")
        cmd = [CLI_EXE, "nndump", "--in", raw_prefix, "--out", feat_prefix]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"nndump failed: {r.stderr}")
        print("  nndump complete! Feature files created.")
    else:
        print(f"Warning: {CLI_EXE} not found, skipping local nndump.")

    # Save NPZ
    print(f"Saving {OUT_NPZ}...")
    np.savez_compressed(
        OUT_NPZ,
        occ0=occ0,
        occ1=occ1,
        stm=stm,
        titanium_cp=ti_cp,
        outcome_p=out_p,
        target=target,
        wsum=wsum,
        bucket=bucket,
    )
    print(f"Done! Created {OUT_NPZ} ({os.path.getsize(OUT_NPZ):,} bytes).")

if __name__ == "__main__":
    main()
