"""Merge Botzone >=2200 Elo anchor positions with Titanium 5k-node search evaluations.

Combines:
  1. data/nnue/botzone_anchors/botzone_2200_anchors.jsonl:
     - D4 canonical bitboards (occ0, occ1, blockers, turn)
     - Stockfish-style D4 calibrated WDL outcomes (w, l, d, n, outcome_p = (w + 0.5*d)/n)
     - Top-10 engine evaluations (engine_cp) from match debug logs (where available)
  2. data/nnue/botzone_anchors/labels/c_*.txt:
     - Titanium 5k-node teacher search evaluations (titanium_cp, depth, nodes)

Supports --partial mode for inspecting intermediate chunks while labeling is in progress.
"""
import argparse
import glob
import json
import os
import re
import sys
import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
DATA_DIR = os.path.join(ROOT, "data/nnue/botzone_anchors")
JSONL_PATH = os.path.join(DATA_DIR, "botzone_2200_anchors.jsonl")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
OUT_JSONL = os.path.join(DATA_DIR, "botzone_2200_anchors_labeled.jsonl")
CHUNK_SIZE = 5000

def main():
    parser = argparse.ArgumentParser(description="Merge Botzone anchors with Titanium labels")
    parser.add_argument("--jsonl", default=JSONL_PATH, help="Input anchors JSONL")
    parser.add_argument("--labels-dir", default=LABELS_DIR, help="Labels output directory")
    parser.add_argument("--out", default=OUT_JSONL, help="Output enriched JSONL")
    parser.add_argument("--partial", action="store_true", help="Allow partial chunks without failing")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        sys.exit(f"Error: {args.jsonl} not found")

    print(f"Loading anchors from {args.jsonl}...")
    anchors = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                anchors.append(json.loads(line))
    total_anchors = len(anchors)
    print(f"Loaded {total_anchors:,} total anchor positions.")

    total_chunks = (total_anchors + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Expected {total_chunks} chunks ({CHUNK_SIZE} pos/chunk). Scanning {args.labels_dir}...")

    # Load chunk files
    chunk_files = {}
    for p in glob.glob(os.path.join(args.labels_dir, "c_*.txt")):
        m = re.search(r"c_(\d+)\.txt$", p)
        if m:
            cid = int(m.group(1))
            chunk_files[cid] = p

    found_chunks = sorted(chunk_files.keys())
    print(f"Found {len(found_chunks)} / {total_chunks} chunk files: {found_chunks}")

    if not args.partial and len(found_chunks) < total_chunks:
        missing = [c for c in range(total_chunks) if c not in chunk_files]
        sys.exit(f"Missing {len(missing)} chunks: {missing[:10]}... Use --partial to merge available.")

    # Parse labels
    labeled_count = 0
    all_titanium_cp = [None] * total_anchors
    all_depths = [None] * total_anchors
    all_nodes = [None] * total_anchors

    for cid in found_chunks:
        c_path = chunk_files[cid]
        lo = cid * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, total_anchors)
        expected_len = hi - lo

        with open(c_path, "r", encoding="utf-8") as cf:
            lines = [l.strip() for l in cf if l.strip()]

        if len(lines) != expected_len:
            print(f"Warning: Chunk {cid} length mismatch: got {len(lines)}, expected {expected_len}")

        for i, line in enumerate(lines[:expected_len]):
            parts = line.split()
            if len(parts) >= 3:
                sc = int(parts[0])
                d = int(parts[1])
                n = int(parts[2])
                pos_idx = lo + i
                all_titanium_cp[pos_idx] = sc
                all_depths[pos_idx] = d
                all_nodes[pos_idx] = n
                labeled_count += 1

    print(f"Successfully loaded {labeled_count:,} Titanium teacher labels ({labeled_count / total_anchors * 100:.1f}%).")

    # Merge and compute statistics
    merged_records = []
    engine_vs_titanium_diffs = []
    outcome_vs_titanium_diffs = []
    multi_visit_count = 0

    for i in range(total_anchors):
        rec = dict(anchors[i])
        ti_cp = all_titanium_cp[i]
        if ti_cp is not None:
            rec["titanium_cp"] = ti_cp
            rec["depth"] = all_depths[i]
            rec["nodes"] = all_nodes[i]

            # Stockfish target blend:
            # lambda * outcome_p + (1 - lambda) * sigmoid(titanium_cp / 400)
            ti_p = 1.0 / (1.0 + np.exp(-np.clip(ti_cp, -1500, 1500) / 400.0))
            out_p = rec["outcome_p"]

            # Weight confidence: if position was visited 2+ times in elite games,
            # Bayesian outcome is reliable; if visited once (n=1), Titanium dominates
            if rec["n"] >= 2:
                multi_visit_count += 1
                lam = 0.35 # Strong empirical WDL anchor
            else:
                lam = 0.15 # Primary teacher search anchor

            rec["target_p"] = float(round(lam * out_p + (1.0 - lam) * ti_p, 4))

            # Statistics comparison
            if rec.get("engine_cp") is not None:
                eng_cp = rec["engine_cp"]
                engine_vs_titanium_diffs.append((eng_cp, ti_cp))

            outcome_vs_titanium_diffs.append((out_p, ti_p))
            merged_records.append(rec)
        elif not args.partial:
            merged_records.append(rec)

    print(f"Writing {len(merged_records):,} records to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as out_f:
        for r in merged_records:
            out_f.write(json.dumps(r) + "\n")

    # Print analysis
    print("\n" + "=" * 60)
    print("=== BOTZONE ANCHOR DATASET ANALYSIS ===")
    print("=" * 60)
    print(f"Total positions in dataset  : {total_anchors:,}")
    print(f"Positions with Titanium eval: {len(merged_records):,}")
    print(f"Multi-visit duplicate pos   : {multi_visit_count:,} (WDL Bayesian anchored)")

    if engine_vs_titanium_diffs:
        eng_arr = np.array([p[0] for p in engine_vs_titanium_diffs])
        ti_arr = np.array([p[1] for p in engine_vs_titanium_diffs])
        diff_arr = ti_arr - eng_arr
        corr = np.corrcoef(eng_arr, ti_arr)[0, 1]
        print(f"\nTop Engine Eval vs Titanium Teacher (N = {len(engine_vs_titanium_diffs):,}):")
        print(f"  Correlation coefficient (r) : {corr:.4f}")
        print(f"  Mean engine evaluation      : {np.mean(eng_arr):+.1f} cp")
        print(f"  Mean Titanium evaluation    : {np.mean(ti_arr):+.1f} cp")
        print(f"  Mean difference (Ti - Eng)  : {np.mean(diff_arr):+.1f} cp (std = {np.std(diff_arr):.1f})")

    if outcome_vs_titanium_diffs:
        out_arr = np.array([p[0] for p in outcome_vs_titanium_diffs])
        ti_prob_arr = np.array([p[1] for p in outcome_vs_titanium_diffs])
        mse = np.mean((ti_prob_arr - out_arr) ** 2)
        corr_out = np.corrcoef(out_arr, ti_prob_arr)[0, 1]
        print(f"\nEmpirical WDL Outcome vs Titanium Sigmoid Eval:")
        print(f"  Outcome - Eval Correlation  : {corr_out:.4f}")
        print(f"  Mean Squared Error (MSE)    : {mse:.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
