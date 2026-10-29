"""Assemble v7 dataset with 2,025,000 fresh +72 Elo D4 Champion relabels.

Integrates:
1. Base v5 training/val dataset (32.53M train, 677k val).
2. Fresh 5,000-node search labels from 405 chunks (2,025,000 top-disagreement positions)
   evaluated by the reigning champion nnue_main.tnn (7B25AC9D).
3. 88,295 external Botzone >= 2200 Elo Stockfish-style WDL anchors.

Outputs compressed npz files to data/nnue/modal_up/final_{train,val}_v7.npz ready for Modal.
"""
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
FINAL = os.path.join(ROOT, "data", "nnue", "final")
MODAL_UP = os.path.join(ROOT, "data", "nnue", "modal_up")
RELABEL_DIR = os.path.join(ROOT, "data", "nnue", "relabel", "out_dis_remaining_37c9a163_31212a7b_d8a135ce")
DIS_ROWS_PATH = os.path.join(ROOT, "data", "nnue", "relabel", "dis_rows.npy")
ANCHORS_PATH = os.path.join(ROOT, "data", "nnue", "botzone_anchors", "botzone_anchors_train.npz")

CHUNK_SIZE = 5000
TRAIN_BASE_ROWS = 32_534_256
VAL_BASE_ROWS = 677_670
TOTAL_BASE_ROWS = 33_891_139


def parse_chunk(path):
    m = re.search(r"c_(\d+)\.txt$", os.path.basename(path))
    if not m:
        return None
    cid = int(m.group(1))
    with open(path, "rb") as f:
        data = f.read()
    lines = data.splitlines()
    n = len(lines)
    sc = np.empty(n, dtype=np.int32)
    dp = np.empty(n, dtype=np.int16)
    nd = np.empty(n, dtype=np.int32)
    for i, line in enumerate(lines):
        p = line.split()
        sc[i] = int(p[0])
        dp[i] = int(p[1])
        nd[i] = int(p[2])
    return cid, sc, dp, nd


def main():
    t_start = time.time()
    os.makedirs(MODAL_UP, exist_ok=True)

    # 1. Discover chunk files
    chunk_files = sorted(
        glob.glob(os.path.join(RELABEL_DIR, "c_*.txt")),
        key=lambda p: int(re.search(r"c_(\d+)\.txt$", p).group(1)),
    )
    total_chunks = len(chunk_files)
    print(f"Found {total_chunks} completed chunk files in {RELABEL_DIR}.", flush=True)
    if total_chunks == 0:
        sys.exit("ERROR: No completed chunk files found.")

    # 2. Parse chunk files
    print(f"Parsing {total_chunks} chunks in parallel...", flush=True)
    chunk_scores = {}
    chunk_depths = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for res in ex.map(parse_chunk, chunk_files):
            if res is not None:
                cid, sc, dp, nd = res
                chunk_scores[cid] = sc
                chunk_depths[cid] = dp

    total_relabeled_positions = sum(len(sc) for sc in chunk_scores.values())
    print(f"Parsed {total_relabeled_positions:,} positions across {len(chunk_scores)} chunks.", flush=True)

    # 3. Read dis_rows mapping
    print(f"Reading dis_rows mapping from {DIS_ROWS_PATH}...", flush=True)
    dis_rows = np.fromfile(DIS_ROWS_PATH, dtype=np.int64)

    # Prepare update dictionaries
    train_updates = {}
    val_updates = {}

    for cid in sorted(chunk_scores.keys()):
        sc = chunk_scores[cid]
        dp = chunk_depths[cid]
        lo = cid * CHUNK_SIZE
        hi = lo + len(sc)
        global_ids = dis_rows[lo:hi]

        # Route by global row
        # Train: 0 .. TRAIN_BASE_ROWS-1
        train_mask = (global_ids < TRAIN_BASE_ROWS)
        if np.any(train_mask):
            row_idx = global_ids[train_mask]
            train_updates.update(zip(row_idx, sc[train_mask]))

        # Val: TRAIN_BASE_ROWS .. TRAIN_BASE_ROWS + VAL_BASE_ROWS - 1
        val_mask = (global_ids >= TRAIN_BASE_ROWS) & (global_ids < TRAIN_BASE_ROWS + VAL_BASE_ROWS)
        if np.any(val_mask):
            row_idx = global_ids[val_mask] - TRAIN_BASE_ROWS
            val_updates.update(zip(row_idx, sc[val_mask]))

    print(f"Disagreement updates mapped: {len(train_updates):,} train rows, {len(val_updates):,} val rows.", flush=True)

    # 4. Load base v5 datasets
    train_v5_path = os.path.join(MODAL_UP, "final_train_v5.npz")
    if not os.path.exists(train_v5_path):
        train_v5_path = os.path.join(FINAL, "final_train_v5.npz")
    val_v5_path = os.path.join(MODAL_UP, "final_val_v5.npz")
    if not os.path.exists(val_v5_path):
        val_v5_path = os.path.join(FINAL, "final_val_v5.npz")

    print(f"Loading base train from {train_v5_path}...", flush=True)
    with np.load(train_v5_path) as z:
        train_data = {k: z[k].copy() for k in z.files}
    assert len(train_data["stm"]) == TRAIN_BASE_ROWS, f"Expected {TRAIN_BASE_ROWS} rows, got {len(train_data['stm'])}"

    print(f"Loading base val from {val_v5_path}...", flush=True)
    with np.load(val_v5_path) as z:
        val_data = {k: z[k].copy() for k in z.files}
    assert len(val_data["stm"]) == VAL_BASE_ROWS, f"Expected {VAL_BASE_ROWS} rows, got {len(val_data['stm'])}"

    # 5. Apply updates to train
    print(f"Applying {len(train_updates):,} updates to train set...", flush=True)
    train_ti_cp = train_data["titanium_cp"].astype(np.float32)
    train_ti_old = train_data.get("titanium_cp_old", train_ti_cp.copy()).astype(np.float32)

    tr_indices = np.array(list(train_updates.keys()), dtype=np.int64)
    tr_scores = np.array(list(train_updates.values()), dtype=np.float32)
    train_ti_old[tr_indices] = train_ti_cp[tr_indices]
    train_ti_cp[tr_indices] = tr_scores

    train_data["titanium_cp"] = train_ti_cp
    train_data["titanium_cp_old"] = train_ti_old
    train_data["teacher_p3"] = (1.0 / (1.0 + np.exp(-np.clip(train_ti_cp, -1500, 1500) / 400.0))).astype(np.float32)

    # 6. Apply updates to val
    print(f"Applying {len(val_updates):,} updates to val set...", flush=True)
    val_ti_cp = val_data["titanium_cp"].astype(np.float32)
    val_ti_old = val_data.get("titanium_cp_old", val_ti_cp.copy()).astype(np.float32)

    va_indices = np.array(list(val_updates.keys()), dtype=np.int64)
    va_scores = np.array(list(val_updates.values()), dtype=np.float32)
    val_ti_old[va_indices] = val_ti_cp[va_indices]
    val_ti_cp[va_indices] = va_scores

    val_data["titanium_cp"] = val_ti_cp
    val_data["titanium_cp_old"] = val_ti_old
    val_data["teacher_p3"] = (1.0 / (1.0 + np.exp(-np.clip(val_ti_cp, -1500, 1500) / 400.0))).astype(np.float32)

    # 7. Merge Botzone >= 2200 Elo anchors
    print(f"Loading Botzone anchors from {ANCHORS_PATH}...", flush=True)
    za = np.load(ANCHORS_PATH)
    n_anchors = len(za["stm"])
    print(f"Loaded {n_anchors:,} anchor rows. Merging into train set...", flush=True)

    ti_cp_anchors = za["titanium_cp"].astype(np.float32)
    ti_prob_anchors = (1.0 / (1.0 + np.exp(-np.clip(ti_cp_anchors, -1500, 1500) / 400.0))).astype(np.float32)

    train_v7 = {
        "occ0": np.concatenate([train_data["occ0"], za["occ0"]]),
        "occ1": np.concatenate([train_data["occ1"], za["occ1"]]),
        "stm": np.concatenate([train_data["stm"], za["stm"]]),
        "outcome_p": np.concatenate([train_data["outcome_p"], za["outcome_p"].astype(np.float32)]),
        "teacher_kind": np.concatenate([train_data["teacher_kind"], np.zeros(n_anchors, dtype=np.uint8)]),
        "kata_td_win": np.concatenate([train_data["kata_td_win"], np.zeros((n_anchors, 4), dtype=np.float16)]),
        "kata_td_draw": np.concatenate([train_data["kata_td_draw"], np.zeros((n_anchors, 4), dtype=np.float16)]),
        "titanium_cp": np.concatenate([train_data["titanium_cp"], ti_cp_anchors]),
        "titanium_cp_old": np.concatenate([train_data["titanium_cp_old"], ti_cp_anchors]),
        "teacher_p3": np.concatenate([train_data["teacher_p3"], ti_prob_anchors]),
        "wsum": np.concatenate([train_data["wsum"], za["wsum"].astype(np.float32)]),
    }

    total_train_rows = len(train_v7["stm"])
    print(f"Total train rows in v7: {total_train_rows:,} (32.53M base + {n_anchors:,} anchors)", flush=True)

    # 8. Save compressed files
    out_train_path = os.path.join(MODAL_UP, "final_train_v7.npz")
    out_val_path = os.path.join(MODAL_UP, "final_val_v7.npz")

    print(f"Saving {out_train_path}...", flush=True)
    t_save = time.time()
    np.savez_compressed(out_train_path, **train_v7)
    train_mb = os.path.getsize(out_train_path) / 1_048_576.0
    print(f"Saved {out_train_path} ({train_mb:.1f} MB) in {time.time() - t_save:.1f}s.", flush=True)

    print(f"Saving {out_val_path}...", flush=True)
    t_save = time.time()
    np.savez_compressed(out_val_path, **val_data)
    val_mb = os.path.getsize(out_val_path) / 1_048_576.0
    print(f"Saved {out_val_path} ({val_mb:.1f} MB) in {time.time() - t_save:.1f}s.", flush=True)

    print(f"\nSUCCESS: v7 dataset assembled in {time.time() - t_start:.1f}s.")
    print(f"Train: {total_train_rows:,} rows (with {len(train_updates):,} fresh 5k labels + {n_anchors:,} anchors)")
    print(f"Val:   {len(val_data['stm']):,} rows (with {len(val_updates):,} fresh 5k labels)")


if __name__ == "__main__":
    main()
