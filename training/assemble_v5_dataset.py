"""Assemble v5 relabeled dataset by mapping the 6,314 chunks into labels_v5.npz.

Matches the exact row mapping of final_{train,val,test}_v3.npz:
- dis_rows.npy: global dataset row index for each of the 31,568,598 packed positions.
- out_dis_remaining_e3f8dc6e_e11827d1_d8a135ce: 6,314 chunk files (c_0.txt .. c_6313.txt).
- Existing verified rows from samehash / static-search agreement retain their labels.
"""
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
REL = os.path.join(ROOT, "data", "nnue", "relabel")
FINAL = os.path.join(ROOT, "data", "nnue", "final")
CHUNK_DIR = os.path.join(REL, "out_dis_remaining_e3f8dc6e_e11827d1_d8a135ce")
CHUNK_SIZE = 5000
TOTAL_ROWS = 33_891_139
DIS_COUNT = 31_568_598


def load_existing_labels():
    old_p = os.path.join(REL, "labels.npz")
    print(f"Loading existing baseline labels from {old_p}...", flush=True)
    L = np.load(old_p)
    rows = L["rows"].copy()
    have = L["have"].copy()
    score = L["score"].copy()
    depth = L["depth"].copy()
    nodes = L["nodes"].copy()
    assert len(rows) == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows, got {len(rows)}"
    return rows, have, score, depth, nodes


def parse_chunk_file(path):
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
    t0 = time.time()
    dis_rows_p = os.path.join(REL, "dis_rows.npy")
    print(f"Reading dis_rows from {dis_rows_p}...", flush=True)
    dis_rows = np.fromfile(dis_rows_p, dtype=np.int64)
    assert len(dis_rows) == DIS_COUNT, f"Expected {DIS_COUNT} rows in dis_rows, got {len(dis_rows)}"

    rows, have, score, depth, nodes = load_existing_labels()

    # Invert rows permutation: global_row -> packed index in labels.npz
    print("Building global_row -> packed_index lookup...", flush=True)
    inv = np.empty(TOTAL_ROWS, dtype=np.int64)
    inv[rows] = np.arange(TOTAL_ROWS)

    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "c_*.txt")),
                         key=lambda p: int(re.search(r"c_(\d+)\.txt$", p).group(1)))
    print(f"Found {len(chunk_files)} chunk files in {CHUNK_DIR}.", flush=True)
    assert len(chunk_files) == 6314, f"Expected 6314 chunks, found {len(chunk_files)}"

    print("Parsing all 6,314 chunks in parallel...", flush=True)
    t_parse = time.time()
    parsed_count = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        for cid, sc, dp, nd in executor.map(parse_chunk_file, chunk_files):
            lo = cid * CHUNK_SIZE
            hi = lo + len(sc)
            assert hi <= DIS_COUNT, f"Chunk {cid} range [{lo}:{hi}] exceeds {DIS_COUNT}"
            global_ids = dis_rows[lo:hi]
            packed_ids = inv[global_ids]

            score[packed_ids] = sc
            depth[packed_ids] = dp
            nodes[packed_ids] = nd
            have[packed_ids] = True
            parsed_count += len(sc)

    print(f"Parsed {parsed_count:,} labels in {time.time() - t_parse:.2f}s.", flush=True)
    assert parsed_count == DIS_COUNT, f"Parsed count {parsed_count} != {DIS_COUNT}"

    # Verify all have flags
    assert have.all(), "Not all rows have valid labels!"
    print(f"All {TOTAL_ROWS:,} rows verified present and valid.", flush=True)

    out_p = os.path.join(REL, "labels_v5.npz")
    print(f"Saving combined labels to {out_p}...", flush=True)
    np.savez_compressed(
        out_p,
        rows=rows,
        have=have,
        score=score,
        depth=depth,
        nodes=nodes
    )
    print(f"Successfully assembled labels_v5.npz in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
