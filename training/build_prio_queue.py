"""Build a prioritized relabel queue: dedupe transpositions, rank by importance.

Importance (highest first):
  1. Known eval failures (data/nnerr/boards.txt, matched by board hash) - these NEED relabeling
  2. High |score| from existing labels (decisive positions = high regret if wrong)
  3. Everything else in shuffled order (positions.bin is already shuffled, seed 11)

Output: data/nnue/relabel/prio_positions.bin (25-byte records) + prio_rows.npy.
The PC relabel reads this instead of positions.bin.
"""
import numpy as np
import os

ROOT = "C:/gitProjects/titanium-ataxx"
D = ROOT + "/data/nnue/relabel"
REC = np.dtype([("o0", "<u8"), ("o1", "<u8"), ("blk", "<u8"), ("stm", "u1")])

print("loading positions...", flush=True)
recs = np.fromfile(D + "/positions.bin", dtype=REC)
n = len(recs)
print(f"  {n} records", flush=True)
print("loading labels for |score|...", flush=True)
z = np.load(D + "/labels.npz")
scores = np.abs(z["score"].astype(np.float64))
print(f"  score range 0..{scores.max():.0f}, mean {scores.mean():.0f}", flush=True)

# 1. Dedupe by (o0,o1,blk,stm) - keep first occurrence
print("deduping transpositions...", flush=True)
# Use a structured view for uniqueness
keys = np.empty(n, dtype=[("o0", "<u8"), ("o1", "<u8"), ("blk", "<u8"), ("stm", "u1")])
keys["o0"], keys["o1"], keys["blk"], keys["stm"] = recs["o0"], recs["o1"], recs["blk"], recs["stm"]
_, uniq_idx = np.unique(keys, return_index=True)
uniq_idx = np.sort(uniq_idx)  # keep original order among uniques
print(f"  {len(uniq_idx)} unique of {n} ({n - len(uniq_idx)} dupes)", flush=True)

# 2. Importance: |score| descending (decisive positions = high regret if wrong).
# Stopping at 20% still covers the most important 20%.
order = np.argsort(-scores[uniq_idx], kind="stable")
prio = uniq_idx[order]

# 3. Boost known failures to front (matched by exact board encoding
# occ0/occ1/blockers/stm <-> 49-char board + side)
print("boosting nnerr boards...", flush=True)
try:
    fail_keys = set()
    for f in ["b1k.txt", "b2k.txt", "b60.txt", "boards.txt"]:
        p = ROOT + "/data/nnerr/" + f
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                t = line.strip().split()
                if len(t[0]) != 49:
                    continue
                s, stm = t[0], (0 if len(t) < 2 or t[1] not in "wx" else 1)
                o0 = o1 = blk = 0
                for i, c in enumerate(s):
                    if c == "x":
                        o0 |= 1 << i
                    elif c == "o":
                        o1 |= 1 << i
                    elif c == "#":
                        blk |= 1 << i
                # stm: our records use 0/1 for black/white to move; nnerr lines end b/w
                fail_keys.add((o0, o1, blk))
    print(f"  {len(fail_keys)} known-failure board hashes", flush=True)
    if fail_keys:
        rec = recs[prio]
        is_fail = np.array([(int(r[0]), int(r[1]), int(r[2])) in fail_keys for r in rec])
        print(f"  {is_fail.sum()} queue positions match known failures -> front", flush=True)
        prio = np.concatenate([prio[is_fail], prio[~is_fail]])
except Exception as e:
    print(f"  nnerr boost failed (continuing without): {e}", flush=True)

print(f"writing {len(prio)} prioritized records...", flush=True)
recs[prio].tofile(D + "/prio_positions.bin")
# rows.npy mapping: global dataset row of each prio record
rows = np.load(D + "/rows.npy")
rows[prio].astype(np.int64).tofile(D + "/prio_rows.npy")
print("done", flush=True)
