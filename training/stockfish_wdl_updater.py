"""Stockfish-Style Canonical WDL Deduplicator & Ground Truth Anchor Updater.

Ingests external/new games (Botzone, Little Golem, self-play) and:
1. Canonicalizes each position under D4 dihedral symmetry (8-fold rotation/reflection group).
2. For matching/duplicate positions in the dataset:
   - Accumulates exact WDL statistics: W += win, L += loss, D += draw, N += 1
   - Computes Bayesian empirical ground-truth posterior: outcome_p = (W + 0.5*D) / N
   - Eliminates outcome variance and anchors the manifold to true game outcomes.
3. For novel out-of-distribution positions:
   - Queues them as fresh anchors to expand dataset coverage and break self-play echo chambers.
"""

import os
import sys
import json
import time
import numpy as np

# 7x7 Ataxx D4 dihedral symmetry index maps
def build_d4_symmetry_maps():
    maps = []
    for mode in range(8):
        m = np.zeros(49, dtype=np.int32)
        for r in range(7):
            for c in range(7):
                idx = r * 7 + c
                if mode == 0: nr, nc = r, c
                elif mode == 1: nr, nc = c, 6 - r
                elif mode == 2: nr, nc = 6 - r, 6 - c
                elif mode == 3: nr, nc = 6 - c, r
                elif mode == 4: nr, nc = r, 6 - c
                elif mode == 5: nr, nc = 6 - r, c
                elif mode == 6: nr, nc = c, r
                elif mode == 7: nr, nc = 6 - c, 6 - r
                m[idx] = nr * 7 + nc
        maps.append(m)
    return maps

D4_MAPS = build_d4_symmetry_maps()

def bitmask_permute(mask: int, perm: np.ndarray) -> int:
    out = 0
    for src in range(49):
        if (mask >> src) & 1:
            out |= (1 << int(perm[src]))
    return out

def canonical_key(occ0: int, occ1: int, blk: int, stm: int) -> tuple:
    """Return the lexicographically minimal canonical (occ0, occ1, blk, stm) across all 8 D4 symmetries."""
    best_occ0 = occ0
    best_occ1 = occ1
    best_blk = blk
    for perm in D4_MAPS[1:]:
        t0 = bitmask_permute(occ0, perm)
        t1 = bitmask_permute(occ1, perm)
        tb = bitmask_permute(blk, perm)
        if (t0 < best_occ0) or (t0 == best_occ0 and t1 < best_occ1) or (t0 == best_occ0 and t1 == best_occ1 and tb < best_blk):
            best_occ0 = t0
            best_occ1 = t1
            best_blk = tb
    return (best_occ0, best_occ1, best_blk, stm)

class WdlAnchorIndex:
    """In-memory canonical index for rapid lookup and Bayesian WDL updating."""
    def __init__(self):
        # canonical_key -> index in dataset or metadata dict
        self.table = {}
        self.stats = {"updated_anchors": 0, "new_anchors": 0}

    def load_dataset_slice(self, occ0_arr, occ1_arr, blk_arr, stm_arr, wins_arr, losses_arr, draws_arr, n_outcomes_arr):
        n = len(stm_arr)
        print(f"Building canonical anchor index over {n:,} positions...", flush=True)
        t0 = time.time()
        for i in range(n):
            key = canonical_key(int(occ0_arr[i]), int(occ1_arr[i]), int(blk_arr[i]), int(stm_arr[i]))
            if key not in self.table:
                self.table[key] = {
                    "row_idx": i,
                    "w": int(wins_arr[i]),
                    "l": int(losses_arr[i]),
                    "d": int(draws_arr[i]),
                    "n": int(n_outcomes_arr[i]),
                }
            else:
                # Merge existing duplicate rows inside dataset Stockfish-style
                entry = self.table[key]
                entry["w"] += int(wins_arr[i])
                entry["l"] += int(losses_arr[i])
                entry["d"] += int(draws_arr[i])
                entry["n"] += int(n_outcomes_arr[i])
        print(f"Index built: {len(self.table):,} unique canonical positions in {time.time()-t0:.2f}s", flush=True)

    def update_with_game_position(self, occ0: int, occ1: int, blk: int, stm: int, game_result: float):
        """Update existing position with new game outcome (1.0 = stm won, 0.0 = stm lost, 0.5 = draw)."""
        key = canonical_key(occ0, occ1, blk, stm)
        if key in self.table:
            entry = self.table[key]
            if game_result == 1.0:
                entry["w"] += 1
            elif game_result == 0.0:
                entry["l"] += 1
            else:
                entry["d"] += 1
            entry["n"] += 1
            entry["outcome_p"] = (entry["w"] + 0.5 * entry["d"]) / entry["n"]
            self.stats["updated_anchors"] += 1
            return True, entry
        else:
            w = 1 if game_result == 1.0 else 0
            l = 1 if game_result == 0.0 else 0
            d = 1 if game_result == 0.5 else 0
            new_entry = {"w": w, "l": l, "d": d, "n": 1, "outcome_p": game_result}
            self.table[key] = new_entry
            self.stats["new_anchors"] += 1
            return False, new_entry

if __name__ == "__main__":
    print("Self-test: verifying canonical symmetry properties...")
    # Corner stone (0, 0)
    k1 = canonical_key(1, 0, 0, 0)
    # Opposite corner (6, 6) under rot180 -> square 48
    k2 = canonical_key(1 << 48, 0, 0, 0)
    assert k1 == k2, f"Dihedral canonicalization failed: {k1} != {k2}"
    print(f"Verified D4 symmetry canonicalization: (0,0) and (6,6) map to identical canonical key: {k1}")
    
    idx = WdlAnchorIndex()
    # Mock row
    idx.table[k1] = {"row_idx": 0, "w": 1, "l": 1, "d": 0, "n": 2, "outcome_p": 0.5}
    # Ingest a new win from opposite corner
    updated, entry = idx.update_with_game_position(1 << 48, 0, 0, 0, game_result=1.0)
    assert updated == True
    assert entry["w"] == 2 and entry["n"] == 3
    print(f"Verified Stockfish WDL updating: new outcome_p = {entry['outcome_p']:.4f} (W={entry['w']}, L={entry['l']}, D={entry['d']}, N={entry['n']})")
    print("Stockfish WDL updater ready for production ingestion.")
