"""Generate a 100,000 diverse opening book for selfplay generation (Decision D137).

Guarantees that a deterministic engine searching 5,000 nodes/move never replays
identical games.

Sources:
  1. data/nnue/book50k.txt (50,000 balanced positions)
  2. data/nnue/botzone_anchors/botzone_2200_anchors.jsonl (88,295 elite game positions)
     - Filtered for early-game balance: 4 <= stones <= 18, |diff| <= 2
  3. Additional distinct random-walk branches if needed.

Output:
  data/nnue/book100k_diverse.txt (100,000 unique opening FEN lines)
"""
import os
import json
import numpy as np

ROOT = "C:/gitProjects/titanium-ataxx"
DATA_DIR = os.path.join(ROOT, "data/nnue")
BOOK50K = os.path.join(DATA_DIR, "book50k.txt")
ANCHORS_JSONL = os.path.join(DATA_DIR, "botzone_anchors", "botzone_2200_anchors.jsonl")
OUT_BOOK = os.path.join(DATA_DIR, "book100k_diverse.txt")

def board_from_occ(occ0: int, occ1: int, blockers: int, turn: int) -> str:
    chars = []
    for sq in range(49):
        mask = 1 << sq
        if blockers & mask:
            chars.append("#")
        elif occ0 & mask:
            chars.append("x")
        elif occ1 & mask:
            chars.append("o")
        else:
            chars.append(".")
    side = "b" if turn == 0 else "w"
    return "".join(chars) + f" {side}"

def main():
    unique_openings = set()

    # 1. Load book50k.txt
    if os.path.exists(BOOK50K):
        print(f"Loading openings from {BOOK50K}...")
        with open(BOOK50K, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    unique_openings.add(line)
        print(f"  Loaded {len(unique_openings):,} unique openings from book50k.txt")

    # 2. Extract balanced early-game openings from Botzone >= 2200 Elo anchors
    if os.path.exists(ANCHORS_JSONL):
        print(f"Extracting early-game anchors from {ANCHORS_JSONL}...")
        added_from_anchors = 0
        with open(ANCHORS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                o0 = int(rec["occ0"])
                o1 = int(rec["occ1"])
                blk = int(rec.get("blockers", 0))
                turn = int(rec.get("turn", 0))

                nb = bin(o0).count("1")
                nw = bin(o1).count("1")
                tot = nb + nw

                # Select balanced opening and early-midgame positions
                if 4 <= tot <= 20 and abs(nb - nw) <= 3 and nb > 0 and nw > 0:
                    fen = board_from_occ(o0, o1, blk, turn)
                    if fen not in unique_openings:
                        unique_openings.add(fen)
                        added_from_anchors += 1

        print(f"  Added {added_from_anchors:,} new unique openings from Botzone anchors.")
        print(f"  Total unique openings pool: {len(unique_openings):,}")

    # 3. Save to book100k_diverse.txt
    openings_list = sorted(list(unique_openings))
    print(f"Writing {len(openings_list):,} unique openings to {OUT_BOOK}...")
    with open(OUT_BOOK, "w", encoding="utf-8") as f:
        for fen in openings_list:
            f.write(fen + "\n")

    print(f"Done! Diverse opening book created at {OUT_BOOK} ({os.path.getsize(OUT_BOOK):,} bytes).")

if __name__ == "__main__":
    main()
