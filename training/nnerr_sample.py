"""Sample positions from final_val_v3.npz for the cheap-vs-big-net error study: writes data/nnerr/boards.txt (one board
string per row) and rows.npy (the row ids, to join teacher columns later). Usage: python training/nnerr_sample.py N"""
import sys
import numpy as np
sys.path.insert(0, "training")
from ataxx_move import board_str

n = int(sys.argv[1])
z = np.load("data/nnue/final/final_val_v3.npz")
o0, o1, blk, stm = z["occ0"], z["occ1"], z["blk"], z["stm"]
rows = np.sort(np.random.default_rng(7).choice(len(o0), n, replace=False))
with open("data/nnerr/boards.txt", "w") as f:
    for r in rows:
        f.write(board_str(int(o0[r]), int(o1[r]), int(blk[r]), int(stm[r])) + "\n")
np.save("data/nnerr/rows.npy", rows)
print("wrote", n)
