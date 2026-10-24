import os
import sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
NNUE = os.path.join(ROOT, "data", "nnue")
NPZ = os.path.join(NNUE, "position_outcomes.npz")
DB = os.path.join(NNUE, "outcomes.db")


def main():
    if os.path.exists(DB):
        os.unlink(DB)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("CREATE TABLE outcomes (o0 INTEGER,o1 INTEGER,stm INTEGER,"
                "wins REAL,losses REAL,draws REAL,n INTEGER,outcome_p REAL)")
    z = np.load(NPZ, mmap_mode="r")
    O0 = z["occ0"]; O1 = z["occ1"]; STM = z["stm"]
    WINS = z["wins"]; LOSSES = z["losses"]; DRAWS = z["draws"]; NN = z["n"]
    n = len(O0)
    t0 = time.time()
    for lo in range(0, n, 100000):
        hi = min(n, lo + 100000)
        vals = []
        for i in range(lo, hi):
            nn = int(NN[i])
            w = float(WINS[i])
            d = float(DRAWS[i])
            l = float(LOSSES[i])
            vals.append((int(O0[i]), int(O1[i]), int(STM[i]),
                         w, l, d, nn, (w + 0.5 * d) / max(nn, 1)))
        con.executemany("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)", vals)
        con.commit()
        if (lo // 100000) % 50 == 0:
            print("loaded", hi, n, "%.0fs" % (time.time() - t0), flush=True)
    con.execute("CREATE INDEX outcomes_idx ON outcomes(o0,o1,stm)")
    con.commit()
    print("done rows", n, "bytes", os.path.getsize(DB), "%.0fs" % (time.time() - t0))
    con.close()


if __name__ == "__main__":
    main()
