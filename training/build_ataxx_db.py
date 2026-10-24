import os
import sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
NNUE = os.path.join(ROOT, "data", "nnue")
NPZ = os.path.join(NNUE, "cache12M_ataxx_clean.npz")
DB = os.path.join(NNUE, "ataxx.db")


def main():
    if os.path.exists(DB):
        os.unlink(DB)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("CREATE TABLE ataxx (o0 INTEGER,o1 INTEGER,blk INTEGER,stm INTEGER,cp REAL)")
    z = np.load(NPZ, mmap_mode="r")
    O0 = z["occ0"]; O1 = z["occ1"]; BLK = z["blk"]; STM = z["stm"]; Q = z["ataxx_q"]
    n = len(O0)
    t0 = time.time()
    for lo in range(0, n, 100000):
        hi = min(n, lo + 100000)
        vals = []
        for i in range(lo, hi):
            q = float(Q[i])
            vals.append((int(O0[i]), int(O1[i]), int(BLK[i]),
                         int(STM[i]), q * 1000.0))
        con.executemany("INSERT INTO ataxx VALUES (?,?,?,?,?)", vals)
        con.commit()
        if (lo // 100000) % 50 == 0:
            print("loaded", hi, n, "%.0fs" % (time.time() - t0), flush=True)
    con.execute("CREATE INDEX ataxx_idx ON ataxx(o0,o1,blk,stm)")
    con.commit()
    print("done rows", n, "bytes", os.path.getsize(DB), "%.0fs" % (time.time() - t0))
    con.close()


if __name__ == "__main__":
    main()
