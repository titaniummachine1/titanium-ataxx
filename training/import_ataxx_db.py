import argparse
import os
import sqlite3
import time

import numpy as np
import sqlite3 as db

ROOT = r"C:\gitProjects\titanium-ataxx"
DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")
ATAXX = os.path.join(ROOT, "data", "nnue", "cache12M_ataxx_clean.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--ataxx", default=ATAXX)
    a = ap.parse_args()
    t0 = time.time()
    con = db.connect(a.db)
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    try:
        con.execute("ALTER TABLE positions ADD COLUMN ataxx_cp REAL")
        print("added ataxx_cp", flush=True)
    except Exception as e:
        print("ataxx_cp column:", e, flush=True)
    con.execute("DROP TABLE IF EXISTS ataxx_labels")
    con.execute("CREATE TABLE ataxx_labels (k0 INTEGER,k1 INTEGER,k2 INTEGER,k3 INTEGER,cp REAL)")
    z = np.load(a.ataxx, mmap_mode="r")
    n = len(z["occ0"])
    print("ataxx rows", n, flush=True)
    for lo in range(0, n, 50000):
        hi = min(n, lo + 50000)
        vals = []
        for i in range(lo, hi):
            q = np.clip(float(z["ataxx_q"][i]) * 1000.0, -1500.0, 1500.0)
            vals.append((int(z["occ0"][i]), int(z["occ1"][i]),
                         int(z["blk"][i]), int(z["stm"][i]), float(q)))
        con.executemany("INSERT INTO ataxx_labels VALUES (?,?,?,?,?)", vals)
        con.commit()
        if (lo // 50000) % 20 == 0:
            print("loaded", hi, n, "%.0fs" % (time.time() - t0), flush=True)
    con.execute("CREATE INDEX ataxx_labels_idx ON ataxx_labels(k0,k1,k2,k3)")
    con.commit()
    print("indexed %.0fs" % (time.time() - t0), flush=True)
    con.execute("""
        UPDATE positions SET ataxx_cp = a.cp
        FROM ataxx_labels a
        WHERE positions.occ0=a.k0 AND positions.occ1=a.k1
          AND positions.blk=a.k2 AND positions.stm=a.k3
    """)
    con.commit()
    matched = con.execute("SELECT count(1) FROM positions WHERE ataxx_cp IS NOT NULL").fetchone()[0]
    print("matched %d %.0fs" % (matched, time.time() - t0), flush=True)
    con.close()


if __name__ == "__main__":
    main()
