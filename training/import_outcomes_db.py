import argparse
import json
import os
import time

import numpy as np
import sqlite3 as db

ROOT = r"C:\gitProjects\titanium-ataxx"
DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")
OUT = os.path.join(ROOT, "data", "nnue", "position_outcomes.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--outcomes", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    con = db.connect(a.db)
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    for col, typ in (("wins", "REAL"), ("losses", "REAL"), ("draws", "REAL"),
                     ("n_outcomes", "INTEGER"), ("outcome_p", "REAL")):
        try:
            con.execute("ALTER TABLE positions ADD COLUMN %s %s" % (col, typ))
            print("added", col, flush=True)
        except Exception as e:
            print("column", col, e, flush=True)
    z = np.load(a.outcomes, mmap_mode="r")
    n = len(z["occ0"])
    con.execute("DROP TABLE IF EXISTS outcomes")
    con.execute("CREATE TABLE outcomes (o0 INTEGER,o1 INTEGER,stm INTEGER,"
                "wins REAL,losses REAL,draws REAL,n INTEGER,outcome_p REAL)")
    print("outcome rows", n, flush=True)
    for lo in range(0, n, 50000):
        hi = min(n, lo + 50000)
        vals = []
        for i in range(lo, hi):
            nn = int(z["n"][i])
            w = float(z["wins"][i])
            d = float(z["draws"][i])
            l = float(z["losses"][i])
            vals.append((int(z["occ0"][i]), int(z["occ1"][i]), int(z["stm"][i]),
                         w, l, d, nn, (w + 0.5 * d) / max(nn, 1)))
        con.executemany("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)", vals)
        con.commit()
        if (lo // 50000) % 20 == 0:
            print("loaded", hi, n, "%.0fs" % (time.time() - t0), flush=True)
    con.execute("CREATE INDEX outcomes_idx ON outcomes(o0,o1,stm)")
    con.commit()
    print("indexed %.0fs" % (time.time() - t0), flush=True)
    con.execute("""
        UPDATE positions SET wins=o.wins, losses=o.losses, draws=o.draws,
            n_outcomes=o.n, outcome_p=o.outcome_p
        FROM outcomes o
        WHERE positions.occ0=o.o0 AND positions.occ1=o.o1
          AND positions.stm=o.stm
    """)
    con.commit()
    c = con.execute("SELECT count(1) FROM positions WHERE n_outcomes IS NOT NULL").fetchone()[0]
    print("matched", c, "%.0fs" % (time.time() - t0), flush=True)
    with open(os.path.join(ROOT, "data", "nnue", "outcomes_import_manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump({"rows": int(n), "matched": int(c),
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    con.close()


if __name__ == "__main__":
    main()
