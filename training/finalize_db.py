import argparse
import json
import os
from libsql_client import dbapi2 as sqlite3
import time

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    t0 = time.time()
    con.execute("""
        UPDATE positions SET
            target_cp = MAX(-1500.0, MIN(1500.0,
                COALESCE(ataxx_cp, deep_cp, engine_cp))),
            weight = CASE WHEN deep_cp IS NOT NULL THEN 1.5 ELSE 1.0 END,
            quality_flags =
                CASE WHEN ABS(engine_cp) > 2500 THEN 1 ELSE 0 END
                | CASE WHEN ABS(engine_cp - (outcome * 2.0 - 1.0) * 1000.0) > 2500 THEN 2 ELSE 0 END
        WHERE relabel_status='done' AND engine_cp IS NOT NULL
    """)
    con.commit()
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                ("finalize", json.dumps({
                    "engine_weight": 0.75,
                    "outcome_weight": 0.25,
                    "deep_engine_weight": 0.75,
                    "regret_weight": 1.5,
                    "created_at": time.time(),
                })))
    con.commit()
    print("finalized in %.0fs" % (time.time() - t0), flush=True)
    for q in [
        "SELECT count(*) FROM positions WHERE target_cp IS NOT NULL",
        "SELECT count(*) FROM positions WHERE deep_cp IS NOT NULL",
        "SELECT count(*) FROM positions WHERE regret=1",
        "SELECT split,count(*) FROM positions WHERE target_cp IS NOT NULL GROUP BY split",
        "SELECT count(*) FROM positions WHERE relabel_status!='done'",
    ]:
        print(q, con.execute(q).fetchall(), flush=True)
    con.close()


if __name__ == "__main__":
    main()
