import argparse
import json
import os
from libsql_client import dbapi2 as sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "nnue", "final_audit.json"))
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    t0 = time.time()
    r = {"db": a.db, "created_at": time.time()}

    def one(q):
        return con.execute(q).fetchone()

    def rows(q):
        return [list(x) for x in con.execute(q).fetchall()]

    r["rows"] = int(one("SELECT count(*) FROM positions")[0])
    r["sources"] = rows("SELECT source,count(*) FROM positions GROUP BY source")
    r["splits"] = rows("SELECT split,count(*) FROM positions GROUP BY split")
    r["phases"] = rows("SELECT phase,count(*) FROM positions GROUP BY phase")
    r["outcome"] = list(map(float, one(
        "SELECT avg(outcome),min(outcome),max(outcome) FROM positions")[:3]))
    r["labels"] = {
        "engine": int(one("SELECT count(*) FROM positions WHERE engine_cp IS NOT NULL")[0]),
        "deep": int(one("SELECT count(*) FROM positions WHERE deep_cp IS NOT NULL")[0]),
        "target": int(one("SELECT count(*) FROM positions WHERE target_cp IS NOT NULL")[0]),
        "pending_engine": int(one(
            "SELECT count(*) FROM positions WHERE relabel_status='pending'")[0]),
        "pending_deep": int(one(
            "SELECT count(*) FROM positions WHERE deep_status='pending'")[0]),
        "regret": int(one("SELECT count(*) FROM positions WHERE regret=1")[0]),
        "quality_flags": rows(
            "SELECT quality_flags,count(*) FROM positions GROUP BY quality_flags ORDER BY quality_flags"),
    }
    r["engine_hashes"] = rows(
        "SELECT label_engine_sha,label_net_sha,engine_nodes,count(*) FROM positions "
        "WHERE engine_cp IS NOT NULL GROUP BY 1,2,3")
    r["deep_hashes"] = rows(
        "SELECT deep_nodes,count(*) FROM positions WHERE deep_cp IS NOT NULL GROUP BY 1")

    def corr(q):
        data = con.execute(q).fetchall()
        if not data:
            return None
        x = np.asarray([v[0] for v in data], dtype=np.float64)
        y = np.asarray([v[1] for v in data], dtype=np.float64)
        if x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    r["corr_engine_outcome"] = corr(
        "SELECT outcome,engine_cp FROM positions WHERE engine_cp IS NOT NULL LIMIT 300000")
    r["corr_target_outcome"] = corr(
        "SELECT outcome,target_cp FROM positions WHERE target_cp IS NOT NULL LIMIT 300000")
    r["corr_old_engine"] = corr(
        "SELECT old_teacher,engine_cp FROM positions WHERE old_teacher IS NOT NULL "
        "AND engine_cp IS NOT NULL LIMIT 300000")
    r["old_engine_mae"] = None
    v = one("SELECT avg(abs(old_teacher-engine_cp)) FROM positions "
            "WHERE old_teacher IS NOT NULL AND engine_cp IS NOT NULL")
    if v and v[0] is not None:
        r["old_engine_mae"] = float(v[0])
    r["target_stats"] = {
        "mean": float(one("SELECT avg(target_cp) FROM positions WHERE target_cp IS NOT NULL")[0]),
        "min": float(one("SELECT min(target_cp) FROM positions WHERE target_cp IS NOT NULL")[0]),
        "max": float(one("SELECT max(target_cp) FROM positions WHERE target_cp IS NOT NULL")[0]),
    }
    r["duplicates"] = int(one(
        "SELECT count(*) - count(distinct occ0 || ':' || occ1 || ':' || blk || ':' || stm) "
        "FROM positions")[0])
    r["invalid_boards"] = int(one(
        "SELECT count(*) FROM positions "
        "WHERE occ0=0 OR occ1=0 OR (occ0 & occ1)!=0 OR (blk & (occ0|occ1))!=0")[0])
    r["empty_range"] = list(one(
        "SELECT min(empty),max(empty) FROM positions"))
    r["integrity"] = one("PRAGMA integrity_check")[0]
    r["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(r, indent=2), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    con.close()


if __name__ == "__main__":
    main()
