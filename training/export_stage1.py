import argparse
import json
import os
import time

import numpy as np
from libsql_client import dbapi2 as db

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")
OUT = os.path.join(ROOT, "data", "nnue")


def export(con, split, path):
    cur = con.execute(
        "SELECT occ0,occ1,blk,stm,outcome,COALESCE(engine_cp,old_teacher),phase "
        "FROM positions WHERE source='titanium_cache12m' AND split=? "
        "AND COALESCE(engine_cp,old_teacher) IS NOT NULL "
        "AND abs(COALESCE(engine_cp,old_teacher)) <= 5000 ORDER BY id", (split,))
    cols = [[], [], [], [], [], [], []]
    n = 0
    for row in cur:
        for c, v in zip(cols, row):
            c.append(v)
        n += 1
        if n % 1000000 == 0:
            print(split, n, flush=True)
    occ0 = np.asarray(cols[0], dtype=np.uint64)
    occ1 = np.asarray(cols[1], dtype=np.uint64)
    blk = np.asarray(cols[2], dtype=np.uint64)
    stm = np.asarray(cols[3], dtype=np.uint8)
    out = np.asarray(cols[4], dtype=np.float32)
    teacher = np.asarray(cols[5], dtype=np.float32)
    phase = np.asarray(cols[6], dtype=np.uint8)
    empty = (49 - np.bitwise_count(occ0 | occ1 | blk)).astype(np.uint8)
    np.savez_compressed(
        path, occ0=occ0, occ1=occ1, blk=blk, stm=stm,
        out=out, teacher=teacher, sc=(teacher / 2000.0).astype(np.float32),
        wsum=np.ones(n, dtype=np.float32), phase=phase, empty=empty,
    )
    print("%s rows=%d bytes=%d -> %s" % (split, n, os.path.getsize(path), path), flush=True)
    return n, os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    con = db.connect(a.db)
    con.execute("PRAGMA busy_timeout=120000")
    result = {"db": a.db, "splits": {}, "created_at": time.time()}
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        path = os.path.join(a.out_dir, "stage1_base_%s.npz" % split)
        n, size = export(con, split, path)
        result["splits"][split] = {"rows": n, "bytes": size, "path": path}
    with open(os.path.join(a.out_dir, "stage1_base_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
