import argparse
import hashlib
import json
import os
from libsql_client import dbapi2 as sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")
DEFAULT_OUT = os.path.join(ROOT, "data", "nnue", "final")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(con, split, path):
    cur = con.execute(
        "SELECT occ0,occ1,blk,stm,outcome,target_cp,weight,phase,regret,"
        "engine_cp,deep_cp,outcome_p,wins,losses,draws,n_outcomes FROM positions "
        "WHERE split=? AND target_cp IS NOT NULL ORDER BY id", (split,))
    cols = [[] for _ in range(16)]
    n = 0
    for row in cur:
        for c, v in zip(cols, row):
            c.append(v)
        n += 1
        if n % 1000000 == 0:
            print("%s %d" % (split, n), flush=True)
    occ0 = np.asarray(cols[0], dtype=np.uint64)
    occ1 = np.asarray(cols[1], dtype=np.uint64)
    blk = np.asarray(cols[2], dtype=np.uint64)
    stm = np.asarray(cols[3], dtype=np.uint8)
    out = np.asarray(cols[4], dtype=np.float32)
    target = np.clip(np.asarray(cols[5], dtype=np.float32), -1500.0, 1500.0)
    weight = np.asarray(cols[6], dtype=np.float32)
    phase = np.asarray(cols[7], dtype=np.uint8)
    regret = np.asarray(cols[8], dtype=np.uint8)
    engine_cp = np.asarray(cols[9], dtype=np.float32)
    deep_cp = np.asarray(cols[10], dtype=np.float32)
    outcome_p = np.asarray(cols[11], dtype=np.float32)
    wins = np.asarray(cols[12], dtype=np.float32)
    losses = np.asarray(cols[13], dtype=np.float32)
    draws = np.asarray(cols[14], dtype=np.float32)
    n_outcomes = np.asarray(cols[15], dtype=np.uint32)
    empty = (49 - np.bitwise_count(occ0 | occ1 | blk)).astype(np.uint8)
    sc = (target / 2000.0).astype(np.float32)
    np.savez_compressed(
        path,
        occ0=occ0, occ1=occ1, blk=blk, stm=stm,
        out=out, outcome_p=outcome_p, wins=wins, losses=losses,
        draws=draws, n_outcomes=n_outcomes,
        teacher=target, sc=sc, wsum=weight,
        phase=phase, empty=empty, regret=regret,
        engine_cp=engine_cp, deep_cp=deep_cp,
    )
    print("%s rows=%d bytes=%d -> %s" % (split, n, os.path.getsize(path), path), flush=True)
    return n, os.path.getsize(path), sha256_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    meta = dict(con.execute("SELECT key,value FROM meta").fetchall())
    result = {"db": a.db, "splits": {}, "meta": meta, "created_at": time.time()}
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        path = os.path.join(a.out, "final_%s.npz" % split)
        n, size, sha = export(con, split, path)
        result["splits"][split] = {"rows": n, "bytes": size, "sha256": sha, "path": path}
    counts = {}
    for row in con.execute(
            "SELECT source,count(*) FROM positions WHERE target_cp IS NOT NULL GROUP BY source"):
        counts[row[0]] = int(row[1])
    result["source_counts"] = counts
    result["phase_counts"] = {
        str(p): int(n) for p, n in con.execute(
            "SELECT phase,count(*) FROM positions WHERE target_cp IS NOT NULL GROUP BY phase")
    }
    result["regret_rows"] = int(con.execute(
        "SELECT count(*) FROM positions WHERE target_cp IS NOT NULL AND regret=1").fetchone()[0])
    result["label_counts"] = {
        "engine": int(con.execute(
            "SELECT count(*) FROM positions WHERE engine_cp IS NOT NULL").fetchone()[0]),
        "deep": int(con.execute(
            "SELECT count(*) FROM positions WHERE deep_cp IS NOT NULL").fetchone()[0]),
    }
    result["target_mean"] = float(con.execute(
        "SELECT avg(target_cp) FROM positions WHERE target_cp IS NOT NULL").fetchone()[0])
    with open(os.path.join(a.out, "final_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
