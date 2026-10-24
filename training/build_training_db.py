import argparse
import hashlib
import json
import os
from libsql_client import dbapi2 as sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
NNUE = os.path.join(ROOT, "data", "nnue")
DEFAULT_DB = os.path.join(NNUE, "titanium_train.db")
SCHEMA = os.path.join(ROOT, "training", "dataset_schema.sql")
KATAGO = [
    os.path.join(NNUE, "katago_2026_b12c384_s3413299200.npz"),
    os.path.join(NNUE, "katago_2026_b12c384_s1835540480.npz"),
    os.path.join(NNUE, "katago_2026_b12c384_s1914773504.npz"),
]
CACHE = os.path.join(NNUE, "cache12M_clean.npz")


def split_for(occ0, occ1, blk, stm):
    b = b"".join(int(x).to_bytes(8, "little", signed=False) for x in (occ0, occ1, blk))
    b += bytes([int(stm)])
    h = int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little") % 100
    if h < 2:
        return "test"
    if h < 4:
        return "val"
    return "train"


def phase_for(empty):
    if empty >= 30:
        return 3
    if empty >= 15:
        return 2
    if empty >= 8:
        return 1
    return 0


def import_npz(con, path, source, limit, batch):
    z = np.load(path, mmap_mode="r")
    n = len(z["occ0"])
    if limit:
        n = min(n, limit)
    base = os.path.basename(path)
    row = con.execute(
        "SELECT max(source_row) FROM positions WHERE source=? AND source_file=?",
        (source, base),
    ).fetchone()[0]
    start = int(row) + 1 if row is not None else 0
    if start:
        print("%s resume at source_row=%d/%d" % (base, start, n), flush=True)
    occ0 = z["occ0"]
    occ1 = z["occ1"]
    blk = z["blk"]
    stm = z["stm"]
    out = z["out"]
    old = z["teacher"] if "teacher" in z else None
    sql = (
        "INSERT OR IGNORE INTO positions "
        "(occ0,occ1,blk,stm,empty,phase,outcome,source,source_file,source_row,split,"
        "old_teacher,relabel_status,deep_status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    kept = 0
    skipped = 0
    inserted = 0
    t0 = time.time()
    for lo in range(start, n, batch):
        hi = min(n, lo + batch)
        rows = []
        for i in range(lo, hi):
            a0 = int(occ0[i])
            a1 = int(occ1[i])
            ab = int(blk[i])
            s = int(stm[i])
            o = float(out[i])
            if a0 == 0 or a1 == 0 or (a0 & a1) or (ab & (a0 | a1)):
                skipped += 1
                continue
            occupied = a0 | a1 | ab
            empty = 49 - occupied.bit_count()
            if empty < 1 or empty > 49 or not np.isfinite(o) or o < 0.0 or o > 1.0:
                skipped += 1
                continue
            rows.append((
                a0, a1, ab, s, empty, phase_for(empty), o, source,
                os.path.basename(path), i, split_for(a0, a1, ab, s),
                float(old[i]) if old is not None else None,
                "pending", "none",
            ))
        before = con.total_changes
        con.executemany(sql, rows)
        con.commit()
        inserted += con.total_changes - before
        kept += len(rows)
        if (lo // batch) % 10 == 0:
            print("%s %d/%d kept=%d inserted=%d skipped=%d (%.0fs)" % (
                source, hi, n, kept, inserted, skipped, time.time() - t0), flush=True)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                ("source:" + source, json.dumps({
                    "path": path, "rows": int(n), "usable": int(kept),
                    "inserted": int(inserted), "skipped": int(skipped),
                })))
    con.commit()
    print("%s DONE rows=%d usable=%d inserted=%d skipped=%d" % (
        source, n, kept, inserted, skipped), flush=True)
    return n, kept, inserted, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit-per-source", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20000)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.db), exist_ok=True)
    con = sqlite3.connect(a.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    with open(SCHEMA, "r", encoding="utf-8") as f:
        con.executescript(f.read())
    con.commit()

    total_in = total_kept = total_ins = total_skip = 0
    sources = []
    for p in KATAGO:
        if not os.path.exists(p):
            print("missing", p, flush=True)
            continue
        n, k, ins, sk = import_npz(con, p, "katago_202604", a.limit_per_source, a.batch)
        total_in += n
        total_kept += k
        total_ins += ins
        total_skip += sk
        sources.append(p)
    if not a.no_cache and os.path.exists(CACHE):
        n, k, ins, sk = import_npz(con, CACHE, "titanium_cache12m", a.limit_per_source, a.batch)
        total_in += n
        total_kept += k
        total_ins += ins
        total_skip += sk
        sources.append(CACHE)

    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                ("build", json.dumps({
                    "inputs": sources, "rows_read": total_in, "usable": total_kept,
                    "inserted_unique": total_ins, "skipped": total_skip,
                    "created_at": time.time(),
                })))
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                ("schema_version", "1"))
    con.commit()
    print("DB %s size %.1fMB rows=%d" % (
        a.db, os.path.getsize(a.db) / 1e6,
        con.execute("SELECT count(*) FROM positions").fetchone()[0]), flush=True)
    for row in con.execute("SELECT source,count(*) FROM positions GROUP BY source"):
        print("source", row[0], row[1], flush=True)
    for row in con.execute("SELECT split,count(*) FROM positions GROUP BY split"):
        print("split", row[0], row[1], flush=True)
    for row in con.execute("SELECT phase,count(*) FROM positions GROUP BY phase"):
        print("phase", row[0], row[1], flush=True)
    con.close()


if __name__ == "__main__":
    main()
