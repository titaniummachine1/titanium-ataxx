import argparse
import glob
import json
import os
from libsql_client import dbapi2 as sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")
GOLDEN = os.path.join(ROOT, "data", "nnue", "golden.npz")
LAB2M = os.path.join(ROOT, "data", "nnue")


def key(o0, o1, b, s):
    return (int(o0), int(o1), int(b), int(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--lab2m-dir", default=LAB2M)
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    con.execute("PRAGMA busy_timeout=120000")
    t0 = time.time()
    now = int(time.time())
    con.execute("CREATE TEMP TABLE IF NOT EXISTS deep (k0 INTEGER,k1 INTEGER,k2 INTEGER,k3 INTEGER,cp REAL,nodes INTEGER)")
    con.execute("CREATE INDEX IF NOT EXISTS deep_idx ON deep(k0,k1,k2,k3)")
    inserted = 0

    def flush(rows):
        nonlocal inserted
        con.executemany("INSERT INTO deep(k0,k1,k2,k3,cp,nodes) VALUES (?,?,?,?,?,?)", rows)
        inserted += len(rows)

    if os.path.exists(a.golden):
        z = np.load(a.golden)
        rows = []
        for i in range(len(z["occ0"])):
            rows.append((int(z["occ0"][i]), int(z["occ1"][i]), int(z["blk"][i]),
                         int(z["stm"][i]), float(z["teacher"][i]), 500000))
        flush(rows)
        print("golden %d" % len(rows), flush=True)

    pairs = sorted(glob.glob(os.path.join(a.lab2m_dir, "lab2m_*[ab].txt")))
    lab_rows = 0
    for txt in pairs:
        out = txt[:-4] + ".out"
        if not os.path.exists(out):
            continue
        with open(txt, "r", encoding="ascii") as f:
            boards = [line.rstrip("\n") for line in f]
        scores = np.loadtxt(out, dtype=np.float32, ndmin=2)
        if len(scores) != len(boards):
            print("SKIP %s rows %d != %d" % (os.path.basename(txt), len(scores), len(boards)), flush=True)
            continue
        rows = []
        for line, sc in zip(boards, scores):
            s = line.strip()
            if len(s) != 51 or s[49] != " " or s[50] not in "bw":
                continue
            o0 = o1 = b = 0
            for q, ch in enumerate(s[:49]):
                if ch == "x":
                    o0 |= (1 << q)
                elif ch == "o":
                    o1 |= (1 << q)
                elif ch == "#":
                    b |= (1 << q)
            rows.append((o0, o1, b, 0 if s[50] == "b" else 1, float(sc[0]), 20000))
        flush(rows)
        lab_rows += len(rows)
        print("%s %d (total %d) %.0fs" % (os.path.basename(txt), len(rows), inserted, time.time() - t0), flush=True)
    con.commit()
    con.execute(
        "UPDATE positions SET deep_cp=d.cp,deep_nodes=d.nodes,deep_status='done',regret=1 "
        "FROM deep d WHERE positions.occ0=d.k0 AND positions.occ1=d.k1 "
        "AND positions.blk=d.k2 AND positions.stm=d.k3")
    con.commit()
    matched = con.execute(
        "SELECT count(*) FROM positions WHERE deep_status='done'").fetchone()[0]
    manifest = {
        "golden": os.path.exists(a.golden),
        "lab2m_pairs": len(pairs),
        "deep_rows_inserted": int(inserted),
        "lab2m_rows": int(lab_rows),
        "matched_positions": int(matched),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ROOT, "data", "nnue", "deep_import_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
