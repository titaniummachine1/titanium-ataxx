import argparse
import heapq
import os
from libsql_client import dbapi2 as sqlite3
import time

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--per-phase", type=int, default=25000)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    if a.reset:
        con.execute("UPDATE positions SET deep_status='none',regret=0,deep_cp=NULL,"
                    "deep_depth=NULL,deep_nodes=NULL WHERE deep_status='done'")
        con.commit()

    heaps = {p: [] for p in range(4)}
    n = 0
    t0 = time.time()
    for pid, phase, cp, out in con.execute(
        "SELECT id,phase,engine_cp,outcome FROM positions "
        "WHERE relabel_status='done' AND engine_cp IS NOT NULL AND deep_cp IS NULL"
    ):
        oc = max(-1500.0, min(1500.0, (float(out) * 2.0 - 1.0) * 1000.0))
        residual = abs(float(cp) - oc)
        h = heaps[int(phase)]
        item = (residual, int(pid))
        if len(h) < a.per_phase:
            heapq.heappush(h, item)
        elif item[0] > h[0][0]:
            heapq.heapreplace(h, item)
        n += 1
        if n % 1000000 == 0:
            print("scanned %d (%.0fs)" % (n, time.time() - t0), flush=True)

    ids = []
    for p in range(4):
        ids.extend(x[1] for x in heaps[p])
        print("phase %d regret candidates %d" % (p, len(heaps[p])), flush=True)
    ids = sorted(set(ids))
    now = int(time.time())
    for i in range(0, len(ids), 20000):
        batch = ids[i:i + 20000]
        con.executemany(
            "UPDATE positions SET deep_status='pending',regret=1 WHERE id=?", 
            [(x,) for x in batch])
        con.commit()
        print("marked %d/%d" % (min(len(ids), i + len(batch)), len(ids)), flush=True)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                ("regret", str(len(ids))))
    con.commit()
    print("regret positions %d -> deep_status=pending" % len(ids), flush=True)
    print("elapsed %.0fs" % (time.time() - t0), flush=True)
    con.close()


if __name__ == "__main__":
    main()
