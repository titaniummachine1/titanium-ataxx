import argparse
import os
from libsql_client import dbapi2 as sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--max-mae", type=float, default=100.0)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=120000")
    rows = con.execute(
        "SELECT old_teacher,engine_cp,outcome FROM positions "
        "WHERE source='titanium_cache12m' AND old_teacher IS NOT NULL "
        "AND engine_cp IS NOT NULL ORDER BY id LIMIT ?", (a.sample,)).fetchall()
    if not rows:
        print("no sampled legacy rows", flush=True)
        return
    old = np.asarray([r[0] for r in rows], dtype=np.float64)
    cur = np.asarray([r[1] for r in rows], dtype=np.float64)
    out = np.asarray([r[2] for r in rows], dtype=np.float64)
    mae = float(np.mean(np.abs(old - cur)))
    corr = float(np.corrcoef(old, cur)[0, 1]) if old.std() > 0 and cur.std() > 0 else float("nan")
    corr_out = float(np.corrcoef(old, out)[0, 1]) if old.std() > 0 and out.std() > 0 else float("nan")
    print("sample=%d old_vs_current_mae=%.1f corr=%.4f old_outcome_corr=%.4f" % (
        len(rows), mae, corr, corr_out), flush=True)
    ok = mae <= a.max_mae and corr >= 0.8
    print("legacy_acceptable=%s (mae<=%.0f corr>=0.8)" % (ok, a.max_mae), flush=True)
    if not (a.apply and ok):
        con.close()
        return
    now = int(time.time())
    con.execute(
        "UPDATE positions SET engine_cp=old_teacher,engine_depth=NULL,engine_nodes=5000,"
        "relabel_status='done',label_engine_sha='legacy_cache12m_5k',"
        "label_net_sha=NULL,labeled_at=?,quality_flags=quality_flags|4 "
        "WHERE source='titanium_cache12m' AND old_teacher IS NOT NULL "
        "AND relabel_status!='done'", (now,))
    con.commit()
    print("promoted legacy cache rows", flush=True)
    con.close()


if __name__ == "__main__":
    main()
