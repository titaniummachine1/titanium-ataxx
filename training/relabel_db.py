import argparse
import hashlib
import os
from libsql_client import dbapi2 as sqlite3
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_DB = os.path.join(ROOT, "data", "nnue", "titanium_train.db")
ENGINE = os.path.join(ROOT, "target", "release", "titanium-cli.exe")
TUPLE = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")
TMP = os.path.join(ROOT, "data", "nnue", "label_tmp")


def board_line(o0, o1, blk, stm):
    return "".join(
        "#" if (blk >> q) & 1 else "x" if (o0 >> q) & 1 else "o" if (o1 >> q) & 1 else "."
        for q in range(49)
    ) + (" b" if stm == 0 else " w")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_rows(rows, nodes, engine, tup):
    os.makedirs(TMP, exist_ok=True)
    fd, inp = tempfile.mkstemp(prefix="in_", suffix=".txt", dir=TMP)
    fd2, out = tempfile.mkstemp(prefix="out_", suffix=".txt", dir=TMP)
    os.close(fd)
    os.close(fd2)
    try:
        with open(inp, "w", encoding="ascii") as f:
            for r in rows:
                f.write(board_line(r[1], r[2], r[3], r[4]) + "\n")
        with open(inp, "rb") as fin, open(out, "wb") as fout:
            p = subprocess.run(
                [engine, "label", "--nodes", str(nodes), "--tuple", tup, "--tuple-native"],
                stdin=fin, stdout=fout, stderr=subprocess.PIPE, timeout=900,
            )
        if p.returncode != 0:
            raise RuntimeError("engine exit %d: %s" % (
                p.returncode, p.stderr.decode("utf-8", "replace")[-500:]))
        vals = []
        with open(out, "r", encoding="ascii") as f:
            for line in f:
                t = line.split()
                if len(t) < 3:
                    raise RuntimeError("bad label output: %r" % line)
                vals.append((int(t[0]), int(t[1]), int(t[2])))
        if len(vals) != len(rows):
            raise RuntimeError("label count %d != %d" % (len(vals), len(rows)))
        return [r[0] for r in rows], vals
    finally:
        for p in (inp, out):
            try:
                os.unlink(p)
            except OSError:
                pass


def audit(con, label):
    n = con.execute("SELECT count(*) FROM positions WHERE relabel_status='done'").fetchone()[0]
    if n == 0:
        print("audit: no labels yet", flush=True)
        return
    rows = con.execute(
        "SELECT outcome,engine_cp FROM positions WHERE relabel_status='done' "
        "AND engine_cp IS NOT NULL ORDER BY id LIMIT 200000"
    ).fetchall()
    import numpy as np
    if not rows:
        return
    out = np.asarray([r[0] for r in rows], dtype=np.float64)
    cp = np.asarray([r[1] for r in rows], dtype=np.float64)
    corr = float(np.corrcoef(out, cp)[0, 1]) if out.std() > 0 and cp.std() > 0 else float("nan")
    print("audit %s: labels=%d sample=%d corr(outcome,cp)=%.4f mean_cp=%+.1f" % (
        label, n, len(rows), corr, cp.mean()), flush=True)
    if corr < 0.05:
        print("AUDIT WARNING: weak/negative correlation; inspect board orientation/labels", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--source", default="")
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--tuple", default=TUPLE)
    a = ap.parse_args()

    eng = sha256_file(a.engine)
    net = sha256_file(a.tuple)
    con = sqlite3.connect(a.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    status_col = "deep_status" if a.deep else "relabel_status"
    mode = "deep" if a.deep else "5k"
    print("db %s mode %s nodes %d workers %d chunk %d" % (
        a.db, mode, a.nodes, a.workers, a.chunk), flush=True)
    print("engine %s net %s" % (eng[:16], net[:16]), flush=True)

    source_clause = " AND source=?" if a.source else ""
    source_params = (a.source,) if a.source else ()
    total_pending = con.execute(
        "SELECT count(*) FROM positions WHERE %s='pending'%s" % (status_col, source_clause),
        source_params,
    ).fetchone()[0]
    target = min(total_pending, a.limit) if a.limit else total_pending
    print("pending %d target %d" % (total_pending, target), flush=True)
    if target == 0:
        audit(con, mode)
        con.close()
        return

    done = 0
    errs = 0
    t0 = time.time()
    last = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        while done + errs < target:
            page = min(a.chunk * a.workers, target - done - errs)
            rows = con.execute(
                "SELECT id,occ0,occ1,blk,stm FROM positions "
                "WHERE %s='pending'%s AND id>? ORDER BY id LIMIT ?" % (status_col, source_clause),
                source_params + (last, page),
            ).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            jobs = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
            futs = {ex.submit(label_rows, j, a.nodes, a.engine, a.tuple): j for j in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    ids, vals = fut.result()
                except Exception as e:
                    errs += len(job)
                    print("ERROR rows=%d %s" % (len(job), e), flush=True)
                    continue
                now = int(time.time())
                if a.deep:
                    updates = [(v[0], v[1], v[2], now, ids[i]) for i, v in enumerate(vals)]
                    con.executemany(
                        "UPDATE positions SET deep_cp=?,deep_depth=?,deep_nodes=?,"
                        "deep_status='done',regret=1,labeled_at=? WHERE id=?", updates)
                else:
                    updates = [(v[0], v[1], v[2], eng, net, now, ids[i])
                               for i, v in enumerate(vals)]
                    con.executemany(
                        "UPDATE positions SET engine_cp=?,engine_depth=?,engine_nodes=?,"
                        "relabel_status='done',label_engine_sha=?,label_net_sha=?,"
                        "labeled_at=? WHERE id=?", updates)
                con.commit()
                done += len(ids)
                rate = done / max(1e-9, time.time() - t0)
                print("%s done=%d/%d errors=%d %.1f rows/s %.0fs" % (
                    mode, done, target, errs, rate, time.time() - t0), flush=True)
    print("run complete done=%d errors=%d elapsed=%.0fs" % (
        done, errs, time.time() - t0), flush=True)
    audit(con, mode)
    con.close()


if __name__ == "__main__":
    main()
