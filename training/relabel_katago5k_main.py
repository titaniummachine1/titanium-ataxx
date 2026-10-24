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
ENGINE = r"C:\gitProjects\main-ataxx\target\release\titanium-cli.exe"
TUPLE = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")
TMP = os.path.join(ROOT, "data", "nnue", "label_tmp_katago")


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
                f.write(board_line(r[0], r[1], r[2], r[3]) + "\n")
        with open(inp, "rb") as fin, open(out, "wb") as fout:
            p = subprocess.run(
                [engine, "label", "--nodes", str(nodes), "--tuple", tup, "--tuple-native"],
                stdin=fin, stdout=fout, stderr=subprocess.PIPE, timeout=1800,
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
        return [r[4] for r in rows], vals
    finally:
        for p in (inp, out):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0)
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
    con.execute("PRAGMA busy_timeout=120000")
    print("db %s nodes %d workers %d chunk %d" % (
        a.db, a.nodes, a.workers, a.chunk), flush=True)
    print("engine %s net %s" % (eng[:16], net[:16]), flush=True)

    total_pending = con.execute(
        "SELECT count(*) FROM positions WHERE relabel_status='pending'"
    ).fetchone()[0]
    target = min(total_pending, a.limit) if a.limit else total_pending
    print("pending katax rows %d target %d" % (total_pending, target), flush=True)
    if target == 0:
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
                "SELECT occ0,occ1,blk,stm,id FROM positions "
                "WHERE relabel_status='pending' AND id>? ORDER BY id LIMIT ?",
                (last, page),
            ).fetchall()
            if not rows:
                break
            last = rows[-1][4]
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
                updates = [(v[0], v[1], v[2], eng, net, int(time.time()), ids[i])
                           for i, v in enumerate(vals)]
                con.executemany(
                    "UPDATE positions SET engine_cp=?,engine_depth=?,engine_nodes=?,"
                    "relabel_status='done',label_engine_sha=?,label_net_sha=?,"
                    "labeled_at=? WHERE id=?", updates)
                con.commit()
                done += len(ids)
                rate = done / max(1e-9, time.time() - t0)
                print("katax done=%d/%d errors=%d %.1f rows/s %.0fs" % (
                    done, target, errs, rate, time.time() - t0), flush=True)
    print("run complete done=%d errors=%d elapsed=%.0fs" % (
        done, errs, time.time() - t0), flush=True)
    con.close()


if __name__ == "__main__":
    main()
