import json
import os
import sqlite3
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
NNUE = os.path.join(ROOT, "data", "nnue")
DB = os.path.join(NNUE, "titanium_train.db")
OUT_DB = os.path.join(NNUE, "outcomes.db")
ATAXX_DB = os.path.join(NNUE, "ataxx.db")
OUT = os.path.join(NNUE, "final")
FIELDS = ("occ0", "occ1", "blk", "stm", "out", "outcome_p", "wins", "losses",
          "draws", "n_outcomes", "teacher", "titanium_cp", "sc", "wsum",
          "teacher_src")
DTYPES = {"occ0": np.uint64, "occ1": np.uint64, "blk": np.uint64, "stm": np.uint8,
          "out": np.float32, "outcome_p": np.float32, "wins": np.float32,
          "losses": np.float32, "draws": np.float32, "n_outcomes": np.uint32,
          "teacher": np.float32, "titanium_cp": np.float32, "sc": np.float32,
          "wsum": np.float32, "teacher_src": np.uint8}


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA cache_size=-200000")
    con.execute("ATTACH DATABASE ? AS o", (OUT_DB,))
    con.execute("ATTACH DATABASE ? AS a", (ATAXX_DB,))
    q = """
        SELECT p.occ0,p.occ1,p.blk,p.stm,p.outcome,p.engine_cp,p.deep_cp,p.split,p.source,
               o.wins,o.losses,o.draws,o.n,o.outcome_p,a.cp
        FROM positions p
        LEFT JOIN o.outcomes o
          ON p.occ0=o.o0 AND p.occ1=o.o1 AND p.stm=o.stm
        LEFT JOIN a.ataxx a
          ON p.occ0=a.o0 AND p.occ1=a.o1 AND p.blk=a.blk AND p.stm=a.stm
    """
    cur = con.execute(q)
    chunks = {s: {k: [] for k in FIELDS} for s in ("train", "val", "test")}
    stats = {s: {"rows": 0, "ataxx": 0, "outcomes": 0, "deep": 0} for s in chunks}
    t0 = time.time()
    n = 0
    while True:
        rows = cur.fetchmany(100000)
        if not rows:
            break
        buf = {s: {k: [] for k in FIELDS} for s in chunks}
        for r in rows:
            occ0, occ1, blk, stm, out, ecp, dcp, split, src, wins, losses, draws, nn, op, acp = r
            if split not in buf:
                continue
            out = float(out)
            ecp = float("nan") if ecp is None else float(ecp)
            dcp = float("nan") if dcp is None else float(dcp)
            tit = dcp if np.isfinite(dcp) else ecp
            if op is not None:
                outcome_p = float(op)
                stats[split]["outcomes"] += 1
            else:
                outcome_p = out
                wins, losses, draws, nn = out, 1.0 - out, 0.0, 1
            if src == "katago_202604":
                # Keep the existing KataGo teacher; do not relabel it.
                teacher = float(np.clip((outcome_p * 2.0 - 1.0) * 1000.0, -1500.0, 1500.0))
                tsrc = 1
            elif acp is not None:
                teacher = float(acp)
                tsrc = 0
                stats[split]["ataxx"] += 1
            elif np.isfinite(dcp):
                teacher = dcp
                tsrc = 2
            else:
                teacher = ecp
                tsrc = 3
            if np.isfinite(dcp):
                stats[split]["deep"] += 1
            b = buf[split]
            b["occ0"].append(int(occ0)); b["occ1"].append(int(occ1))
            b["blk"].append(int(blk)); b["stm"].append(int(stm))
            b["out"].append(out); b["outcome_p"].append(outcome_p)
            b["wins"].append(float(wins)); b["losses"].append(float(losses))
            b["draws"].append(float(draws)); b["n_outcomes"].append(int(nn))
            b["teacher"].append(float(teacher)); b["titanium_cp"].append(float(tit))
            b["teacher_src"].append(int(tsrc))
            b["sc"].append(float(teacher) / 2000.0)
            b["wsum"].append(float(np.sqrt(max(1, int(nn)))))
            stats[split]["rows"] += 1
        for split, b in buf.items():
            if not b["occ0"]:
                continue
            for k in FIELDS:
                chunks[split][k].append(np.asarray(b[k], dtype=DTYPES[k]))
        n += len(rows)
        if (n // 1000000) != ((n - len(rows)) // 1000000):
            print("joined", n, "%.0fs" % (time.time() - t0), flush=True)
    manifest = {"rows": n, "splits": {}, "elapsed_s": round(time.time() - t0, 1)}
    for split, d in chunks.items():
        arr = {k: np.concatenate(v) if v else np.asarray([], dtype=DTYPES[k])
               for k, v in d.items()}
        path = os.path.join(OUT, "final_%s.npz" % split)
        np.savez_compressed(path, **arr)
        manifest["splits"][split] = {"rows": int(len(arr["occ0"])),
                                     "bytes": os.path.getsize(path), "path": path}
        print(split, manifest["splits"][split], flush=True)
    manifest["coverage"] = stats
    with open(os.path.join(OUT, "final_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
