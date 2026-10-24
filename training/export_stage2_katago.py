import argparse
import json
import os
import time

import numpy as np
from libsql_client import dbapi2 as db

ROOT = r"C:\gitProjects\titanium-ataxx"
NNUE = os.path.join(ROOT, "data", "nnue")
DB = os.path.join(NNUE, "titanium_train.db")
RAW = [
    os.path.join(NNUE, "katago_2026_b12c384_s3413299200.npz"),
    os.path.join(NNUE, "katago_2026_b12c384_s1835540480.npz"),
    os.path.join(NNUE, "katago_2026_b12c384_s1914773504.npz"),
]
OUT = os.path.join(NNUE, "stage2_katago")
MANIFEST = os.path.join(NNUE, "stage2_katago_manifest.json")
FIELDS = ("occ0", "occ1", "blk", "stm", "out", "teacher", "sc",
          "wsum", "engine_cp", "phase", "empty")


def key(o0, o1, b, s):
    return ((int(o0) << 147) | (int(o1) << 98) | (int(b) << 49) | int(s))


def save_checked(path, arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    z = np.load(tmp, mmap_mode="r")
    for k, v in arrays.items():
        if len(z[k]) != len(v):
            raise RuntimeError("bad %s %s" % (path, k))
    del z
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)

    con = db.connect(a.db)
    con.execute("PRAGMA busy_timeout=120000")
    eng = {}
    for r in con.execute(
            "SELECT occ0,occ1,blk,stm,engine_cp FROM positions "
            "WHERE engine_cp IS NOT NULL"):
        eng[key(r[0], r[1], r[2], r[3])] = float(r[4])
    con.close()
    print("engine labels loaded", len(eng), flush=True)

    chunks = {s: {k: [] for k in FIELDS} for s in ("train", "val", "test")}
    matched = 0
    total = 0
    per_file = []
    for path in RAW:
        z = np.load(path)
        o0 = z["occ0"].astype(np.uint64)
        o1 = z["occ1"].astype(np.uint64)
        bb = z["blk"].astype(np.uint64)
        stm = z["stm"].astype(np.uint8)
        out = z["out"].astype(np.float32)
        n = len(o0)
        ecp = np.full(n, np.nan, dtype=np.float32)
        for i in range(n):
            v = eng.get(key(o0[i], o1[i], bb[i], stm[i]))
            if v is not None:
                ecp[i] = v
                matched += 1
        occp = np.clip((out * 2.0 - 1.0) * 1000.0, -1500.0, 1500.0).astype(np.float32)
        tcp = np.where(np.isfinite(ecp), ecp, occp).astype(np.float32)
        empty = (49 - np.bitwise_count(o0 | o1 | bb)).astype(np.uint8)
        phase = np.zeros(n, dtype=np.uint8)
        phase[empty >= 8] = 1
        phase[empty >= 15] = 2
        phase[empty >= 30] = 3
        sc = (tcp / 2000.0).astype(np.float32)
        wsum = np.ones(n, dtype=np.float32)
        for sp, m in (("val", (np.arange(n) % 50) == 0),
                      ("test", (np.arange(n) % 50) == 1),
                      ("train", (np.arange(n) % 50) >= 2)):
            d = chunks[sp]
            for name, arr in (("occ0", o0), ("occ1", o1), ("blk", bb),
                              ("stm", stm), ("out", out), ("teacher", tcp),
                              ("sc", sc), ("wsum", wsum), ("engine_cp", ecp),
                              ("phase", phase), ("empty", empty)):
                d[name].append(arr[m])
        total += n
        per_file.append({"file": os.path.basename(path), "rows": int(n),
                         "engine": int(np.isfinite(ecp).sum())})
        print("%s rows=%d engine=%d (%.0fs)" % (
            os.path.basename(path), n, int(np.isfinite(ecp).sum()),
            time.time() - t0), flush=True)
        del z, o0, o1, bb, stm, out, ecp, occp, tcp, empty, phase, sc, wsum
    result = {"raw_rows": int(total), "engine_matched": int(matched),
              "split": {}, "files": per_file}
    for sp, d in chunks.items():
        arrays = {k: np.concatenate(v) for k, v in d.items()}
        n = len(arrays["occ0"])
        path = os.path.join(a.out, "stage2_katago_%s.npz" % sp)
        save_checked(path, arrays)
        result["split"][sp] = {
            "rows": n, "bytes": os.path.getsize(path), "path": path,
            "outcome_mean": float(arrays["out"].mean()),
            "teacher_mean": float(arrays["teacher"].mean()),
            "engine_rows": int(np.isfinite(arrays["engine_cp"]).sum()),
        }
        print("wrote", sp, n, os.path.getsize(path), flush=True)
    result["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.manifest, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
