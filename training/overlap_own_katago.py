import argparse
import json
import os
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
OWN = os.path.join(ROOT, "data", "nnue", "cache12M_clean.npz")
KATAGO = [
    os.path.join(ROOT, "data", "nnue", "katago_2026_b12c384_s3413299200.npz"),
    os.path.join(ROOT, "data", "nnue", "katago_2026_b12c384_s1835540480.npz"),
    os.path.join(ROOT, "data", "nnue", "katago_2026_b12c384_s1914773504.npz"),
]
OUT = os.path.join(ROOT, "data", "nnue", "overlap_own_katago.npz")
MANIFEST = os.path.join(ROOT, "data", "nnue", "overlap_own_katago.json")


def encode(o0, o1, b, s):
    return ((int(o0) << 147) | (int(o1) << 98) | (int(b) << 49) | int(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default=OWN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.own, mmap_mode="r")
    n_own = len(z["occ0"])
    own_lookup = {}
    for i in range(n_own):
        k = encode(z["occ0"][i], z["occ1"][i], z["blk"][i], z["stm"][i])
        if k not in own_lookup:
            own_lookup[k] = i
        if (i + 1) % 2000000 == 0:
            print("own indexed %d/%d (%.0fs)" % (i + 1, n_own, time.time() - t0), flush=True)
    print("own unique keys", len(own_lookup), flush=True)

    samples = []
    per_file = []
    total_kg = 0
    overlaps = 0
    for path in KATAGO:
        if not os.path.exists(path):
            continue
        kz = np.load(path, mmap_mode="r")
        n = len(kz["occ0"])
        local = 0
        for i in range(n):
            k = encode(kz["occ0"][i], kz["occ1"][i], kz["blk"][i], kz["stm"][i])
            oi = own_lookup.get(k)
            if oi is not None:
                local += 1
                if len(samples) < 200000:
                    samples.append((k, oi, i, float(kz["out"][i])))
            if (i + 1) % 2000000 == 0:
                print("%s %d/%d overlap=%d (%.0fs)" % (
                    os.path.basename(path), i + 1, n, local, time.time() - t0), flush=True)
        total_kg += n
        overlaps += local
        per_file.append({"file": os.path.basename(path), "rows": int(n), "overlap": int(local)})
        print("DONE %s rows=%d overlap=%d" % (os.path.basename(path), n, local), flush=True)

    if samples:
        ks = np.asarray([s[0] for s in samples], dtype=object)
        oi = np.asarray([s[1] for s in samples], dtype=np.int64)
        ki = np.asarray([s[2] for s in samples], dtype=np.int64)
        kg_out = np.asarray([s[3] for s in samples], dtype=np.float32)
        own_teacher = z["teacher"][oi].astype(np.float32)
        own_out = z["out"][oi].astype(np.float32)
        np.savez_compressed(
            a.out, own_index=oi, katago_index=ki, own_teacher=own_teacher,
            own_out=own_out, katago_out=kg_out,
            own_occ0=z["occ0"][oi], own_occ1=z["occ1"][oi],
            own_blk=z["blk"][oi], own_stm=z["stm"][oi],
        )
    manifest = {
        "own_file": a.own,
        "own_rows": int(n_own),
        "own_unique": len(own_lookup),
        "katago_rows": int(total_kg),
        "overlap_rows": int(overlaps),
        "overlap_percent": round(100.0 * overlaps / max(1, total_kg), 4),
        "per_file": per_file,
        "samples_file": a.out if samples else None,
        "samples": len(samples),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Exact bitboard key matching. KataGo label is valueTarget outcome; own label is cache12M teacher.",
    }
    with open(a.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
