import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
DEFAULT_SRC = os.path.join(
    ROOT, "data", "datasets", "KataGo_Ataxx_202604",
    "b12c384h6tfrs2-s3413299200-d1434585433", "tdata",
)
DEFAULT_OUT = os.path.join(ROOT, "data", "nnue", "katago_2026_b12c384_s3413299200.npz")
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "nnue", "katago_2026_b12c384_s3413299200_manifest.json")
BIT_WEIGHTS = (1 << np.arange(49, dtype=np.uint64))


def bitmaps(packed):
    bits = np.unpackbits(packed, axis=2)[:, :, :49]
    own = bits[:, 1].astype(np.uint64)
    opp = bits[:, 2].astype(np.uint64)
    return np.sum(own * BIT_WEIGHTS, axis=1), np.sum(opp * BIT_WEIGHTS, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--source-tar", default="b12c384h6tfrs2-s3413299200-d1434585433.tar")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--min-empty", type=int, default=1)
    ap.add_argument("--max-empty", type=int, default=49)
    ap.add_argument("--dedup", type=int, default=1)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*.npz")))
    if not files:
        raise SystemExit("no tdata/*.npz under %s" % a.src)
    print("source files", len(files), flush=True)

    oc0, oc1, outs = [], [], []
    src_rows = 0
    used_files, skipped_files = [], []
    t0 = time.time()
    for fi, path in enumerate(files):
        try:
            z = np.load(path)
            b = z["binaryInputNCHWPacked"]
            n = len(b)
            if a.max_rows:
                n = min(n, a.max_rows - src_rows)
            if n <= 0:
                break
            b = b[:n]
            own, opp = bitmaps(b)
            gt = z["globalTargetsNC"][:n]
            win = gt[:, 0].astype(np.float32)
            draw = gt[:, 2].astype(np.float32)
            none = gt[:, 3].astype(np.float32)
            out = win + 0.5 * draw
            occupied = own | opp
            empty = 49 - np.bitwise_count(occupied).astype(np.int16)
            keep = (
                (none < 0.5)
                & (out >= 0.0) & (out <= 1.0)
                & (empty >= a.min_empty) & (empty <= a.max_empty)
                & (own != 0) & (opp != 0)
            )
        except Exception as e:
            skipped_files.append({"file": os.path.basename(path), "error": str(e)})
            print("skip %s: %s" % (os.path.basename(path), e), flush=True)
            continue
        oc0.append(own[keep])
        oc1.append(opp[keep])
        outs.append(out[keep].astype(np.float32))
        used_files.append(os.path.basename(path))
        src_rows += n
        print("file %d/%d %s rows %d kept %d" % (fi + 1, len(files), os.path.basename(path), n, int(keep.sum())), flush=True)

    if not oc0:
        raise SystemExit("no rows kept")
    occ0 = np.concatenate(oc0)
    occ1 = np.concatenate(oc1)
    out = np.concatenate(outs)
    del oc0, oc1, outs
    rows_kept = len(occ0)

    if a.dedup:
        keys = np.stack([occ0, occ1], axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        first.sort()
        occ0, occ1, out = occ0[first], occ1[first], out[first]
        print("dedup %d -> %d" % (rows_kept, len(occ0)), flush=True)

    n = len(occ0)
    stm = np.zeros(n, dtype=np.uint8)
    blk = np.zeros(n, dtype=np.uint64)
    teacher = ((out * 2.0 - 1.0) * 1000.0).astype(np.float32)
    sc = (teacher / 2000.0).astype(np.float32)
    wsum = np.ones(n, dtype=np.float32)
    empty = (49 - np.bitwise_count(occ0 | occ1)).astype(np.uint8)
    phase = np.zeros(n, dtype=np.uint8)
    phase[(empty >= 8) & (empty < 15)] = 1
    phase[(empty >= 15) & (empty < 30)] = 2
    phase[empty >= 30] = 3

    np.savez_compressed(
        a.out,
        occ0=occ0, occ1=occ1, blk=blk, stm=stm,
        out=out, teacher=teacher, sc=sc, wsum=wsum,
        empty=empty, phase=phase,
    )
    counts = {str(i): int((phase == i).sum()) for i in range(4)}
    manifest = {
        "dataset": "KataGo_Ataxx_202604",
        "source": "https://www.modelscope.cn/datasets/sigmoid/KataGo_Ataxx_202604",
        "source_tar": a.source_tar,
        "source_tdata": os.path.join(a.src, "*.npz"),
        "source_files_used": used_files,
        "source_files_skipped": skipped_files,
        "source_files_total": len(files),
        "source_rows_read": int(src_rows),
        "rows_kept_before_dedup": int(rows_kept),
        "rows": int(n),
        "dedup": bool(a.dedup),
        "label": "KataGo valueTarget win/loss/draw for side to move; out=win+0.5*draw",
        "relative": "occ0=current side, occ1=opponent, stm=0",
        "teacher": "(out*2-1)*1000 cp",
        "phase_counts": counts,
        "out_mean": float(out.mean()),
        "out_std": float(out.std()),
        "empty_min": int(empty.min()), "empty_max": int(empty.max()),
        "out_file": a.out,
        "out_bytes": int(os.path.getsize(a.out)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(a.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
