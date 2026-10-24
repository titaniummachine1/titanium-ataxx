import json
import os

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
CACHE = os.path.join(ROOT, "data", "nnue", "cache12M_clean.npz")
LAB = os.path.join(ROOT, "data", "nnue", "ataxx_label")
OUT = os.path.join(ROOT, "data", "nnue", "cache12M_ataxx_clean.npz")
MANIFEST = os.path.join(ROOT, "data", "nnue", "cache12M_ataxx_manifest.json")
CH = 4


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    z = np.load(CACHE, mmap_mode="r")
    n = len(z["occ0"])
    o0 = np.asarray(z["occ0"])
    o1 = np.asarray(z["occ1"])
    stm = np.asarray(z["stm"])
    blk = np.asarray(z["blk"])
    out = np.asarray(z["out"], dtype=np.float32)
    wsum = np.asarray(z["wsum"], dtype=np.float32)
    old = np.asarray(z["teacher"], dtype=np.float32)

    rows, qs, ss = [], [], []
    coverage = []
    for c in range(CH):
        lo, hi = c * n // CH, (c + 1) * n // CH
        path = os.path.join(LAB, "out_%d.txt" % c)
        m = count_lines(path)
        if m > hi - lo:
            raise SystemExit("chunk %d too many lines %d > %d" % (c, m, hi - lo))
        arr = np.loadtxt(path, dtype=np.float32, max_rows=m)
        rows.append(np.arange(lo, lo + m))
        qs.append(arr[:, 0])
        ss.append(arr[:, 1])
        coverage.append({"chunk": c, "have": m, "wanted": hi - lo, "missing_tail": hi - lo - m})
        print("chunk %d labels %d/%d" % (c, m, hi - lo), flush=True)
    idx = np.concatenate(rows)
    q = np.concatenate(qs)
    score = np.concatenate(ss)
    del rows, qs, ss

    o0 = o0[idx]
    o1 = o1[idx]
    stm = stm[idx]
    blk = blk[idx]
    out = out[idx]
    wsum = wsum[idx]
    old = old[idx]
    keep = (
        np.isfinite(q) & (np.abs(q) <= 1.0)
        & np.isfinite(score) & (np.abs(score) <= 1.0)
        & np.isfinite(out) & (out >= 0.0) & (out <= 1.0)
        & np.isfinite(wsum)
    )
    idx, q, score = idx[keep], q[keep], score[keep]
    o0, o1, stm, blk = o0[keep], o1[keep], stm[keep], blk[keep]
    out, wsum, old = out[keep], wsum[keep], old[keep]
    print("label filters kept %d/%d" % (len(idx), n), flush=True)
    q = q.astype(np.float32)
    score = score.astype(np.float32)
    empty = (49 - np.bitwise_count(o0 | o1)).astype(np.uint8)

    # Deduplicate exact positions. Keep the first occurrence; all rows are
    # from the same cache and labels are deterministic for a position.
    keys = np.stack([o0, o1, stm, blk], axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    idx = idx[first]
    o0, o1, stm, blk = o0[first], o1[first], stm[first], blk[first]
    out, wsum, old = out[first], wsum[first], old[first]
    q, score, empty = q[first], score[first], empty[first]
    print("dedup kept %d/%d" % (len(idx), len(keys)), flush=True)

    teacher = (q * 1000.0).astype(np.float32)
    sc = (np.clip(old, -1500.0, 1500.0) / 2000.0).astype(np.float32)
    phase = np.zeros(len(empty), np.uint8)
    phase[(empty >= 8) & (empty < 15)] = 1
    phase[(empty >= 15) & (empty < 30)] = 2
    phase[empty >= 30] = 3

    tmp = OUT + ".tmp.npz"
    np.savez_compressed(
        tmp,
        occ0=o0, occ1=o1, blk=blk, stm=stm,
        out=out, wsum=wsum, teacher=teacher, sc=sc,
        ataxx_q=q, ataxx_score=score, empty=empty, phase=phase,
    )
    check = np.load(tmp, mmap_mode="r")
    expected = {
        "occ0": o0, "occ1": o1, "blk": blk, "stm": stm, "out": out,
        "wsum": wsum, "teacher": teacher, "sc": sc, "ataxx_q": q,
        "ataxx_score": score, "empty": empty, "phase": phase,
    }
    for k, v in expected.items():
        if len(check[k]) != len(v):
            raise RuntimeError("ataxx output check failed: %s" % k)
    del check
    os.replace(tmp, OUT)
    counts = {str(i): int((phase == i).sum()) for i in range(4)}
    manifest = {
        "source_cache": CACHE,
        "source_cache_rows": int(n),
        "engine": os.path.join(ROOT, "data", "engines", "ataxx_cpp", "build", "engine.exe"),
        "engine_commit": "d412e90d549ec1fb10cd96f998dc1db2064e740c",
        "engine_weights": "champion.bin + p0-8/p8-15/p15-30/p30-50",
        "teacher": "Ataxx_CPP raw transformer, phase net selected by empty-square count",
        "teacher_ply": "2*(49-empty), clamped 0..400",
        "coverage": coverage,
        "rows_after_filters": int(len(out)),
        "dedup": "exact (occ0,occ1,stm,blk)",
        "phase_counts": counts,
        "q_mean": float(q.mean()), "q_std": float(q.std()),
        "score_mean": float(score.mean()), "score_std": float(score.std()),
        "out_mean": float(out.mean()),
        "old_teacher_corr_q": float(np.corrcoef(q, np.clip(old, -5000, 5000))[0, 1]),
        "out_corr_q": float(np.corrcoef(q, out)[0, 1]),
        "out_corr_score": float(np.corrcoef(score, out)[0, 1]),
        "out_file": OUT,
        "out_bytes": int(os.path.getsize(OUT)),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
