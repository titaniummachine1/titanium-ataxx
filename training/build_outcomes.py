import glob
import json
import os
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
NNUE = os.path.join(ROOT, "data", "nnue")
CACHE = os.path.join(NNUE, "cache12M.npz")
OUT = os.path.join(NNUE, "position_outcomes.npz")
MANIFEST = os.path.join(NNUE, "position_outcomes_manifest.json")
W = (1 << np.arange(49, dtype=np.uint64))


def raw_dirs():
    return sorted(glob.glob(os.path.join(
        ROOT, "data", "datasets", "KataGo_Ataxx_202604", "*", "tdata")))


def add_chunk(k0, k1, k2, win, draw, store):
    if len(k0) == 0:
        return
    store["o0"].append(np.asarray(k0, dtype=np.uint64))
    store["o1"].append(np.asarray(k1, dtype=np.uint64))
    store["stm"].append(np.asarray(k2, dtype=np.uint8))
    store["win"].append(np.asarray(win, dtype=np.float32))
    store["draw"].append(np.asarray(draw, dtype=np.float32))


def aggregate(store):
    o0 = np.concatenate(store["o0"])
    o1 = np.concatenate(store["o1"])
    stm = np.concatenate(store["stm"])
    win = np.concatenate(store["win"])
    draw = np.concatenate(store["draw"])
    order = np.lexsort((stm, o1, o0))
    o0, o1, stm = o0[order], o1[order], stm[order]
    win, draw = win[order], draw[order]
    change = np.empty(len(o0), dtype=bool)
    change[0] = True
    change[1:] = (o0[1:] != o0[:-1]) | (o1[1:] != o1[:-1]) | (stm[1:] != stm[:-1])
    start = np.flatnonzero(change)
    n = np.diff(np.r_[start, len(o0)]).astype(np.uint32)
    wins = np.add.reduceat(win, start).astype(np.float32)
    draws = np.add.reduceat(draw, start).astype(np.float32)
    losses = (n.astype(np.float32) - wins - draws).astype(np.float32)
    return {
        "occ0": o0[start], "occ1": o1[start], "stm": stm[start],
        "wins": wins, "losses": losses, "draws": draws, "n": n,
    }


def main():
    t0 = time.time()
    store = {k: [] for k in ("o0", "o1", "stm", "win", "draw")}
    files = 0
    rows = 0
    for d in raw_dirs():
        paths = sorted(glob.glob(os.path.join(d, "*.npz")))
        print("dir", d, "files", len(paths), flush=True)
        for path in paths:
            try:
                z = np.load(path)
                b = z["binaryInputNCHWPacked"]
                bits = np.unpackbits(b, axis=2)[:, :, :49]
                o0 = np.sum(bits[:, 1].astype(np.uint64) * W, axis=1)
                o1 = np.sum(bits[:, 2].astype(np.uint64) * W, axis=1)
                gt = z["globalTargetsNC"]
                win = gt[:, 0].astype(np.float32)
                draw = gt[:, 2].astype(np.float32)
                add_chunk(o0, o1, np.zeros(len(o0), np.uint8), win, draw, store)
                rows += len(o0)
                files += 1
                if files % 50 == 0:
                    print("files", files, "rows", rows, "%.0fs" % (time.time() - t0), flush=True)
            except Exception as e:
                print("SKIP", os.path.basename(path), e, flush=True)
    z = np.load(CACHE)
    out = z["out"].astype(np.float32)
    win = (out > 0.75).astype(np.float32)
    draw = (np.abs(out - 0.5) <= 0.25).astype(np.float32)
    add_chunk(z["occ0"], z["occ1"], z["stm"], win, draw, store)
    rows += len(out)
    print("raw rows", rows, "files", files, flush=True)
    agg = aggregate(store)
    np.savez_compressed(OUT, **agg)
    n = len(agg["n"])
    manifest = {
        "raw_rows": int(rows), "files": files, "unique_positions": int(n),
        "multiplicity": {
            "1": int((agg["n"] == 1).sum()),
            "2": int((agg["n"] == 2).sum()),
            "3+": int((agg["n"] >= 3).sum()),
            "max": int(agg["n"].max()),
        },
        "wins_sum": float(agg["wins"].sum()),
        "losses_sum": float(agg["losses"].sum()),
        "draws_sum": float(agg["draws"].sum()),
        "out_file": OUT, "out_bytes": int(os.path.getsize(OUT)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
