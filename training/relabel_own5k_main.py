import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = r"C:\gitProjects\titanium-ataxx"
BOARDS = os.path.join(ROOT, "data", "nnue", "lab12m")
OUT = os.path.join(ROOT, "data", "nnue", "lab12m_main5k")
ENGINE = r"C:\gitProjects\main-ataxx\target\release\titanium-cli.exe"
TUPLE = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def run_chunk(src, dst, start, count, nodes, engine, tup):
    with open(src, "r", encoding="ascii") as f:
        f.seek(0)
        limits = []
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + count:
                break
            limits.append(line)
    fd, inp = tempfile.mkstemp(prefix="b_", suffix=".txt", dir=OUT)
    fd2, tmp = tempfile.mkstemp(prefix="s_", suffix=".txt", dir=OUT)
    os.close(fd)
    os.close(fd2)
    try:
        with open(inp, "w", encoding="ascii") as f:
            f.writelines(limits)
        with open(inp, "rb") as fin, open(tmp, "wb") as fout:
            p = subprocess.run(
                [engine, "label", "--nodes", str(nodes), "--tuple", tup, "--tuple-native"],
                stdin=fin, stdout=fout, stderr=subprocess.PIPE, timeout=1800,
            )
        if p.returncode != 0:
            raise RuntimeError("engine exit %d: %s" % (
                p.returncode, p.stderr.decode("utf-8", "replace")[-500:]))
        got = count_lines(tmp)
        if got != len(limits):
            raise RuntimeError("got %d expected %d" % (got, len(limits)))
        os.replace(tmp, dst)
        return dst, got
    finally:
        for p in (inp, tmp):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=BOARDS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--tuple", default=TUPLE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eng = sha256_file(a.engine)
    net = sha256_file(a.tuple)
    print("engine", a.engine, eng[:16], flush=True)
    print("tuple", a.tuple, net[:16], flush=True)

    files = sorted(
        os.path.join(a.boards, n) for n in os.listdir(a.boards)
        if n.startswith("boards_") and n.endswith(".txt")
    )
    jobs = []
    for path in files:
        n = count_lines(path)
        tag = os.path.splitext(os.path.basename(path))[0].split("_")[1]
        for lo in range(0, n, a.chunk):
            hi = min(n, lo + a.chunk)
            dst = os.path.join(a.out, "scores_%s_%03d.txt" % (tag, lo // a.chunk))
            jobs.append((path, dst, lo, hi - lo))
    if a.limit_chunks:
        jobs = jobs[:a.limit_chunks]
    todo = []
    for path, dst, lo, count in jobs:
        if os.path.exists(dst) and count_lines(dst) == count:
            continue
        todo.append((path, dst, lo, count))
    print("board_files=%d chunks=%d todo=%d workers=%d" % (
        len(files), len(jobs), len(todo), a.workers), flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {
            ex.submit(run_chunk, p, d, lo, c, a.nodes, a.engine, a.tuple): (p, d, lo, c)
            for p, d, lo, c in todo
        }
        for fut in as_completed(futs):
            p, d, lo, c = futs[fut]
            try:
                dst, got = fut.result()
            except Exception as e:
                print("ERROR %s[%d:%d] %s" % (os.path.basename(p), lo, lo + c, e), flush=True)
                continue
            done += got
            rate = done / max(1e-9, time.time() - t0)
            print("done=%d/%d file=%s row=%d rows/s=%.1f elapsed=%.0fs" % (
                done, sum(x[3] for x in todo), os.path.basename(dst), lo, rate,
                time.time() - t0), flush=True)

    manifest = {
        "source": a.boards,
        "old_labels": "lab12m/scores_*.txt (5k, Moon eval)",
        "new_labels": a.out,
        "engine": a.engine,
        "engine_sha256": eng,
        "tuple": a.tuple,
        "tuple_sha256": net,
        "nodes": a.nodes,
        "workers": a.workers,
        "chunk": a.chunk,
        "chunks_total": len(jobs),
        "chunks_present": sum(1 for p, d, lo, c in jobs if os.path.exists(d) and count_lines(d) == c),
        "elapsed_s": round(time.time() - t0, 1),
        "created_at": time.time(),
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
