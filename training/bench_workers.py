import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = r"C:\gitProjects\titanium-ataxx"
BOARDS = os.path.join(ROOT, "data", "nnue", "lab12m", "boards_0.txt")
ENGINE = r"C:\gitProjects\titanium-main\target\release\titanium-cli.exe"
TUPLE = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")
CHUNKS = 24
CHUNK = 5000
NODES = 5000
OUT = os.path.join(ROOT, "data", "nnue", "lab12m_bench")


def run_one(lines, tag):
    fd, inp = tempfile.mkstemp(prefix="bin_", dir=OUT)
    fd2, outp = tempfile.mkstemp(prefix="bout_", dir=OUT)
    os.close(fd)
    os.close(fd2)
    try:
        with open(inp, "w", encoding="ascii") as f:
            f.writelines(lines)
        t0 = time.time()
        with open(inp, "rb") as fin, open(outp, "wb") as fout:
            p = subprocess.run(
                [ENGINE, "label", "--nodes", str(NODES), "--tuple", TUPLE, "--tuple-native"],
                stdin=fin, stdout=fout, stderr=subprocess.PIPE, timeout=1800,
            )
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", "replace")[-300:])
        with open(outp, "rb") as f:
            got = sum(1 for _ in f)
        if got != len(lines):
            raise RuntimeError("rows %d != %d" % (got, len(lines)))
        return tag, len(lines), time.time() - t0
    finally:
        for p in (inp, outp):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(BOARDS, "r", encoding="ascii") as f:
        all_lines = [f.readline() for _ in range(CHUNKS * CHUNK)]
    jobs = []
    for i in range(CHUNKS):
        lines = all_lines[i * CHUNK:(i + 1) * CHUNK]
        jobs.append((i, lines))
    results = {}
    for workers in (8, 12, 16, 20, 24):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, lines, i) for i, lines in jobs]
            for fut in futs:
                fut.result()
        elapsed = time.time() - t0
        rows = CHUNKS * CHUNK
        results[workers] = {
            "rows": rows,
            "elapsed_s": round(elapsed, 1),
            "rows_per_s": round(rows / elapsed, 1),
        }
        print(json.dumps(results[workers]), flush=True)
    best = max(results.items(), key=lambda kv: kv[1]["rows_per_s"])
    print("BEST", best[0], json.dumps(best[1]), flush=True)
    with open(os.path.join(OUT, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
