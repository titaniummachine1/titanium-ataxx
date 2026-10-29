"""Gate candidate SmallNet vs baseline builtin SmallNet (Gate Law v5).

Usage:
  python training/gate_smallnet.py --cand-sn C:/gitProjects/titanium-ataxx/data/nnue/smallnet/fut_small_d4_t0.02.tsn --tag sn_d4_100ms --time-ms 100 --par 4
"""
import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = "C:/gitProjects/titanium-ataxx"
MAIN = "C:/gitProjects/main-ataxx"
EXE = MAIN + "/target/release/titanium-cli.exe"
BOOK = ROOT + "/data/nnue/book50k.txt"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-sn", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--games", type=int, default=26)
    ap.add_argument("--book0", type=int, default=3000)
    ap.add_argument("--par", type=int, default=4, help="workers (4 for time gates, 10 for node gates)")
    ap.add_argument("--nodes", type=int, default=0, help="fixed node limit (0 = use time)")
    ap.add_argument("--time-ms", type=int, default=100, help="movetime in ms")
    a = ap.parse_args()

    g = f"{ROOT}/logs/gate_{a.tag}"
    os.makedirs(g, exist_ok=True)

    limits_desc = f"{a.time_ms} ms/move" if a.time_ms > 0 else f"{a.nodes} nodes (pure)"
    head = (
        f"SMALLNET GATING MATCH: Candidate ({a.cand_sn}, {sha(a.cand_sn)}) vs Baseline Builtin SmallNet\n"
        f"Exe: {EXE} ({sha(EXE)}) | Limits: {limits_desc} | {a.shards} x {a.games} = {a.shards * a.games} games | book {a.book0}..\n"
    )
    print(head, flush=True)
    with open(f"{g}/HEADER.txt", "w") as f:
        f.write(head)

    def shard(k):
        f = f"{g}/p{k:02d}.txt"
        if os.path.exists(f + ".done"):
            return f
        cmd = [
            EXE,
            "--sn", a.cand_sn,
            "match",
            "--games", str(a.games),
            "--opp", f"{EXE} serve",
            "--book", BOOK,
            "--book-start", str(a.book0 + k * a.games // 2),
            "--start-game", "1",
            "--out", f,
        ]
        if a.time_ms > 0:
            cmd += ["--time", str(a.time_ms), "--opp-time", str(a.time_ms)]
        else:
            cmd += ["--nodes", str(a.nodes), "--time", "0", "--opp-nodes", str(a.nodes)]

        r = subprocess.run(cmd, cwd=MAIN, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            open(f + ".done", "w").close()
        return f

    print(f"Launching {a.shards} shards across {a.par} workers...")
    with ThreadPoolExecutor(a.par) as ex:
        files = list(ex.map(shard, range(a.shards)))

    # Determinism tripwire: replay shard 0
    print("Running determinism tripwire (replaying shard 0)...")
    trip = f"{g}/trip.txt"
    cmd_trip = [
        EXE,
        "--sn", a.cand_sn,
        "match",
        "--games", str(a.games),
        "--opp", f"{EXE} serve",
        "--book", BOOK,
        "--book-start", str(a.book0),
        "--start-game", "1",
        "--out", trip,
    ]
    if a.time_ms > 0:
        cmd_trip += ["--time", str(a.time_ms), "--opp-time", str(a.time_ms)]
    else:
        cmd_trip += ["--nodes", str(a.nodes), "--time", "0", "--opp-nodes", str(a.nodes)]

    subprocess.run(cmd_trip, cwd=MAIN, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    games_fn = lambda p: [l for l in open(p, errors="ignore").read().splitlines() if l.startswith("-- game")]
    if os.path.exists(trip) and os.path.exists(files[0]):
        g_trip = games_fn(trip)
        g_file0 = games_fn(files[0])
        tw = (
            f"BIT-IDENTICAL ({len(g_trip)} games)"
            if g_trip == g_file0 and g_trip
            else f"REPLAYED ({len(g_trip)} games)"
        )
    else:
        tw = "TRIPWIRE MISSING"
    print(f"tripwire: {tw}", flush=True)

    pooled = f"{g}/pooled.txt"
    with open(pooled, "w") as fo:
        for f in files:
            if os.path.exists(f):
                fo.write(open(f, errors="ignore").read())

    res = subprocess.run(
        [sys.executable, f"{ROOT}/training/gate_summary.py", pooled],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = head + "\ntripwire: " + tw + "\n" + res.stdout + res.stderr
    with open(f"{g}/SUMMARY.txt", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
