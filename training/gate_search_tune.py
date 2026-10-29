"""Gate candidate search tuning parameters vs main baseline (D84/Gate Law v5).

Runs a 416-game pure 5,000-node SPRT gate across 16 shards x 26 games with paired openings.
Candidate: Tuned search (dis_lmr_num=1, dis_sn_num=48).
Baseline: Main default (dis_lmr_num=0, dis_sn_num=0).

Usage:
  python training/gate_search_tune.py --tag tweaksels_lmr1_sn48
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--games", type=int, default=26)
    ap.add_argument("--book0", type=int, default=3000)
    ap.add_argument("--par", type=int, default=10)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--cand-tune", default="dis_lmr_num=1,dis_sn_num=48")
    ap.add_argument("--base-tune", default="dis_lmr_num=0,dis_sn_num=0")
    a = ap.parse_args()

    g = f"{ROOT}/logs/gate_{a.tag}"
    os.makedirs(g, exist_ok=True)

    head = (
        f"GATING MATCH: Candidate ({a.cand_tune}) vs Baseline ({a.base_tune})\n"
        f"Exe: {EXE} ({sha(EXE)}) | {a.nodes} nodes (pure) | {a.shards} x {a.games} = {a.shards * a.games} games | book {a.book0}..\n"
    )
    print(head, flush=True)
    with open(f"{g}/HEADER.txt", "w") as f:
        f.write(head)

    def shard(k):
        f = f"{g}/p{k:02d}.txt"
        if os.path.exists(f + ".done"):
            return f
        cmd = [
            EXE, "match",
            "--games", str(a.games),
            "--nodes", str(a.nodes),
            "--time", "0",
            "--opp-nodes", str(a.nodes),
            "--tune", a.cand_tune,
            "--opp", f"{EXE} serve --tune {a.base_tune}",
            "--book", BOOK,
            "--book-start", str(a.book0 + k * a.games // 2),
            "--start-game", "1",
            "--out", f,
        ]
        r = subprocess.run(cmd, cwd=MAIN, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            open(f + ".done", "w").close()
        return f

    print(f"Launching {a.shards} shards across {a.par} workers...")
    with ThreadPoolExecutor(a.par) as ex:
        files = list(ex.map(shard, range(a.shards)))

    # Determinism tripwire: replay shard 0 exactly
    print("Running determinism tripwire (replaying shard 0)...")
    trip = f"{g}/trip.txt"
    subprocess.run([
        EXE, "match",
        "--games", str(a.games),
        "--nodes", str(a.nodes),
        "--time", "0",
        "--opp-nodes", str(a.nodes),
        "--tune", a.cand_tune,
        "--opp", f"{EXE} serve --tune {a.base_tune}",
        "--book", BOOK,
        "--book-start", str(a.book0),
        "--start-game", "1",
        "--out", trip,
    ], cwd=MAIN, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    games = lambda p: [l for l in open(p, errors="ignore").read().splitlines() if l.startswith("-- game")]
    tw = "BIT-IDENTICAL (%d games)" % len(games(trip)) if games(files[0]) == games(trip) and games(trip) else "MISMATCH -> RUN VOID"
    print("tripwire:", tw, flush=True)

    pooled = f"{g}/pooled.txt"
    with open(pooled, "w") as fo:
        for f in files:
            fo.write(open(f, errors="ignore").read())

    s = subprocess.run([sys.executable, f"{ROOT}/training/gate_summary.py", pooled], cwd=ROOT, capture_output=True, text=True)
    out = head + "\ntripwire: " + tw + "\n" + s.stdout + s.stderr
    with open(f"{g}/SUMMARY.txt", "w") as f:
        f.write(out)
    print("\n" + out)


if __name__ == "__main__":
    main()
