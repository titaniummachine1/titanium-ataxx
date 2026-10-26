"""Gate a trained NNUE checkpoint against the CHAMPION = main's own NNUE (c4829350), same main exe both sides (D84).

1. export the .pt to TNN and verify Rust == torch (`titanium-cli nneval` on the export's reference boards; no gate on a mismatch)
2. 5k nodes both sides, PURE node limit (`--time 0`: `match` otherwise caps the in-process candidate at --time 1000 ms
   too, which under heavy CPU load stops searches early = nondeterministic + handicapped, D102), (same architecture/width = same eval speed, so the fixed-node gate is the fair one, law 2),
   16 shards x 26 games = 416 games, paired book lines from BOTH seats, `uainewgame` every game (match does it),
   candidate = main exe --nnue <cand.tnn>, champion = main exe `serve` --opp-nnue nnue_main.tnn, 7 shards in parallel
3. determinism tripwire = shard 0 replayed (must replay bit-identically, else VOID), then training/gate_summary.py on the pooled log.
Nothing is merged or staged: the verdict is the user's (laws 3/4/10).
Usage: python training/gate_net_vs_main.py --pt data/nnue/modal_ckpt/base_v4s/net.pt --tag s1v4
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
CHAMP = MAIN + "/data/nnue/nnue_main.tnn"
BOOK = ROOT + "/data/nnue/book50k.txt"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--games", type=int, default=26, help="per shard (even: every book line from both seats)")
    ap.add_argument("--book0", type=int, default=3000)
    ap.add_argument("--par", type=int, default=10, help="pure-node gate: 10 = PC sweet spot (D111)")
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--champ", default=CHAMP, help="opponent net (default: main's nnue_main.tnn)")
    ap.add_argument("--cand-exe", default=EXE, help="candidate engine binary (default: main's; the champion side always runs main's exe)")
    ap.add_argument("--tnn", default="", help="use this .tnn for the candidate instead of exporting --pt")
    a = ap.parse_args()
    champ = a.champ
    cexe = a.cand_exe
    g = "%s/logs/gate_%s" % (ROOT, a.tag)
    os.makedirs(g, exist_ok=True)
    tnn = "%s/cand.tnn" % g
    if a.tnn:
        import shutil
        shutil.copy(a.tnn, tnn)
    else:
        subprocess.run([sys.executable, "training/export_nnue.py", "--pt", a.pt, "--out", tnn], cwd=ROOT, check=True)
        par = subprocess.run([EXE, "nneval", "--nnue", tnn, "--boards", tnn + ".ref.txt"], capture_output=True, text=True)
        print(par.stdout.strip()[-400:])
        if par.returncode != 0:
            sys.exit("PARITY FAILED -> not gating")
    ref = subprocess.run(["git", "-C", MAIN, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", MAIN, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    head = "cand exe %s, champ exe %s (main %s%s) cand net %s (%s) champ net %s | %d nodes | %d x %d games, book %d.." % (
        sha(cexe), sha(EXE), ref, " DIRTY" if dirty else "", sha(tnn), a.pt or a.tnn, sha(champ), a.nodes, a.shards, a.games, a.book0)
    print(head, flush=True)
    open(g + "/HEADER.txt", "w").write(head + "\n")
    if dirty:
        sys.exit("main worktree is dirty -> law 8, not gating")

    def shard(k):
        f = "%s/p%02d.txt" % (g, k)
        if os.path.exists(f + ".done"):
            return f
        cmd = [cexe, "match", "--games", str(a.games), "--nodes", str(a.nodes), "--time", "0", "--opp-nodes", str(a.nodes), "--tuple", "none",
               "--nnue", tnn, "--opp", EXE + " serve", "--opp-tuple", "none", "--opp-nnue", champ, "--book", BOOK,
               "--book-start", str(a.book0 + k * a.games // 2), "--start-game", "1", "--out", f]
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            open(f + ".done", "w").close()
        return f

    with ThreadPoolExecutor(a.par) as ex:
        files = list(ex.map(shard, range(a.shards)))
    trip = g + "/trip.txt"
    subprocess.run([cexe, "match", "--games", str(a.games), "--nodes", str(a.nodes), "--time", "0", "--opp-nodes", str(a.nodes), "--tuple", "none",
                    "--nnue", tnn, "--opp", EXE + " serve", "--opp-tuple", "none", "--opp-nnue", champ, "--book", BOOK,
                    "--book-start", str(a.book0), "--start-game", "1", "--out", trip], cwd=ROOT, stdout=subprocess.DEVNULL)
    games = lambda p: [l for l in open(p, errors="ignore").read().splitlines() if l.startswith("-- game")]
    tw = "BIT-IDENTICAL (%d games)" % len(games(trip)) if games(files[0]) == games(trip) and games(trip) else "MISMATCH -> RUN VOID"
    print("tripwire:", tw, flush=True)
    pooled = g + "/pooled.txt"
    with open(pooled, "w") as fo:
        for f in files:
            fo.write(open(f, errors="ignore").read())
    s = subprocess.run([sys.executable, "training/gate_summary.py", pooled], cwd=ROOT, capture_output=True, text=True)
    out = head + "\ntripwire: " + tw + "\n" + s.stdout + s.stderr
    open(g + "/SUMMARY.txt", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
