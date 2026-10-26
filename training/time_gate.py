"""Time-control gate: candidate exe vs main's exe, SAME net both sides (main's nnue_main.tnn, --tuple none), fixed ms per
move both sides, paired book lines from BOTH seats, `uainewgame` every game (match does it). Speed-affected changes need
the 100 ms + 50 ms legs (law 2). No tripwire (time limits are not deterministic); keep the box otherwise idle (D102):
--par 4 on this 4-core box. Verdict by training/gate_summary.py; nothing is merged (user's call).
Usage: python training/time_gate.py --cand-exe <exe> --tag <t> --ms 100
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
NET = MAIN + "/data/nnue/nnue_main.tnn"
BOOK = ROOT + "/data/nnue/book50k.txt"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-exe", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ms", type=int, required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--games", type=int, default=52, help="per shard (even: every book line from both seats)")
    ap.add_argument("--book0", type=int, default=3000)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--champ-exe", default=EXE, help="champion binary (default: main's exe)")
    a = ap.parse_args()
    cexe = os.path.abspath(a.cand_exe)
    champ = os.path.abspath(a.champ_exe)
    g = "%s/logs/gate_%s" % (ROOT, a.tag)
    os.makedirs(g, exist_ok=True)
    ref = subprocess.run(["git", "-C", MAIN, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", MAIN, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    head = "cand exe %s, champ exe %s (main is %s%s), net %s both sides | %d ms/move | %d x %d games, book %d.., par %d" % (
        sha(cexe), sha(champ), ref, " DIRTY" if dirty else "", sha(NET), a.ms, a.shards, a.games, a.book0, a.par)
    print(head, flush=True)
    open(g + "/HEADER.txt", "w").write(head + "\n")
    if dirty:
        sys.exit("main worktree is dirty -> law 8, not gating")

    def shard(k):
        f = "%s/p%02d.txt" % (g, k)
        if os.path.exists(f + ".done"):
            return f
        cmd = [cexe, "match", "--games", str(a.games), "--time", str(a.ms), "--opp-time", str(a.ms), "--tuple", "none",
               "--nnue", NET, "--opp", champ + " serve", "--opp-tuple", "none", "--opp-nnue", NET, "--book", BOOK,
               "--book-start", str(a.book0 + k * a.games // 2), "--start-game", "1", "--out", f]
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            open(f + ".done", "w").close()
        return f

    with ThreadPoolExecutor(a.par) as ex:
        files = list(ex.map(shard, range(a.shards)))
    with open(g + "/pooled.txt", "w") as out:
        for f in files:
            out.write(open(f).read())
    subprocess.run([sys.executable, ROOT + "/training/gate_summary.py", g + "/pooled.txt"])


if __name__ == "__main__":
    main()
