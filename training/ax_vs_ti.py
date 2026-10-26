"""Arena: Titanium (alpha-beta, S57 net) vs the full Ataxx_CPP engine (net + its own MCTS).

Paired book openings (each line played from both seats), Botzone end rules (the
game ends the moment EITHER side cannot move; empties go to the mobile side) --
the rules Ataxx_CPP was built for. No passes.

Modes (per side):
  Titanium : --ti-ms 100            -> "go movetime 100"
             --ti-nodes N           -> "go nodes N"
  Ataxx_CPP: --cpp-ms 100           -> "go time 0.100" (persistent tree, like in real play)
             --cpp-evals N          -> fresh tree each move ("set" + "go evals N"); N=1 means one
                                       net evaluation at the root and nothing else.
Log: one result line per game plus a summary with paired-bootstrap CI.
"""
import argparse
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ataxx_move import RING, apply_move, parse_board  # noqa: E402
import ataxx_cpp_match as _M  # noqa: E402
from ataxx_cpp_match import Cpp, Reader, to_fen  # noqa: E402

# engine-wdl.exe = original engine + set/wdl/stat commands (go/raw output byte-identical); needed for "set".
_M.ENGINE = os.environ.get("CPP_EXE") or os.path.join(_M.ROOT, "data", "engines", "ataxx_cpp", "build", "engine-ctl.exe")  # = engine-wdl + policy/batchfull/valuegreedy

ROOT = r"C:\gitProjects\titanium-ataxx"
TI_EXE = r"C:\gitProjects\main-ataxx\target\release\titanium-cli.exe"
TI_TUP = os.path.join("data", "nnue", "final_v2_wdl_ep8.tup")
# 2026-10-26: Titanium = main's default big NNUE (e11827d1), tuple off (the gated config)
TI_NNUE = os.environ.get("TI_NNUE", "C:/gitProjects/main-ataxx/data/nnue/nnue_main.tnn")
FULL = (1 << 49) - 1

REACH2 = []
for s in range(49):
    r, f = divmod(s, 7)
    m = 0
    for rr in range(max(0, r - 2), min(7, r + 3)):
        for ff in range(max(0, f - 2), min(7, f + 3)):
            if (rr, ff) != (r, f):
                m |= 1 << (rr * 7 + ff)
    REACH2.append(m)


def can_move(own, occ):
    empty = ~occ & FULL
    x = own
    while x:
        b = x & -x
        if REACH2[b.bit_length() - 1] & empty:
            return True
        x ^= b
    return False


def flip_rows(b):
    """Titanium packed index (row-major from the top) -> Ataxx_CPP index (rank 1 first)."""
    out = 0
    while b:
        lsb = b & -b
        i = lsb.bit_length() - 1
        r, c = divmod(i, 7)
        out |= 1 << ((6 - r) * 7 + c)
        b ^= lsb
    return out


KATA_CFG = os.path.join(ROOT, "data", "engines", "kataataxx", "ext", "KataAtaxx_20260501_other_or_cpu", "engine_cpu_arena.cfg")  # 1 thread, no ponder


class Ti:
    """Generic UAI player: Titanium by default, or any UAI command (the Kata bridge)."""

    def __init__(self, go, cmd=None, env=None):
        self.go = go
        self.p = subprocess.Popen(cmd or [TI_EXE, "serve", "--tuple", "none", "--nnue", TI_NNUE], cwd=ROOT, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
                                  encoding="utf-8", errors="replace")
        self.r = Reader(self.p)
        self.send("uai")
        while "uaiok" not in self.r.line(30):
            pass
        self.send("isready")
        while "readyok" not in self.r.line(30):
            pass

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def move(self, state):
        self.send("position fen " + to_fen(state))
        self.send(self.go)
        while True:
            line = self.r.line(120)
            if line.startswith("bestmove "):
                t = line.split()[1]
                return None if t in ("0000", "pass") else t

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class CppSide(Cpp):
    def set_state(self, state):
        occ0, occ1, _, stm = state
        ply = 2 * (occ0 | occ1).bit_count()
        self.send("set %d %d %d %d" % (flip_rows(occ0), flip_rows(occ1), stm, ply))

    def go_cmd(self, cmd):
        self.send(cmd)
        line = self.r.line(60)
        v = [int(x) for x in line.split()[:4]]
        x0, y0, x1, y1 = v
        return "%s%d%s%d" % ("abcdefg"[x0], y0 + 1, "abcdefg"[x1], y1 + 1)


def result(state):
    occ0, occ1, _, _ = state
    c0, c1 = occ0.bit_count(), occ1.bit_count()
    m0, m1 = can_move(occ0, occ0 | occ1), can_move(occ1, occ0 | occ1)
    if m0 and not m1:
        c0 = 49 - c1
    elif m1 and not m0:
        c1 = 49 - c0
    return c0, c1


def play_game(book_line, ti_is_black, a):
    state = parse_board(book_line)
    if a.a.startswith("cmd:"):
        ti = Ti("go movetime %d" % a.ti_ms, cmd=[a.a[4:]])
    else:
        ti = Ti("go movetime %d" % a.ti_ms if a.ti_nodes is None else "go nodes %d" % a.ti_nodes)
    kata = cpp = None
    if a.opp == "kata":
        env = dict(os.environ, KATA_CFG=KATA_CFG, KATA_LOG=os.path.join(ROOT, "logs", "kata_arena_%d.log" % os.getpid()))
        kgo = ("go movetime %d" % a.kata_ms) if a.kata_ms else ("go nodes %d" % a.kata_visits)
        kata = Ti(kgo, cmd=[sys.executable, os.path.join(ROOT, "training", "kata_bridge.py")], env=env)
        kata.send("uainewgame")
    elif a.opp == "uai":
        kata = Ti(a.opp_go, cmd=a.opp_cmd.split("|"))      # generic UAI opponent (Moonbird); args split on |
        kata.send("uainewgame")
    else:
        cpp = CppSide()
        cpp.set_state(state)
    ti.send("uainewgame")
    log = []
    try:
        for _ in range(300):
            occ0, occ1, _, stm = state
            occ = occ0 | occ1
            if not (can_move(occ0, occ) and can_move(occ1, occ)):
                break
            ti_turn = (stm == 0) == ti_is_black
            if ti_turn:
                tok = ti.move(state)
                ns = apply_move(*state, tok) if tok else None
                if ns is None:
                    return None, "TI-ILLEGAL %r" % tok, log
                if cpp and a.cpp_evals is None and not a.cpp_policy and not a.cpp_valuegreedy:
                    cpp.play(tok, state)
            elif kata:
                tok = kata.move(state)
                ns = apply_move(*state, tok) if tok else None
                if ns is None:
                    return None, "KATA-ILLEGAL %r" % tok, log
            else:
                if a.cpp_valuegreedy:
                    cpp.set_state(state)
                    tok = cpp.go_cmd("valuegreedy")
                elif a.cpp_policy:
                    cpp.set_state(state)
                    tok = cpp.go_cmd("policy")
                elif a.cpp_evals is not None:
                    cpp.set_state(state)
                    tok = cpp.go_cmd("go evals %d" % a.cpp_evals)
                else:
                    tok = cpp.go_cmd("go time %.3f" % (a.cpp_ms / 1000.0))
                ns = apply_move(*state, tok)
                if ns is None:
                    return None, "CPP-ILLEGAL %r" % tok, log
            log.append(tok)
            state = ns
        c0, c1 = result(state)
        ti_c, cpp_c = (c0, c1) if ti_is_black else (c1, c0)
        return (1 if ti_c > cpp_c else -1 if ti_c < cpp_c else 0), "%d-%d" % (ti_c, cpp_c), log
    finally:
        ti.close()
        if cpp:
            cpp.close()
        if kata:
            kata.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--opp", default="cpp", choices=["cpp", "kata", "uai"])
    ap.add_argument("--kata-ms", type=int, default=None, help="KataAtaxx time per move (maxTime) instead of visits")
    ap.add_argument("--opp-cmd", default="", help="uai opponent command, arguments separated by |")
    ap.add_argument("--opp-go", default="go movetime 100", help="uai opponent go command")
    ap.add_argument("--kata-visits", type=int, default=16)
    ap.add_argument("--a", default="ab", help="side A: ab (Titanium alpha-beta) or cmd:<path to a UAI engine>")
    ap.add_argument("--ti-ms", type=int, default=100)
    ap.add_argument("--ti-nodes", type=int, default=None)
    ap.add_argument("--cpp-ms", type=int, default=100)
    ap.add_argument("--cpp-evals", type=int, default=None)
    ap.add_argument("--cpp-valuegreedy", action="store_true", help="value head only, 1 ply")
    ap.add_argument("--cpp-policy", action="store_true", help="one net forward pass, play highest-prior move")
    ap.add_argument("--book", default=os.path.join(ROOT, "data", "nnue", "book50k.txt"))
    ap.add_argument("--book-start", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    book = [l.strip() for l in open(a.book) if l.strip()]
    lines = a.games // 2
    tasks = [(a.book_start + i, black) for i in range(lines) for black in (True, False)]
    if a.opp == "kata":
        opp_desc = "KataAtaxx-2026 transformer, CPU 1 thread, %s" % (("%dms" % a.kata_ms) if a.kata_ms else ("%d visits" % a.kata_visits))
    elif a.opp == "uai":
        opp_desc = "UAI %s [%s]" % (os.path.basename(a.opp_cmd.split("|")[0]), a.opp_go)
    elif a.cpp_valuegreedy:
        opp_desc = "Ataxx_CPP VALUE-only 1-ply"
    elif a.cpp_policy:
        opp_desc = "Ataxx_CPP policy-only (1 forward pass)"
    else:
        opp_desc = "Ataxx_CPP %s" % (("%dms" % a.cpp_ms) if a.cpp_evals is None else ("%d evals (fresh tree)" % a.cpp_evals))
    desc = "%s %s vs %s" % ("Titanium-AB" if a.a == "ab" else "EXT[%s]" % os.path.basename(a.a[4:]),
                            ("%dms" % a.ti_ms) if a.ti_nodes is None else ("%d nodes" % a.ti_nodes), opp_desc)
    exe_sha = subprocess.run(["certutil", "-hashfile", TI_EXE, "SHA256"], capture_output=True, text=True).stdout.split("\n")[1][:8]
    out = open(a.out, "w", encoding="utf-8")
    lock = threading.Lock()

    def emit(s):
        with lock:
            out.write(s + "\n")
            out.flush()

    emit("ARENA %s | games %d | book %d..%d | ti exe %s nnue %s | cpp %s champion+4 phases | Botzone end rules" % (
        desc, len(tasks), a.book_start, a.book_start + lines - 1, exe_sha, os.path.basename(TI_NNUE), os.path.basename(_M.ENGINE)))
    scores = {}

    def run(t):
        idx, black = t
        t0 = time.time()
        try:
            res, info, log = play_game(book[idx], black, a)
        except Exception as e:  # noqa: BLE001
            res, info, log = None, "ERROR %r" % e, []
        with lock:
            if res is not None:
                scores.setdefault(idx, []).append((1.0 + res) / 2.0)
        emit("game book %d ti=%s: %s (%s) plies %d %.0fs" % (idx, "x" if black else "o",
             {1: "TITANIUM", -1: "ATAXX_CPP", 0: "DRAW", None: "VOID"}[res], info, len(log), time.time() - t0))

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run, tasks))
    pairs = [sum(v) / len(v) for v in scores.values() if len(v) == 2]
    games = [x for v in scores.values() for x in v]
    n = len(games)
    s = sum(games) / n if n else float("nan")
    random.seed(1)
    bs = sorted(sum(random.choices(pairs, k=len(pairs))) / len(pairs) for _ in range(4000)) if pairs else [0, 0]
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]

    def elo(x):
        x = min(max(x, 1e-4), 1 - 1e-4)
        return -400 * math.log10(1 / x - 1)
    wins = sum(1 for x in games if x == 1.0)
    losses = sum(1 for x in games if x == 0.0)
    emit("SUMMARY %s: games %d  Titanium W-L-D %d-%d-%d  score %.3f  elo %+.0f  CI[%.3f,%.3f] eloCI[%+.0f,%+.0f]" % (
        desc, n, wins, losses, n - wins - losses, s, elo(s), lo, hi, elo(lo), elo(hi)))
    print(open(a.out, encoding="utf-8").read().split("\n")[-2])


if __name__ == "__main__":
    main()
