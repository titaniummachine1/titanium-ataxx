"""UAI bridge for Ataxx_CPP (data/engines/ataxx_cpp, the #1 C++ bot on the Botzone ladder) so `titanium-cli match --opp`
can play it. Ataxx_CPP's own protocol (src/engine.cpp): `set b0 b1 side ply` (49-bit stone sets, square = y*7+x in
ITS coordinates), `go evals N` (N network evaluations = its node budget) or `go time S` (CPU seconds), answer
`x0 y0 x1 y1 evals sims q secs` or `pass`, `legal` lists legal moves.
UAI side: `position fen F` (Titanium's to_ataxx_fen: rows top=rank 7 first, x = black to move first), `go nodes N` ->
go evals N, `go movetime MS` -> go time MS/1000.
Coordinates: FEN cell (file f = 0..6 from a, rank 1..7) <-> engine (x, y) with x = f and y = Y_OF_RANK(rank); side
0 = black ('x'). Both verified by `python training/cpp_bridge.py --verify` (engine `legal` == FEN legal moves).
"""
import os
import subprocess
import sys

ROOT = r"C:\gitProjects\titanium-ataxx"
WDIR = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "weights")
ENGINE = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "build", "engine.exe")
PHASES = ["p0-8.bin:0:9", "p8-15.bin:9:16", "p15-30.bin:16:31", "p30-50.bin:31:50"]
FLIP = os.environ.get("CPP_FLIP", "1") == "1"      # engine y = 7 - rank (top row = 0) when True, rank - 1 otherwise
BLACK_SIDE = int(os.environ.get("CPP_BLACK_SIDE", "0"))


def y_of_rank(rank):
    return 7 - rank if FLIP else rank - 1


def rank_of_y(y):
    return 7 - y if FLIP else y + 1


def parse_fen(fen):
    parts = fen.strip().split()
    cells = {}
    for r, row in enumerate(parts[0].split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                cells[(f, 7 - r)] = ch
                f += 1
    side = parts[1] if len(parts) > 1 else "x"
    return cells, side


def to_set(cells, side):
    b = [0, 0]
    for (f, rank), ch in cells.items():
        if ch in "xo":
            s = y_of_rank(rank) * 7 + f
            b[0 if ch == "x" else 1] |= 1 << s
    stones = bin(b[0]).count("1") + bin(b[1]).count("1")
    ply = max(0, stones - 4)                      # FEN carries no move counter; only the judge's 400-ply cap uses it
    eside = BLACK_SIDE if side == "x" else 1 - BLACK_SIDE
    # Ataxx_CPP's two stone sets are indexed by its side numbering
    e = [b[0], b[1]] if BLACK_SIDE == 0 else [b[1], b[0]]
    return "set %d %d %d %d" % (e[0], e[1], eside, ply)


def uci_of(x0, y0, x1, y1):
    a = "abcdefg"[x0] + str(rank_of_y(y0))
    t = "abcdefg"[x1] + str(rank_of_y(y1))
    return t if max(abs(x1 - x0), abs(y1 - y0)) == 1 else a + t


class Engine:
    def __init__(self):
        self.p = subprocess.Popen([ENGINE, "champion.bin", "cmd", "0", "0"] + PHASES, cwd=WDIR, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def line(self):
        return self.p.stdout.readline().strip()


def fen_legal(cells, side):
    own = side
    occ = {k for k, v in cells.items() if v in "xo#"}
    out = set()
    for (f, r), ch in cells.items():
        if ch != own:
            continue
        for df in range(-2, 3):
            for dr in range(-2, 3):
                g, s = f + df, r + dr
                if (df or dr) and 0 <= g < 7 and 1 <= s <= 7 and (g, s) not in occ:
                    out.add(((f, r), (g, s)))
    return out


def verify():
    """Engine `legal` must equal the FEN's legal moves (as cell pairs) on asymmetric positions, for both sides."""
    import random
    e = Engine()
    rnd = random.Random(3)
    bad = 0
    for t in range(40):
        cells = {}
        for f in range(7):
            for r in range(1, 8):
                cells[(f, r)] = rnd.choice("xxooo......")
        side = "x" if t % 2 == 0 else "o"
        e.send(to_set(cells, side))
        e.send("legal")
        toks = [int(v) for v in e.line().split()]
        got = set()
        for i in range(0, len(toks), 4):
            x0, y0, x1, y1 = toks[i:i + 4]
            got.add(((x0, rank_of_y(y0)), (x1, rank_of_y(y1))))
        want = fen_legal(cells, side)
        # clones: Ataxx_CPP lists one source per clone target; compare destinations of clones and full jumps
        def norm(ms):
            return {(m[1], None) if max(abs(m[0][0] - m[1][0]), abs(m[0][1] - m[1][1])) == 1 else m for m in ms}
        if norm(got) != norm(want):
            bad += 1
            if bad <= 2:
                print("MISMATCH pos %d side %s: engine %d moves, fen %d moves" % (t, side, len(norm(got)), len(norm(want))))
    e.send("quit")
    print("verify: %d/40 positions mismatched (FLIP=%s BLACK_SIDE=%d)" % (bad, FLIP, BLACK_SIDE))
    return bad == 0


def main():
    e = Engine()
    cells, side = {}, "x"
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "uai":
            print("id name Ataxx_CPP (bridge)\nuaiok", flush=True)
        elif line == "isready":
            print("readyok", flush=True)
        elif line == "uainewgame":
            e.send("new")
        elif line.startswith("position fen "):
            cells, side = parse_fen(line[len("position fen "):])
            e.send(to_set(cells, side))
        elif line.startswith("go"):
            toks = line.split()
            nodes = movetime = None
            for i, t in enumerate(toks):
                if t == "nodes" and i + 1 < len(toks):
                    nodes = int(toks[i + 1])
                if t == "movetime" and i + 1 < len(toks):
                    movetime = int(toks[i + 1])
            e.send("go evals %d" % nodes if nodes else "go time %.3f" % ((movetime or 1000) / 1000.0))
            ans = e.line()
            if ans.startswith("pass") or not ans:
                print("bestmove 0000", flush=True)
            else:
                x0, y0, x1, y1 = (int(v) for v in ans.split()[:4])
                print("bestmove " + uci_of(x0, y0, x1, y1), flush=True)
        elif line == "quit":
            e.send("quit")
            break


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    main()
