import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ataxx_move import RING, apply_move, sq_of

ROOT = r"C:\gitProjects\titanium-ataxx"
ENGINE_DIR = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "weights")
ENGINE = os.path.join(ROOT, "data", "engines", "ataxx_cpp", "build", "engine.exe")
TITANIUM = os.path.join(ROOT, "target", "release", "titanium-cli.exe")
TUP = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")
PHASES = ["p0-8.bin:0:9", "p8-15.bin:9:16", "p15-30.bin:16:31", "p30-50.bin:31:50"]
OPP = os.environ.get("OPP", "titanium").lower()
KATA_DIR = os.path.join(ROOT, "data", "engines", "kataataxx", "kata2025", "KataAtaxx_20250131")
KATA_CWD = os.path.join(KATA_DIR, "engine")
KATA_EXE = os.path.join(KATA_CWD, "ataxx7x.exe")
KATA_MODEL = os.path.join(KATA_DIR, "weights", "b18.bin.gz")
KATA_CFG = os.path.join(r"C:\Users\TERMIN~1\AppData\Local\Temp\opencode", "kata_b18_test.cfg")
MOON = os.path.join(ROOT, "scripts", "moonbird", "Moonbird-1.1.0-windows-amd64.exe")


def bits(x):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


class Reader:
    def __init__(self, proc):
        self.q = queue.Queue()
        self.proc = proc
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                self.q.put(line)
        finally:
            self.q.put(None)

    def line(self, timeout=20):
        item = self.q.get(timeout=timeout)
        if item is None:
            raise EOFError("engine closed")
        return item.strip()


class Uai:
    def __init__(self):
        self.p = subprocess.Popen(
            [TITANIUM, "serve", "--tuple", TUP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        self.r = Reader(self.p)
        self.send("uai")
        while "uaiok" not in self.r.line(20):
            pass
        self.send("isready")
        while "readyok" not in self.r.line(20):
            pass

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def newgame(self):
        self.send("uainewgame")

    def move(self, fen, ms=1000):
        self.send("position fen " + fen)
        self.send("go movetime %d" % ms)
        while True:
            line = self.r.line(20)
            if line.startswith("bestmove "):
                t = line.split()[1]
                return None if t in ("0000", "pass") else t

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class MoonUai:
    def __init__(self):
        self.p = subprocess.Popen(
            [MOON],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        self.r = Reader(self.p)
        self.send("uai")
        while "uaiok" not in self.r.line(20):
            pass
        self.send("isready")
        while "readyok" not in self.r.line(20):
            pass

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def newgame(self):
        self.send("uainewgame")

    def move(self, fen, ms=1000):
        self.send("position fen " + fen)
        self.send("go movetime %d" % ms)
        while True:
            line = self.r.line(20)
            if line.startswith("bestmove "):
                t = line.split()[1]
                return None if t in ("0000", "pass") else t

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class KataUai:
    def __init__(self):
        env = dict(os.environ)
        env.update({
            "KATA_CWD": KATA_CWD,
            "KATA_EXE": KATA_EXE,
            "KATA_MODEL": KATA_MODEL,
            "KATA_CFG": KATA_CFG,
            "KATA_LOG": os.path.join(ROOT, "logs", "kata_vs_cpp_bridge.log"),
        })
        self.p = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "training", "kata_bridge.py")],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
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

    def newgame(self):
        self.send("uainewgame")

    def move(self, fen, ms=1000):
        self.send("position fen " + fen)
        self.send("go nodes 50")
        while True:
            line = self.r.line(30)
            if line.startswith("bestmove "):
                t = line.split()[1]
                return None if t in ("0000", "pass") else t

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


class Cpp:
    def __init__(self):
        args = [ENGINE, "champion.bin", "cmd", "0", "0"] + PHASES
        self.p = subprocess.Popen(
            args,
            cwd=ENGINE_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        self.r = Reader(self.p)
        self.send("new")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def play(self, token, state):
        if len(token) == 2:
            to = sq_of(token)
            if to is None:
                return
            own = state[0] if state[3] == 0 else state[1]
            cand = [s for s in RING[to] if (own >> s) & 1]
            if not cand:
                return
            frm = cand[0]
            a, b = frm, to
        else:
            a = sq_of(token[:2])
            b = sq_of(token[2:4])
            if a is None or b is None:
                return
        af, ar = a % 7, a // 7
        bf, br = b % 7, b // 7
        self.send("play %d %d %d %d" % (af, 6 - ar, bf, 6 - br))

    def go(self, seconds=1.0):
        self.send("go time %.3f" % seconds)
        line = self.r.line(20)
        v = [int(x) for x in line.split()[:4]]
        x0, y0, x1, y1 = v
        return "%s%d%s%d" % ("abcdefg"[x0], y0 + 1, "abcdefg"[x1], y1 + 1)

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def start_board():
    occ0 = (1 << sq_of("a1")) | (1 << sq_of("g7"))
    occ1 = (1 << sq_of("a7")) | (1 << sq_of("g1"))
    return occ0, occ1, 0, 0


def to_fen(state):
    occ0, occ1, blk, stm = state
    rows = []
    for r in range(7):
        row = ""
        gap = 0
        for f in range(7):
            i = r * 7 + f
            bit = 1 << i
            if blk & bit:
                c = "#"
            elif occ0 & bit:
                c = "x"
            elif occ1 & bit:
                c = "o"
            else:
                c = "."
            if c == ".":
                gap += 1
            else:
                if gap:
                    row += str(gap)
                    gap = 0
                row += c
        if gap:
            row += str(gap)
        rows.append(row)
    return "/".join(rows) + (" x 0 1" if stm == 0 else " o 0 1")


def legal_moves(state):
    occ0, occ1, blk, stm = state
    own = occ0 if stm == 0 else occ1
    empty = ~(occ0 | occ1 | blk) & ((1 << 49) - 1)
    out = []
    for frm in bits(own):
        fr, fc = divmod(frm, 7)
        for to in bits(empty):
            tr, tc = divmod(to, 7)
            d = max(abs(fr - tr), abs(fc - tc))
            if d not in (1, 2):
                continue
            token = "%s%d%s%d" % ("abcdefg"[fc], 7 - fr, "abcdefg"[tc], 7 - tr)
            ns = apply_move(*state, token)
            if ns is not None:
                out.append((token, ns))
    return out


def play_game(seed, engine_black):
    cpp = Cpp()
    if OPP == "kata":
        uai = KataUai()
    elif OPP == "moon":
        uai = MoonUai()
    else:
        uai = Uai()
    state = start_board()
    cpp_side = 0 if engine_black else 1
    passes = 0
    log = []
    try:
        for ply in range(300):
            moves = legal_moves(state)
            if not moves:
                passes += 1
                if passes >= 2:
                    break
                state = (state[0], state[1], state[2], 1 - state[3])
                uai.newgame()
                cpp.send("new")
                continue
            passes = 0
            side = state[3]
            if side == cpp_side:
                token = cpp.go(1.0)
                ns = apply_move(*state, token)
                if ns is None:
                    token, ns = moves[0]
            else:
                token = uai.move(to_fen(state), 1000)
                if token is None:
                    token, ns = moves[0]
                else:
                    ns = apply_move(*state, token)
                    if ns is None:
                        token, ns = moves[0]
                if ns is not None:
                    cpp.play(token, state)
            log.append(token)
            state = ns
        occ0, occ1, _, _ = state
        b = occ0.bit_count()
        w = occ1.bit_count()
        if b == w:
            res = 0
        elif (b > w) == engine_black:
            res = 1
        else:
            res = -1
        return res, b, w, state[3], len(log), " ".join(log)
    finally:
        cpp.close()
        uai.close()


def main():
    tasks = [(i, i % 2 == 0) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(play_game, i, b) for i, b in tasks]
        for i, f in enumerate(futs):
            try:
                res, b, w, stm, plies, log = f.result()
                print("game %d: result %d black %d white %d plies %d" % (i + 1, res, b, w, plies))
            except Exception as e:
                import traceback
                print("game %d: ERROR %r" % (i + 1, e))
                traceback.print_exc()


if __name__ == "__main__":
    main()
