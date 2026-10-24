import os
import subprocess
import sys
import traceback

D = os.environ.get("KATA_DIR", r"C:\gitProjects\titanium-ataxx\data\engines\kataataxx\ext\KataAtaxx_20260501_other_or_cpu")
CWD = os.environ.get("KATA_CWD", os.path.join(D, "engine_cpu"))
EXE = os.environ.get("KATA_EXE", os.path.join(CWD, "ataxx7x_cpu.exe"))
MODEL = os.environ.get("KATA_MODEL", os.path.join(D, "weights", "b12c384h6tfrs2-s3413299200-d1434585433.onnx"))
CFG = os.environ.get("KATA_CFG", os.path.join(D, "engine_cpu.cfg"))
LOG = os.environ.get("KATA_LOG", os.path.join(r"C:\gitProjects\titanium-ataxx", "logs", "kata_bridge_dbg.log"))


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def parse_fen(fen):
    parts = fen.strip().split()
    board = parts[0]
    side = parts[1] if len(parts) > 1 else 'b'
    rows = board.split('/')
    cells = {}
    for r, row in enumerate(rows):
        x = 0
        for ch in row:
            if ch.isdigit():
                x += int(ch)
            else:
                cells[(x, 7 - r)] = ch
                x += 1
    return cells, side


def vertex_to_xy(v):
    v = (v or '').strip().upper()
    if len(v) < 2 or not v[0].isalpha() or not v[1].isdigit():
        return None
    x = ord(v[0]) - ord('A')
    y = int(v[1:])
    if not (0 <= x < 7 and 1 <= y <= 7):
        return None
    return x, y


def is_black(side):
    return (side or '').lower() in ('x', 'b')


def fallback_move(cells, side):
    own = 'x' if is_black(side) else 'o'
    empty = [k for k, v in cells.items() if v == '.']
    stones = [k for k, v in cells.items() if v == own]
    for (sx, sy) in stones:
        for (ex, ey) in empty:
            if max(abs(sx - ex), abs(sy - ey)) == 1:
                return "%s%d" % ("abcdefg"[ex], ey)
    for (sx, sy) in stones:
        for (ex, ey) in empty:
            if max(abs(sx - ex), abs(sy - ey)) == 2:
                return "%s%d%s%d" % ("abcdefg"[sx], sy, "abcdefg"[ex], ey)
    return None


class Kata:
    def __init__(self):
        self.p = subprocess.Popen([EXE, "gtp", "-model", MODEL, "-config", CFG],
                                  cwd=CWD,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  encoding="utf-8", errors="replace")
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if "GTP ready" in line:
                break

    def cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l:
                return out
            out.append(l.rstrip())
            if l.startswith('=') or l.startswith('?'):
                return out

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


def main():
    k = Kata()
    k.cmd("boardsize 7")
    k.cmd("clear_board")
    cells, side = {}, 'b'
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            if line == 'uai':
                print("id name KataAtaxx", flush=True)
                print("uaiok", flush=True)
            elif line == 'isready':
                print("readyok", flush=True)
            elif line == 'uainewgame':
                k.cmd("clear_board")
                cells, side = {}, 'b'
            elif line.startswith('position fen '):
                fen = line[len('position fen '):].strip()
                cells, side = parse_fen(fen)
                out = k.cmd("setfen " + fen)
                if out and out[-1].startswith('?'):
                    log("setfen failed: " + " | ".join(out) + " fen=" + fen)
            elif line.startswith('go'):
                toks = line.split()
                nodes = None
                movetime = None
                for i, t in enumerate(toks):
                    if t == 'nodes' and i + 1 < len(toks):
                        nodes = int(toks[i + 1])
                    if t == 'movetime' and i + 1 < len(toks):
                        movetime = int(toks[i + 1])
                if movetime is not None:
                    k.cmd("kata-set-param maxTime %.3f" % (movetime / 1000.0))
                else:
                    k.cmd("kata-set-param maxVisits %d" % (nodes if nodes else 500))
                color = 'b' if is_black(side) else 'w'
                out = k.cmd("genmove " + color)
                log("genmove " + color + " -> " + " | ".join(out))
                mv = ''
                for l in out:
                    if l.startswith('='):
                        mv = l[1:].strip()
                xy = vertex_to_xy(mv)
                if xy is None or mv.lower() in ('pass', 'resign'):
                    fb = fallback_move(cells, side) if cells else None
                    log("fallback (" + mv + "): " + str(fb))
                    if fb:
                        print("bestmove " + fb, flush=True)
                    else:
                        print("bestmove pass", flush=True)
                    continue
                fx, fy = xy
                ch = cells.get((fx, fy), '.')
                own = 'x' if is_black(side) else 'o'
                if ch == own:
                    out2 = k.cmd("genmove " + color)
                    log("genmove-stage2 " + color + " -> " + " | ".join(out2))
                    mv2 = ''
                    for l in out2:
                        if l.startswith('='):
                            mv2 = l[1:].strip()
                    xy2 = vertex_to_xy(mv2)
                    if xy2 is None:
                        fb = fallback_move(cells, side) if cells else None
                        log("fallback2 (" + mv2 + "): " + str(fb))
                        print("bestmove " + (fb if fb else "pass"), flush=True)
                    else:
                        tx, ty = xy2
                        print("bestmove " + "%s%d%s%d" % ("abcdefg"[fx], fy,
                                                          "abcdefg"[tx], ty), flush=True)
                else:
                    print("bestmove %s%d" % ("abcdefg"[fx], fy), flush=True)
        except Exception:
            log("EXCEPTION: " + traceback.format_exc())
            fb = fallback_move(cells, side) if cells else None
            print("bestmove " + (fb if fb else "pass"), flush=True)
    k.close()


if __name__ == '__main__':
    main()
