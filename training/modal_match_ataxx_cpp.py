"""400-game match gate on Modal: Titanium (main pinned release af7e376) vs Ataxx_CPP (Botzone #1)
Runs with 1.0s thinking time per move (matching Botzone standard).
Paired openings (both sides play every seed as Black and White).
"""
import math
import os
import random
import sys
import time

import modal

ROOT = "C:/gitProjects/titanium-ataxx"
ENGINE_SRC = os.environ.get("ENGINE_SRC", "C:/gitProjects/main-ataxx")
ATAXX_CPP_DIR = os.path.join(ROOT, "data", "engines", "ataxx_cpp")

IGNORE_TITANIUM = [
    "target", "logs", "web", "training", "docs", "downloads", "botzone",
    "**/*.pt", "**/*.npz", "node_modules", ".git",
    "data/nnue/dis_positions.bin", "data/nnue/positions.bin", "data/nnue/positions_s*.bin",
    "data/nnue/*.npy", "data/nnue/labels.npz", "data/nnue/out_*", "data/nnue/relabel*",
]
IGNORE_CPP = [
    "build", ".git", "ataxx", "bot", "search", "uv.lock", "pyproject.toml",
]

CARGO = "/root/.cargo/bin/cargo"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "g++", "make")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .add_local_dir(ENGINE_SRC, "/eng/titanium", copy=True, ignore=IGNORE_TITANIUM)
    .run_commands("cd /eng/titanium && %s build -p titanium-cli --release" % CARGO)
    .add_local_dir(ATAXX_CPP_DIR, "/eng/ataxx_cpp", copy=True, ignore=IGNORE_CPP)
    .run_commands("cd /eng/ataxx_cpp && make build/engine")
    .add_local_file(os.path.join(ROOT, "training", "ataxx_move.py"), "/work/ataxx_move.py")
)

app = modal.App("titanium-vs-ataxx-cpp", image=image)


def run_game_logic(args, use_gpu=False):
    import os
    import queue
    import subprocess
    import sys
    import threading
    import time
    sys.path.insert(0, "/work")
    from ataxx_move import RING, apply_move, name_of, parse_board, sq_of

    def bits(x):
        while x:
            b = x & -x
            yield b.bit_length() - 1
            x ^= b

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

    game_id, seed, titanium_black, move_time_ms = args

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

        def line(self, timeout=30):
            item = self.q.get(timeout=timeout)
            if item is None:
                raise EOFError("engine closed")
            return item.strip()

    # Launch Titanium
    tit_proc = subprocess.Popen(
        ["/eng/titanium/target/release/titanium-cli", "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )
    tit_r = Reader(tit_proc)
    tit_proc.stdin.write("uai\n"); tit_proc.stdin.flush()
    while "uaiok" not in tit_r.line(20): pass
    tit_proc.stdin.write("isready\n"); tit_proc.stdin.flush()
    while "readyok" not in tit_r.line(20): pass

    # Launch Ataxx_CPP (Botzone champion weights + phase nets)
    phases = ["p0-8.bin:0:9", "p8-15.bin:9:16", "p15-30.bin:16:31", "p30-50.bin:31:50"]
    cpp_proc = subprocess.Popen(
        ["/eng/ataxx_cpp/build/engine", "champion.bin", "cmd", "0", "0"] + phases,
        cwd="/eng/ataxx_cpp/weights",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )
    cpp_r = Reader(cpp_proc)
    cpp_proc.stdin.write("new\n"); cpp_proc.stdin.flush()

    state = start_board()
    tit_side = 0 if titanium_black else 1
    passes = 0
    log = []
    positions = []  # List of position records for dataset

    try:
        for ply in range(250):
            moves = legal_moves(state)
            if not moves:
                passes += 1
                if passes >= 2:
                    break
                state = (state[0], state[1], state[2], 1 - state[3])
                tit_proc.stdin.write("uainewgame\n"); tit_proc.stdin.flush()
                cpp_proc.stdin.write("new\n"); cpp_proc.stdin.flush()
                continue
            passes = 0
            side = state[3]
            fen = to_fen(state)
            tit_eval_cp = None
            cpp_q_eval = None

            if side == tit_side:
                # Titanium to move
                tit_proc.stdin.write("position fen " + fen + "\n")
                tit_proc.stdin.write(f"go movetime {move_time_ms}\n")
                tit_proc.stdin.flush()
                token = None
                while True:
                    l = tit_r.line(30)
                    if "score cp " in l:
                        try:
                            parts = l.split()
                            idx = parts.index("cp")
                            tit_eval_cp = int(parts[idx + 1])
                        except Exception:
                            pass
                    if l.startswith("bestmove "):
                        t = l.split()[1]
                        token = None if t in ("0000", "pass") else t
                        break
                if token is None:
                    token, ns = moves[0]
                else:
                    ns = apply_move(*state, token)
                    if ns is None:
                        token, ns = moves[0]

                # Tell CPP what Titanium played: "play x0 y0 x1 y1"
                if len(token) == 2:
                    to = sq_of(token)
                    own = state[0] if state[3] == 0 else state[1]
                    cand = [s for s in RING[to] if (own >> s) & 1]
                    frm = cand[0] if cand else 0
                    tr, tc = divmod(to, 7)
                    fr, fc = divmod(frm, 7)
                    cpp_proc.stdin.write(f"play {fr} {fc} {tr} {tc}\n")
                else:
                    frm = sq_of(token[:2])
                    to = sq_of(token[2:4])
                    fr, fc = divmod(frm, 7)
                    tr, tc = divmod(to, 7)
                    cpp_proc.stdin.write(f"play {fr} {fc} {tr} {tc}\n")
                cpp_proc.stdin.flush()
            else:
                # Ataxx_CPP to move: "go time <sec>"
                sec = max(0.1, move_time_ms / 1000.0)
                cpp_proc.stdin.write(f"go time {sec:.2f}\n")
                cpp_proc.stdin.flush()
                token = None
                while True:
                    l = cpp_r.line(30)
                    parts = l.strip().split()
                    if len(parts) >= 7:
                        try:
                            x0, y0, x1, y1 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                            cpp_q_eval = float(parts[6])
                            f_sq = x0 * 7 + y0
                            t_sq = x1 * 7 + y1
                            token = f"{name_of(f_sq)}{name_of(t_sq)}"
                            break
                        except Exception:
                            pass
                    elif l == "pass":
                        token = None
                        break
                ns = apply_move(*state, token) if token else None
                if ns is None:
                    token, ns = moves[0]

            log.append(token)
            positions.append({
                "ply": ply,
                "fen": fen,
                "side": "b" if side == 0 else "w",
                "move": token,
                "tit_cp": tit_eval_cp,
                "cpp_q": cpp_q_eval,
                "gpu": use_gpu,
            })
            state = ns

        occ0, occ1, _, _ = state
        b = occ0.bit_count()
        w = occ1.bit_count()

        if b == w:
            winner = "draw"
            score = 0.5
            outcome_val = 0.5
        elif (b > w) == titanium_black:
            winner = "titanium"
            score = 1.0
            outcome_val = 1.0 if titanium_black else 0.0
        else:
            winner = "ataxx_cpp"
            score = 0.0
            outcome_val = 0.0 if titanium_black else 1.0

        for p in positions:
            p["outcome"] = outcome_val

        return {
            "game_id": game_id,
            "seed": seed,
            "titanium_black": titanium_black,
            "gpu": use_gpu,
            "winner": winner,
            "score": score,
            "black_score": b,
            "white_score": w,
            "plies": len(log),
            "positions": positions,
        }
    finally:
        try: tit_proc.kill()
        except Exception: pass
        try: cpp_proc.kill()
        except Exception: pass


@app.function(cpu=2.0, memory=1024, timeout=1200)
def play_match_cpu(args):
    """Play a CPU-only game."""
    return run_game_logic(args, use_gpu=False)


@app.function(gpu="T4", cpu=2.0, memory=2048, timeout=1200)
def play_match_gpu(args):
    """Play a GPU-accelerated game with NVIDIA T4."""
    return run_game_logic(args, use_gpu=True)


@app.local_entrypoint()
def main(cpu_games: int = 200, gpu_games: int = 200, ms: int = 1000):
    total_games = cpu_games + gpu_games
    print(f"=== Starting {total_games}-game Modal Clash: Titanium vs Ataxx_CPP ({ms}ms/move) ===", flush=True)
    print(f"Set 1: {cpu_games} games CPU-only | Set 2: {gpu_games} games with NVIDIA T4 GPU", flush=True)
    t0 = time.time()

    # Generate paired jobs for CPU (each seed played as Black and White)
    cpu_jobs = []
    gid = 1
    for s in range(cpu_games // 2):
        cpu_jobs.append((gid, s, True, ms))
        gid += 1
        cpu_jobs.append((gid, s, False, ms))
        gid += 1

    # Generate paired jobs for GPU (each seed played as Black and White)
    gpu_jobs = []
    for s in range(gpu_games // 2):
        gpu_jobs.append((gid, s + 1000, True, ms))
        gid += 1
        gpu_jobs.append((gid, s + 1000, False, ms))
        gid += 1

    results_cpu = []
    results_gpu = []

    print(f"Dispatched {len(cpu_jobs)} CPU games and {len(gpu_jobs)} GPU games to Modal...", flush=True)

    if cpu_jobs:
        for res in play_match_cpu.map(cpu_jobs, order_outputs=False):
            results_cpu.append(res)
            done = len(results_cpu)
            if done % 20 == 0 or done == len(cpu_jobs):
                tw = sum(1 for r in results_cpu if r["winner"] == "titanium")
                cw = sum(1 for r in results_cpu if r["winner"] == "ataxx_cpp")
                dr = sum(1 for r in results_cpu if r["winner"] == "draw")
                wr = (tw + 0.5 * dr) / done * 100.0
                print(f"[CPU {done:3d}/{len(cpu_jobs)}] Titanium: +{tw} -{cw} ={dr} ({wr:.1f}%)", flush=True)

    if gpu_jobs:
        for res in play_match_gpu.map(gpu_jobs, order_outputs=False):
            results_gpu.append(res)
            done = len(results_gpu)
            if done % 20 == 0 or done == len(gpu_jobs):
                tw = sum(1 for r in results_gpu if r["winner"] == "titanium")
                cw = sum(1 for r in results_gpu if r["winner"] == "ataxx_cpp")
                dr = sum(1 for r in results_gpu if r["winner"] == "draw")
                wr = (tw + 0.5 * dr) / done * 100.0
                print(f"[GPU {done:3d}/{len(gpu_jobs)}] Titanium: +{tw} -{cw} ={dr} ({wr:.1f}%)", flush=True)

    all_results = results_cpu + results_gpu
    dt = time.time() - t0
    total = len(all_results)

    def stats(res_list, name):
        if not res_list: return f"{name}: no games played\n"
        w_tit = sum(1 for r in res_list if r["winner"] == "titanium")
        w_cpp = sum(1 for r in res_list if r["winner"] == "ataxx_cpp")
        d = sum(1 for r in res_list if r["winner"] == "draw")
        n = len(res_list)
        s = (w_tit + 0.5 * d) / n
        elo = -400.0 * math.log10(1.0 / s - 1.0) if 0 < s < 1 else (999.0 if s == 1 else -999.0)
        return f"{name:10s}: +{w_tit} -{w_cpp} ={d} / {n} ({s*100:.2f}%, {elo:+.1f} Elo)"

    report = (
        f"\n=======================================================\n"
        f"400-GAME MODAL CLASH: Titanium vs Ataxx_CPP ({ms}ms/move)\n"
        f"=======================================================\n"
        f"{stats(results_cpu, 'CPU Set')}\n"
        f"{stats(results_gpu, 'GPU Set')}\n"
        f"{stats(all_results, 'Overall')}\n"
        f"Duration  : {dt:.1f}s ({dt/max(1, total):.2f}s per game)\n"
        f"=======================================================\n"
    )
    print(report, flush=True)

    log_path = os.path.join(ROOT, "logs", f"clash_titanium_vs_ataxx_cpp_{total}g.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
        for r in all_results:
            gpu_tag = "GPU" if r.get("gpu") else "CPU"
            f.write(f"[{gpu_tag}] Game {r['game_id']} (seed {r['seed']}, tit_black={r['titanium_black']}): "
                    f"{r['winner']} ({r['black_score']}-{r['white_score']}, {r['plies']} plies)\n")
    print(f"Saved match log to: {log_path}", flush=True)

    # Save every single position with Titanium eval and Ataxx_CPP eval into dataset JSONL
    import json
    jsonl_path = os.path.join(ROOT, "data", "nnue", f"clash_positions_{total}g.jsonl")
    total_positions = 0
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in all_results:
            for p in r.get("positions", []):
                p["game_id"] = r["game_id"]
                p["winner"] = r["winner"]
                jf.write(json.dumps(p) + "\n")
                total_positions += 1
    print(f"Saved {total_positions:,} evaluated positions with dual-engine scores to: {jsonl_path}", flush=True)
