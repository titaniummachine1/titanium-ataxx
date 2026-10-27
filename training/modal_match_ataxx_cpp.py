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
    "**/*.pt", "**/*.npz", "node_modules", ".git", "data",
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


@app.function(cpu=2.0, memory=1024, timeout=1200)
def play_match_pair(args):
    """Play a paired game (game_id, seed, titanium_black, move_time_ms)."""
    import os
    import queue
    import subprocess
    import sys
    import threading
    import time
    sys.path.insert(0, "/work")
    from ataxx_move import RING, apply_move, legal_moves, name_of, parse_board, sq_of, start_board, to_fen

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

    # Launch Ataxx_CPP
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
                            # format: x0 y0 x1 y1 evals sims q cpu_time
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

        # Annotate positions with game outcome
        for p in positions:
            p["outcome"] = outcome_val

        return {
            "game_id": game_id,
            "seed": seed,
            "titanium_black": titanium_black,
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


@app.local_entrypoint()
def main(games: int = 400, ms: int = 1000):
    print(f"=== Starting {games}-game Modal Clash: Titanium vs Ataxx_CPP (1.0s/move) ===", flush=True)
    t0 = time.time()

    # Generate paired jobs: each seed played twice (Titanium as Black, then as White)
    num_pairs = games // 2
    jobs = []
    gid = 1
    for s in range(num_pairs):
        jobs.append((gid, s, True, ms))   # Titanium = Black
        gid += 1
        jobs.append((gid, s, False, ms))  # Titanium = White
        gid += 1

    print(f"Dispatched {len(jobs)} games to Modal in parallel...", flush=True)

    results = []
    tit_wins = 0
    cpp_wins = 0
    draws = 0

    for res in play_match_pair.map(jobs, order_outputs=False):
        results.append(res)
        w = res["winner"]
        if w == "titanium": tit_wins += 1
        elif w == "ataxx_cpp": cpp_wins += 1
        else: draws += 1

        total = len(results)
        if total % 10 == 0 or total == len(jobs):
            winrate = (tit_wins + 0.5 * draws) / total * 100.0
            print(f"[{total:3d}/{len(jobs)}] Titanium: +{tit_wins} -{cpp_wins} ={draws} ({winrate:.1f}%) | "
                  f"Last: {res['winner']} ({res['black_score']}-{res['white_score']} in {res['plies']} plies)", flush=True)

    dt = time.time() - t0
    total = len(results)
    score = (tit_wins + 0.5 * draws) / total
    elo_diff = -400.0 * math.log10(1.0 / score - 1.0) if 0 < score < 1 else (999.0 if score == 1 else -999.0)

    # 95% bootstrap confidence interval
    scores_by_pair = {}
    for r in results:
        scores_by_pair.setdefault(r["seed"], []).append(r["score"])
    pair_means = [sum(v) / len(v) for v in scores_by_pair.values()]
    random.seed(42)
    boot = sorted(sum(random.choices(pair_means, k=len(pair_means))) / len(pair_means) for _ in range(2000))
    lo_pct = boot[int(0.025 * len(boot))] * 100.0
    hi_pct = boot[int(0.975 * len(boot))] * 100.0

    report = (
        f"\n=======================================================\n"
        f"MATCH REPORT: Titanium vs Ataxx_CPP ({total} games, {ms}ms/move)\n"
        f"=======================================================\n"
        f"Titanium: +{tit_wins} -{cpp_wins} ={draws} / {total} games\n"
        f"Score:    {score*100.0:.2f}% (95% CI: [{lo_pct:.1f}%, {hi_pct:.1f}%])\n"
        f"Elo Diff: {elo_diff:+.1f} Elo\n"
        f"Duration: {dt:.1f}s ({dt/total:.2f}s per game across fleet)\n"
        f"=======================================================\n"
    )
    print(report, flush=True)

    log_path = os.path.join(ROOT, "logs", f"clash_titanium_vs_ataxx_cpp_{total}g.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
        for r in results:
            f.write(f"Game {r['game_id']} (seed {r['seed']}, tit_black={r['titanium_black']}): "
                    f"{r['winner']} ({r['black_score']}-{r['white_score']}, {r['plies']} plies)\n")
    print(f"Saved full report to: {log_path}", flush=True)

    # Save every single position with Titanium eval and Ataxx_CPP eval into dataset JSONL
    import json
    jsonl_path = os.path.join(ROOT, "data", "nnue", f"clash_positions_{total}g.jsonl")
    total_positions = 0
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in results:
            for p in r.get("positions", []):
                p["game_id"] = r["game_id"]
                p["winner"] = r["winner"]
                jf.write(json.dumps(p) + "\n")
                total_positions += 1
    print(f"Saved {total_positions:,} evaluated positions with dual-engine scores to: {jsonl_path}", flush=True)
