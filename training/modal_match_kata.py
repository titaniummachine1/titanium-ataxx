"""Modal match gate: Titanium vs KataAtaxx on GPU (NVIDIA T4).
- 1.0s thinking time per move.
- Early stop: if first batch goes 20-0, log it and stop immediately.
- Saves all positions with evaluations to JSONL for NNUE training.
"""
import argparse
import math
import os
import random
import sys
import time

import modal

ROOT = "C:/gitProjects/titanium-ataxx"
ENGINE_SRC = os.environ.get("ENGINE_SRC", "C:/gitProjects/main-ataxx")
KATA_DIR = os.path.join(ROOT, "data", "engines", "kataataxx")

IGNORE_TITANIUM = [
    "target", "logs", "web", "training", "docs", "downloads", "botzone",
    "**/*.pt", "**/*.npz", "node_modules", ".git",
    "data/nnue/dis_positions.bin", "data/nnue/positions.bin", "data/nnue/positions_s*.bin",
    "data/nnue/*.npy", "data/nnue/labels.npz", "data/nnue/out_*", "data/nnue/relabel*",
]

CARGO = "/root/.cargo/bin/cargo"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "git", "libzip-dev", "zlib1g-dev", "build-essential", "curl")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .add_local_dir(ENGINE_SRC, "/eng/titanium", copy=True, ignore=IGNORE_TITANIUM)
    .run_commands("cd /eng/titanium && %s build -p titanium-cli --release" % CARGO)
    .run_commands(
        "git clone --depth 1 --branch Ataxx2026-lightvector-cuda https://github.com/hzyhhzy/KataGomo.git /eng/KataGomo",
        "cd /eng/KataGomo/cpp && cmake -B build -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)",
    )
    .add_local_file(os.path.join(KATA_DIR, "ext", "KataAtaxx_20260501_other_or_cpu", "weights", "b12c384h6tfrs2-s3413299200-d1434585433.onnx"), "/eng/models/kata_transformer.onnx")
    .add_local_file(os.path.join(KATA_DIR, "kata2025", "KataAtaxx_20250131", "weights", "b18.bin.gz"), "/eng/models/b18.bin.gz")
    .add_local_file(os.path.join(KATA_DIR, "ext", "KataAtaxx_20260501_other_or_cpu", "engine.cfg"), "/eng/models/engine.cfg")
    .add_local_file(os.path.join(ROOT, "training", "ataxx_move.py"), "/work/ataxx_move.py")
)

app = modal.App("titanium-vs-kataataxx-gpu", image=image)


def run_game_logic(args):
    import os
    import queue
    import re
    import subprocess
    import sys
    import threading
    import time
    sys.path.insert(0, "/work")
    from ataxx_move import RING, apply_move, name_of, parse_board, sq_of

    book_idx, ti_is_black, ms, book_line = args

    def bits(x):
        while x:
            b = x & -x
            yield b.bit_length() - 1
            x ^= b

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

    def vertex_to_xy(v):
        v = (v or '').strip().upper()
        if len(v) < 2 or not v[0].isalpha() or not v[1].isdigit():
            return None
        x = ord(v[0]) - ord('A')
        y = int(v[1:])
        if not (0 <= x < 7 and 1 <= y <= 7):
            return None
        return x, y

    # Start Titanium (CPU SIMD NNUE)
    ti_proc = subprocess.Popen(
        ["/eng/titanium/target/release/titanium-cli", "serve", "--threads", "1", "--nnue", "/eng/titanium/data/nnue/nnue_main.tnn"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )

    ti_q = queue.Queue()
    def ti_reader():
        for line in iter(ti_proc.stdout.readline, ''):
            ti_q.put(line.strip())
        ti_proc.stdout.close()
    threading.Thread(target=ti_reader, daemon=True).start()

    def ti_send(cmd):
        ti_proc.stdin.write(cmd + "\n")
        ti_proc.stdin.flush()

    ti_send("uai")
    while True:
        try:
            line = ti_q.get(timeout=10)
            if "uaiok" in line:
                break
        except queue.Empty:
            break

    ti_send("isready")
    while True:
        try:
            line = ti_q.get(timeout=10)
            if "readyok" in line:
                break
        except queue.Empty:
            break

    # Start KataAtaxx with CUDA on GPU
    # Check model: use transformer ONNX or b18
    model_path = "/eng/models/kata_transformer.onnx"
    if not os.path.exists(model_path):
        model_path = "/eng/models/b18.bin.gz"

    kata_proc = subprocess.Popen(
        ["/eng/KataGomo/cpp/build/katago", "gtp", "-model", model_path, "-config", "/eng/models/engine.cfg"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )

    kata_q = queue.Queue()
    def kata_reader():
        for line in iter(kata_proc.stdout.readline, ''):
            kata_q.put(line.strip())
        kata_proc.stdout.close()
    threading.Thread(target=kata_reader, daemon=True).start()

    def kata_cmd(cmd):
        kata_proc.stdin.write(cmd + "\n")
        kata_proc.stdin.flush()
        out = []
        while True:
            try:
                line = kata_q.get(timeout=15)
                out.append(line)
                if line.startswith('=') or line.startswith('?'):
                    return out
            except queue.Empty:
                return out

    # Init KataGo GTP
    kata_cmd("boardsize 7")
    kata_cmd("clear_board")
    kata_cmd("kata-set-param maxTime %.3f" % (ms / 1000.0))

    state = parse_board(book_line)
    moves_history = []
    positions_log = []

    res = None
    info = "UNKNOWN"

    try:
        ti_send("uainewgame")
        for ply in range(300):
            occ0, occ1, blk, stm = state
            occ = occ0 | occ1
            if not (can_move(occ0, occ) and can_move(occ1, occ)):
                break

            fen = to_fen(state)
            is_ti = (stm == 0) == ti_is_black

            pos_record = {
                "game": book_idx,
                "ply": ply,
                "fen": fen,
                "stm": "x" if stm == 0 else "o",
                "actor": "titanium" if is_ti else "kataataxx",
            }

            if is_ti:
                ti_send("position fen " + fen)
                ti_send("go movetime %d" % ms)
                best_move = None
                tit_cp = None
                while True:
                    try:
                        line = ti_q.get(timeout=10)
                        if "score cp" in line:
                            m = re.search(r"score cp (-?\d+)", line)
                            if m:
                                tit_cp = int(m.group(1))
                        if line.startswith("bestmove "):
                            best_move = line.split()[1]
                            break
                    except queue.Empty:
                        break

                if not best_move:
                    res = -1 if ti_is_black else 1
                    info = "TI-TIMEOUT"
                    break

                pos_record["tit_cp"] = tit_cp
                pos_record["move"] = best_move
                ns = apply_move(*state, best_move)
                if ns is None:
                    res = -1 if ti_is_black else 1
                    info = "TI-ILLEGAL %r" % best_move
                    break
                tok = best_move
            else:
                kata_cmd("setfen " + fen)
                color = "b" if stm == 0 else "w"
                out = kata_cmd("genmove " + color)
                mv = ''
                for l in out:
                    if l.startswith('='):
                        mv = l[1:].strip()
                xy = vertex_to_xy(mv)
                if xy is None or mv.lower() in ('pass', 'resign'):
                    res = 1 if ti_is_black else -1
                    info = "KATA-RESIGN/PASS %r" % mv
                    break

                fx, fy = xy
                # Stage 2 check for jumps
                bit_from = 1 << (sq_of("abcdefg"[fx] + str(fy)))
                own_mask = occ0 if stm == 0 else occ1
                if bit_from & own_mask:
                    out2 = kata_cmd("genmove " + color)
                    mv2 = ''
                    for l in out2:
                        if l.startswith('='):
                            mv2 = l[1:].strip()
                    xy2 = vertex_to_xy(mv2)
                    if xy2 is None:
                        res = 1 if ti_is_black else -1
                        info = "KATA-STAGE2-FAIL %r" % mv2
                        break
                    tok = "%s%d%s%d" % ("abcdefg"[fx], fy, "abcdefg"[xy2[0]], xy2[1])
                else:
                    tok = "%s%d" % ("abcdefg"[fx], fy)

                pos_record["move"] = tok
                ns = apply_move(*state, tok)
                if ns is None:
                    res = 1 if ti_is_black else -1
                    info = "KATA-ILLEGAL %r" % tok
                    break

            positions_log.append(pos_record)
            moves_history.append(tok)
            state = ns

        if res is None:
            occ0, occ1, blk, _ = state
            c0, c1 = occ0.bit_count(), occ1.bit_count()
            m0, m1 = can_move(occ0, occ0 | occ1), can_move(occ1, occ0 | occ1)
            if m0 and not m1:
                c0 = 49 - c1
            elif m1 and not m0:
                c1 = 49 - c0
            ti_c, kata_c = (c0, c1) if ti_is_black else (c1, c0)
            res = 1 if ti_c > kata_c else -1 if ti_c < kata_c else 0
            info = "%d-%d" % (ti_c, kata_c)

    finally:
        try:
            ti_send("quit")
            ti_proc.kill()
        except Exception:
            pass
        try:
            kata_cmd("quit")
            kata_proc.kill()
        except Exception:
            pass

    for p in positions_log:
        p["outcome_ti"] = res
        p["score_ti"] = 1.0 if res == 1 else 0.5 if res == 0 else 0.0

    return {
        "book_idx": book_idx,
        "ti_is_black": ti_is_black,
        "res": res,
        "info": info,
        "plies": len(moves_history),
        "positions": positions_log
    }


@app.function(gpu="T4", timeout=600)
def play_match_game(args):
    return run_game_logic(args)


@app.local_entrypoint()
def main(games: int = 200, ms: int = 1000, early_exit_batch: int = 10):
    import json
    book_path = os.path.join(ROOT, "data", "nnue", "book50k.txt")
    book = [l.strip() for l in open(book_path) if l.strip()]

    lines = games // 2
    tasks = [(i, black, ms, book[i]) for i in range(lines) for black in (True, False)]

    print(f"=== CLASH: Titanium vs KataAtaxx GPU (T4) ===")
    print(f"Total Games: {len(tasks)} (ms={ms}, early_stop_threshold={early_exit_batch})")

    out_jsonl = os.path.join(ROOT, "data", "nnue", "clash_positions_kata_200g.jsonl")
    log_file = os.path.join(ROOT, "logs", "clash_titanium_vs_kata_gpu.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    wins, losses, draws = 0, 0, 0
    scores = {}
    total_positions = 0

    with open(out_jsonl, "w", encoding="utf-8") as jf, open(log_file, "w", encoding="utf-8") as lf:
        batch_size = early_exit_batch
        for b_start in range(0, len(tasks), batch_size):
            b_tasks = tasks[b_start: b_start + batch_size]
            print(f"\n--- Launching Batch {b_start // batch_size + 1} (games {b_start + 1} to {b_start + len(b_tasks)}) ---")
            
            for res_dict in play_match_game.map(b_tasks):
                idx = res_dict["book_idx"]
                res = res_dict["res"]
                info = res_dict["info"]
                plies = res_dict["plies"]
                color = "x" if res_dict["ti_is_black"] else "o"

                if res == 1:
                    wins += 1
                    winner = "TITANIUM"
                elif res == -1:
                    losses += 1
                    winner = "KATAATAXX"
                else:
                    draws += 1
                    winner = "DRAW"

                scores.setdefault(idx, []).append((1.0 + res) / 2.0 if res is not None else 0.5)

                for pos in res_dict["positions"]:
                    jf.write(json.dumps(pos) + "\n")
                    total_positions += 1
                jf.flush()

                total_played = wins + losses + draws
                line_str = f"Game {total_played:3d} (book {idx:3d} ti={color}): {winner:10s} ({info:10s}) plies {plies:2d} | Score: +{wins} -{losses} ={draws}"
                print(line_str)
                lf.write(line_str + "\n")
                lf.flush()

                # Stop immediately if a wipeout is detected after >= 10 games
                if total_played >= early_exit_batch and (wins == total_played or losses == total_played):
                    print(f"\n[EARLY EXIT] WIPEOUT DETECTED: Score is {wins}-{losses} after {total_played} games! Stopping immediately to save compute.")
                    lf.write(f"\n[EARLY EXIT] WIPEOUT DETECTED: Score is {wins}-{losses} after {total_played} games.\n")
                    break

            total_played = wins + losses + draws
            if total_played >= early_exit_batch and (wins == total_played or losses == total_played):
                break

    pairs = [sum(v) / len(v) for v in scores.values() if len(v) == 2]
    total_games = wins + losses + draws
    score_pct = (wins + 0.5 * draws) / total_games if total_games else 0.0

    print(f"\n================ FINAL SUMMARY ================")
    print(f"Total Games Played: {total_games}")
    print(f"Titanium: +{wins} -{losses} ={draws} ({score_pct * 100:.1f}%)")
    print(f"Total Evaluated Positions Saved: {total_positions} -> {out_jsonl}")
    print(f"===============================================\n")
