"""SPSA search selectivity tuner (fishtest-style) for Titanium Disagreement Head (TweakSels).

Tunes the 7 selectivity & volatility knobs modulated by the TNN3 Disagreement Head:
  - dis_fut_num: volatility credit in Child Futility Pruning
  - dis_rfp_num: volatility dampening in Reverse Futility Pruning
  - dis_lmr_num: volatility discount on Late Move Reductions
  - fut_div: Child Futility divisor
  - rfp_scale: RFP baseline margin percentage
  - lmr_half: Moonbird LMR halflife parameter
  - dis_vol_thresh: centipawn disagreement threshold for +1 ply extension

Each iteration: draw delta in {-1,+1}^n, theta+ = theta + c_k*delta, theta- = theta - c_k*delta,
play ONE paired game (same random book line, both seats, `match --games 2`),
r = score(theta+) - 1 in [-1, 1], then theta += a_k * r / (2 c_k) * delta.

Usage:
  python training/spsa.py --bin ../main-ataxx/target/release/titanium-cli.exe --book data/nnue/book100k_diverse.txt --iters 600 --workers 6 --nodes 1500
"""
import argparse
import math
import os
import random
import subprocess
import tempfile
import threading
import time

# (start, lo, hi, c_end)
PARAMS = {
    "dis_fut_num": (48, 0, 128, 8),
    "dis_rfp_num": (32, 0, 96, 6),
    "dis_lmr_num": (0, 0, 64, 4),
    "fut_div": (57, 30, 90, 6),
    "rfp_scale": (100, 60, 160, 10),
    "lmr_half": (47, 20, 80, 5),
    "dis_vol_thresh": (300, 150, 500, 25),
}
R_END = 0.002


def spec(theta):
    return ",".join("%s=%d" % (k, int(round(v))) for k, v in theta.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to titanium-cli binary")
    ap.add_argument("--book", required=True, help="Opening book path")
    ap.add_argument("--net", default=None, help="Optional network path (default: bundled NNUE)")
    ap.add_argument("--iters", type=int, default=600, help="Total SPSA iterations")
    ap.add_argument("--workers", type=int, default=6, help="Parallel worker threads")
    ap.add_argument("--ms", type=int, default=0, help="Time limit per move (ms, 0 = use nodes)")
    ap.add_argument("--nodes", type=int, default=1500, help="Node limit per move (if ms == 0)")
    ap.add_argument("--book-lines", type=int, default=20000, help="Number of book lines to sample from")
    ap.add_argument("--out", default="spsa_trace.csv", help="Output trace CSV")
    a = ap.parse_args()

    bin_path = os.path.abspath(a.bin)
    book_path = os.path.abspath(a.book)
    bin_dir = os.path.dirname(bin_path)

    N, A = a.iters, 0.1 * a.iters
    theta = {k: float(v[0]) for k, v in PARAMS.items()}
    aa = {k: R_END * v[3] ** 2 * (A + N) ** 0.602 for k, v in PARAMS.items()}
    lock = threading.Lock()
    state = {"k": 0, "done": 0, "score": 0.0}
    names = list(PARAMS)

    with open(a.out, "w") as f:
        f.write("iter,pairs_done,mean_pair_score," + ",".join(names) + "\n")

    print(f"Starting SPSA Search Selectivity Tuning ({len(names)} knobs)")
    print(f"Initial: {spec(theta)}")
    print(f"Workers: {a.workers}, Iters: {N}, Limits: {f'{a.ms} ms' if a.ms > 0 else f'{a.nodes} nodes'}")

    rng = random.Random(12345)
    start_time = time.time()

    def worker(wid):
        tmp = os.path.join(tempfile.gettempdir(), "spsa_w%d.txt" % wid)
        while True:
            with lock:
                if state["k"] >= N:
                    return
                state["k"] += 1
                k = state["k"]
                delta = {n: rng.choice((-1, 1)) for n in names}
                ck = {n: PARAMS[n][3] * (N ** 0.101) / (k ** 0.101) for n in names}
                tp = {n: min(PARAMS[n][2], max(PARAMS[n][1], theta[n] + ck[n] * delta[n])) for n in names}
                tm = {n: min(PARAMS[n][2], max(PARAMS[n][1], theta[n] - ck[n] * delta[n])) for n in names}
                line = rng.randrange(a.book_lines)

            # Build command
            spec_p = spec(tp)
            spec_m = spec(tm)

            cmd = [
                bin_path, "match",
                "--games", "2",
                "--tune", spec_p,
                "--opp", f"{bin_path} serve --tune {spec_m}",
                "--book", book_path,
                "--book-start", str(line),
                "--start-game", "1",
                "--out", tmp,
            ]
            if a.ms > 0:
                cmd += ["--time", str(a.ms), "--opp-time", str(a.ms)]
            else:
                cmd += ["--nodes", str(a.nodes), "--opp-nodes", str(a.nodes)]

            if a.net:
                cmd += ["--network", a.net, "--opp-network", a.net]

            res = subprocess.run(cmd, cwd=bin_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            g = [l for l in open(tmp, errors="replace") if l.startswith("-- game")] if os.path.exists(tmp) else []
            if len(g) != 2:
                continue

            sc = sum(1.0 if "titanium wins" in l else 0.0 if "opponent wins" in l else 0.5 for l in g)
            r = sc - 1.0

            with lock:
                for n in names:
                    ak = aa[n] / ((A + k) ** 0.602)
                    theta[n] = min(PARAMS[n][2], max(PARAMS[n][1], theta[n] + ak * r / (2 * ck[n]) * delta[n]))
                state["done"] += 1
                state["score"] += sc / 2
                done = state["done"]
                if done % 10 == 0 or done == N:
                    elapsed = time.time() - start_time
                    rate = done / max(elapsed, 0.001)
                    print(
                        f"[{done}/{N} pairs, {elapsed:.1f}s, {rate:.2f} pairs/s] "
                        f"Mean score: {state['score'] / done:.4f} | {spec(theta)}"
                    )
                    with open(a.out, "a") as f:
                        f.write(
                            f"{k},{done},{state['score'] / done:.4f},"
                            + ",".join(f"{theta[n]:.2f}" for n in names)
                            + "\n"
                        )

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    with open(a.out, "a") as f:
        f.write(
            f"FINAL,{state['done']},{state['score'] / max(state['done'], 1):.4f},"
            + ",".join(f"{theta[n]:.2f}" for n in names)
            + "\n"
        )
    print("\n=== TUNING COMPLETE ===")
    print("FINAL SPEC:", spec(theta))


if __name__ == "__main__":
    main()
