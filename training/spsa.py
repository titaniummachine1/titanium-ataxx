"""SPSA search tuner (fishtest-style) for the exp/search-spsa tuning build. Runs anywhere with python3 + the binary.

Each iteration: draw delta in {-1,+1}^n, theta+ = theta + c_k*delta, theta- = theta - c_k*delta, play ONE paired game
pair theta+ vs theta- (same random book line, both seats, `match --games 2`, uainewgame per game), result
r = score(theta+) - 1 in [-1, 1], then theta += a_k * r / (2 c_k) * delta (per parameter, in its own units).
Schedules (Spall / fishtest): c_k = c_end * N^0.101 / k^0.101 (so c_N = c_end), a_k = a / (A + k)^0.602 with
a = R_end * c_end^2 * (A + N)^0.602 (so the end step is R_end * c_end^2), A = 0.1 N.
Workers run asynchronously (fishtest does the same); every 100 iterations the current theta goes to spsa_trace.csv.
Time control on purpose (not fixed nodes): a knob cannot win by doing work the node counter does not see.

Usage: python3 spsa.py --bin ./bins/spsa --net net.tup --book book.txt --iters 6000 --workers 16 --ms 50
"""
import argparse
import math
import os
import random
import subprocess
import threading

# name: (start = staging b0c4329, lo, hi, c_end)
PARAMS = {
    "fut_div": (57, 20, 150, 8),
    "rfp_scale": (100, 30, 250, 15),
    "lmr_half": (47, 10, 100, 6),
    "lmr_min": (6, 2, 12, 1),
    "nmp_sub": (3, 0, 6, 1),
    "nmp_dpct": (0, 0, 60, 8),
    "nmp_min": (3, 1, 6, 1),
    "nmp_targets": (11, 0, 25, 2),
    "nmp_fill": (26, 15, 40, 2),
    "ord_hist": (2000, 0, 6000, 300),
    "ord_jump": (500, -500, 2000, 100),
    "ord_conn": (250, 0, 800, 60),
    "ord_clone": (500, -500, 2000, 100),
}
R_END = 0.002


def spec(theta):
    return ",".join("%s=%d" % (k, int(round(v))) for k, v in theta.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ms", type=int, default=50)
    ap.add_argument("--book-lines", type=int, default=20000)
    ap.add_argument("--out", default="spsa_trace.csv")
    a = ap.parse_args()
    N, A = a.iters, 0.1 * a.iters
    theta = {k: float(v[0]) for k, v in PARAMS.items()}
    aa = {k: R_END * v[3] ** 2 * (A + N) ** 0.602 for k, v in PARAMS.items()}
    lock = threading.Lock()
    state = {"k": 0, "done": 0, "score": 0.0}
    names = list(PARAMS)
    with open(a.out, "w") as f:
        f.write("iter,pairs_done,mean_pair_score," + ",".join(names) + "\n")
    rng = random.Random(12345)

    def worker(wid):
        tmp = "/tmp/spsa_w%d.txt" % wid
        while True:
            with lock:
                if state["k"] >= N:
                    return
                state["k"] += 1
                k = state["k"]
                delta = {n: rng.choice((-1, 1)) for n in names}
                ck = {n: PARAMS[n][3] * N ** 0.101 / k ** 0.101 for n in names}
                tp = {n: min(PARAMS[n][2], max(PARAMS[n][1], theta[n] + ck[n] * delta[n])) for n in names}
                tm = {n: min(PARAMS[n][2], max(PARAMS[n][1], theta[n] - ck[n] * delta[n])) for n in names}
                line = rng.randrange(a.book_lines)
            env = dict(os.environ, TITANIUM_TUNE=spec(tp))
            cmd = [a.bin, "match", "--games", "2", "--time", str(a.ms), "--tuple", a.net,
                   "--opp", "env TITANIUM_TUNE=%s %s serve" % (spec(tm), a.bin), "--opp-tuple", a.net,
                   "--opp-time", str(a.ms), "--book", a.book, "--book-start", str(line), "--start-game", "1", "--out", tmp]
            subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            g = [l for l in open(tmp, errors="replace") if l.startswith("-- game")] if os.path.exists(tmp) else []
            if len(g) != 2:
                continue
            sc = sum(1.0 if "titanium wins" in l else 0.0 if "opponent wins" in l else 0.5 for l in g)
            r = sc - 1.0
            with lock:
                for n in names:
                    ak = aa[n] / (A + k) ** 0.602
                    theta[n] = min(PARAMS[n][2], max(PARAMS[n][1], theta[n] + ak * r / (2 * ck[n]) * delta[n]))
                state["done"] += 1
                state["score"] += sc / 2
                if state["done"] % 100 == 0:
                    with open(a.out, "a") as f:
                        f.write("%d,%d,%.4f," % (k, state["done"], state["score"] / state["done"])
                                + ",".join("%.2f" % theta[n] for n in names) + "\n")

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    with open(a.out, "a") as f:
        f.write("FINAL,%d,%.4f," % (state["done"], state["score"] / max(state["done"], 1))
                + ",".join("%.2f" % theta[n] for n in names) + "\n")
    print("FINAL", spec(theta))


if __name__ == "__main__":
    main()
