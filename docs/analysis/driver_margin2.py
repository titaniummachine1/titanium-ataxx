"""D662: per-move margins of the Botzone driver, split into (a) what the ENGINE reports about its own answer time and (b) what a
requester (the simulator, standing in for Botzone) measures from "request written" to "marker read".
margin_sim = limit - sim_elapsed ; margin_engine = limit - t (t = the engine's own 't<ms>' in the debug line, measured from the turn clock start)
outside = margin_engine - margin_sim  = pipe + process wake-up + reader latency (what the engine cannot see).
usage: python driver_margin2.py --repo <worktree> --lib <dll/so> --net <tnn> [--games 3] [--scale 0.25] [--load 0,4,8] [--moves 14] [--mode long|pyloop]
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--lib", required=True)
ap.add_argument("--net", required=True)
ap.add_argument("--games", type=int, default=3)
ap.add_argument("--scale", type=float, default=0.25)
ap.add_argument("--load", default="0,4,8")
ap.add_argument("--moves", type=int, default=14)
ap.add_argument("--mode", default="long")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "botzone"))
import sim_local as S  # noqa: E402


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def one_game(driver_black, seed):
    random.seed(seed)
    env = dict(os.environ, TI_DEBUG="0", TI_MODE="long", TI_LIB=a.lib, TI_NET=a.net, TI_TIME_SCALE=str(a.scale), TI_TT_BITS="22")
    if a.mode == "pyloop":
        env["TI_PYLOOP"] = "1"
    script = os.path.join(a.repo, "botzone", "titanium_botzone.py")
    proc = subprocess.Popen([sys.executable, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    g = S.Game()
    drv = 0 if driver_black else 1
    out, first, req, n = [], True, (-1, -1, -1, -1), 0
    try:
        while not g.over():
            if g.turn == drv:
                limit = (12000 if first else 4000) * a.scale
                t = time.perf_counter()
                proc.stdin.write(("1\n%d %d %d %d\n" % req) if first else ("%d %d %d %d\n" % req))
                proc.stdin.flush()
                resp = S.read_response(proc)
                el = (time.perf_counter() - t) * 1000
                m = re.search(r" t(\d+) ", " " + resp[1] + " ") if len(resp) > 1 else None
                te = float(m.group(1)) if m else None
                g.play(tuple(int(v) for v in resp[0].split()))
                out.append((first, limit - el, (limit - te) if te is not None else None, "FALLBACK" in (resp[1] if len(resp) > 1 else "")))
                first = False
                n += 1
                if n >= a.moves:
                    break
            else:
                mv = S.greedy(g)
                req = mv
                g.play(mv)
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    return out


for load in [int(x) for x in a.load.split(",")]:
    busy = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(load)]
    rows = []
    try:
        for gi in range(a.games):
            rows += one_game(gi % 2 == 0, 200 + gi)
    finally:
        for b in busy:
            b.kill()
    later = [r for r in rows if not r[0]]
    first = [r for r in rows if r[0]]
    ms = [r[1] for r in later]
    me = [r[2] for r in later if r[2] is not None]
    outside = [r[2] - r[1] for r in later if r[2] is not None]
    print("mode %s load %d | later turns n=%d (limit %.0f ms): SIM margin min %.1f p5 %.1f p50 %.1f | ENGINE-reported margin min %.1f p5 %.1f p50 %.1f | outside-engine latency p50 %.1f p95 %.1f max %.1f | LATE(sim) %d | fallbacks %d | first turn sim margins %s"
          % (a.mode, load, len(later), 4000 * a.scale, min(ms), pct(ms, 0.05), pct(ms, 0.5), min(me) if me else 0, pct(me, 0.05) if me else 0, pct(me, 0.5) if me else 0,
             pct(outside, 0.5) if outside else 0, pct(outside, 0.95) if outside else 0, max(outside) if outside else 0,
             len([m for m in ms if m < 0]), len([r for r in rows if r[3]]), ["sim %.0f / engine %s" % (r[1], ("%.0f" % r[2]) if r[2] is not None else "?") for r in first]))
