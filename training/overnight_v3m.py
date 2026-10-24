"""Overnight chain 2026-10-24 (user-ordered: "keep it running overnight and get best results").

Stages, strictly one job at a time, each logged with timestamps to logs/overnight_v3m.txt:
  1. Moonbird@1k relabel of a stratified KataGo calibration sample (100k rows)        relabel_moonbird.py --part calib
  2. fit Moonbird cp->p scale per empties band against KataGo TD h1                   calibrate_moonbird.py
  3. Moonbird@1k relabel of all 11.87M own-game train rows (~3 h)                      relabel_moonbird.py --part train
     (1-3 failing -> fallback: no Moonbird, own rows keep the Titanium 5k label)
  4. build final_{val,test,train}_v3m.npz (Kata TD h1 + Moonbird + per-row lambda ow_row + Kata-only weight column)
  5. TRAIN A  "v3m": all rows (Kata + Moonbird), init S57, 4 epochs, cosine LR 1e-3 -> 1e-5
  6. GATE  A  vs S57 @5k nodes: 8 legs x 200 games (book lines 1000-1799, both seats) + determinism tripwire
  7. TRAIN B  "v3k": Kata rows only (own rows weight 0) -- answers "train without the Titanium positions?"
  8. GATE  B  same protocol
Nothing is merged or rejected: verdicts are for the user (iron laws 3/4/10).
STOP file C:/gitProjects/logs/STOP halts between stages (and between relabel chunks).
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = r"C:/gitProjects/titanium-ataxx"
PY = sys.executable
LOG = ROOT + "/logs/overnight_v3m.txt"
STATUS = ROOT + "/logs/overnight_v3m_status.json"
STOP = r"C:/gitProjects/logs/STOP"
FINAL = ROOT + "/data/nnue/final"
EXE = r"C:/gitProjects/main-ataxx/target/release/titanium-cli.exe"
CHAMP = "data/nnue/final_v2_wdl_ep8.tup"          # S57 = main default net (sha 53F2B29C)
BOOK = "data/nnue/book50k.txt"
NODES = 1000
state = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save():
    with open(STATUS, "w") as f:
        json.dump(state, f, indent=1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8].upper()


def check_stop():
    if os.path.exists(STOP):
        log("STOP file present -> halting chain")
        state["halted"] = "STOP file"
        save()
        sys.exit(3)


def run(name, args, out):
    check_stop()
    log("START %s: %s" % (name, " ".join(args)))
    t0 = time.time()
    with open(ROOT + "/logs/" + out, "a", encoding="utf-8") as f:
        rc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
    dt = time.time() - t0
    state["stages"][name] = {"rc": rc, "secs": round(dt), "log": "logs/" + out}
    save()
    log("END   %s rc=%d (%.0f min)" % (name, rc, dt / 60))
    return rc


def train(tag, weight_col, cache):
    out = "data/nnue/final_%s.tup" % tag
    rc = run("train_" + tag, [PY, "training/train_stream.py",
                              "--cache", cache, "--sidecar-dir", "data/nnue/final/sidecar",
                              "--wdl-loss", "--wdl-scale", "400", "--outcome-col", "outcome_p",
                              "--teacher-prob-col", "teacher_p3", "--weight-col", weight_col,
                              "--outcome-weight-col", "ow_row",
                              "--tcol", "teacher", "--blend", "1.0", "--outcome-weight", "0.0",
                              "--dirs", "0,1,2,3", "--chunk-rows", "800000",
                              "--opt", "adam", "--opt-lr", "0.001", "--conv-lr", "0.003", "--wd", "0", "--freq-decay", "0.0",
                              "--cyclic-T0", "4", "--cyclic-mult", "2.0", "--cyclic-lr-min", "1e-5",
                              "--init", CHAMP, "--epochs", "4", "--max-epochs", "4", "--plateau", "0",
                              "--seed", "7", "--threads", "4", "--min-free-mb", "1000", "--resume",
                              "--ckpt", "data/nnue/final_%s_ep{ep}.tup" % tag, "--out", out],
             "train_%s.txt" % tag)
    guard = "RAM GUARD" in open(ROOT + "/logs/train_%s.txt" % tag, errors="ignore").read()
    if guard:
        log("TRAIN %s stopped by its RAM guard -> partial net, NOT gated (resume with the same command)" % tag)
        return None
    if rc != 0 or not os.path.exists(ROOT + "/" + out):
        log("TRAIN %s FAILED (rc=%d) -- no gate" % (tag, rc))
        return None
    return out


def games_block(path, n):
    """first n game records (result line + move line) for the determinism tripwire"""
    lines = open(path, errors="ignore").read().splitlines()
    out, k = [], 0
    for i, l in enumerate(lines):
        if l.startswith("-- game"):
            k += 1
            if k > n:
                break
            out.append(l)
            if i + 1 < len(lines):
                out.append(lines[i + 1])
    return out


def gate(tag, cand):
    check_stop()
    legs = 8
    hashes = {"exe": sha(EXE), "cand": sha(ROOT + "/" + cand), "champ": sha(ROOT + "/" + CHAMP)}
    log("GATE %s hashes %s" % (tag, hashes))
    procs, files = [], []
    t0 = time.time()
    for i in range(legs):
        f = "logs/gate_%s_p%d.txt" % (tag, i)
        if os.path.exists(ROOT + "/" + f):
            os.remove(ROOT + "/" + f)
        files.append(f)
        procs.append(subprocess.Popen([EXE, "match", "--games", "200", "--nodes", "5000", "--tuple", cand,
                                       "--opp", "self", "--opp-tuple", CHAMP, "--opp-nodes", "5000",
                                       "--book", BOOK, "--book-start", str(1000 + 100 * i), "--start-game", "1",
                                       "--out", f], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    rcs = [p.wait() for p in procs]
    log("GATE %s legs done rc=%s (%.0f min)" % (tag, rcs, (time.time() - t0) / 60))
    # determinism tripwire: replay the first 20 games of leg 0; must be bit-identical
    trip = "logs/gate_%s_trip.txt" % tag
    if os.path.exists(ROOT + "/" + trip):
        os.remove(ROOT + "/" + trip)
    subprocess.run([EXE, "match", "--games", "20", "--nodes", "5000", "--tuple", cand, "--opp", "self",
                    "--opp-tuple", CHAMP, "--opp-nodes", "5000", "--book", BOOK, "--book-start", "1000",
                    "--start-game", "1", "--out", trip], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a, b = games_block(ROOT + "/" + files[0], 20), games_block(ROOT + "/" + trip, 20)
    ident = len(a) == len(b) == 40 and a == b
    log("GATE %s tripwire replay of 20 games: %s" % (tag, "BIT-IDENTICAL" if ident else "MISMATCH -> RUN VOID"))
    pooled = ROOT + "/logs/gate_%s_pooled.txt" % tag
    with open(pooled, "w", encoding="utf-8") as fo:
        for f in files:
            fo.write(open(ROOT + "/" + f, errors="ignore").read())
    summ = subprocess.run([PY, "training/gate_summary.py", pooled], cwd=ROOT, capture_output=True, text=True)
    for l in summ.stdout.splitlines():
        log("  " + l)
    state["stages"]["gate_" + tag] = {"hashes": hashes, "tripwire_identical": ident, "summary": summ.stdout,
                                      "legs": files}
    save()


def build(suffix, moon):
    bargs = [PY, "training/build_v3_targets.py", "--splits", "val,test,train", "--suffix", suffix]
    if moon:
        bargs += ["--moon-dir", FINAL + "/moon%d" % NODES]
    if run("build" + suffix, bargs, "build%s.txt" % suffix) != 0:
        log("BUILD %s FAILED" % suffix)
        return None
    for l in open(ROOT + "/logs/build%s.txt" % suffix, errors="ignore"):
        if l.startswith("train "):
            log("  build: " + l.rstrip()[:1200])
    return "data/nnue/final/final_train%s.npz" % suffix


def main():
    log("=" * 70)
    log("overnight chain v2 start (A: kata+titanium, B: kata-only, then Moonbird relabel -> C)")
    cache = build("_v3t", moon=False)
    if cache:
        cand = train("v3t", "wsum3", cache)       # A: Kata TD + own rows (Titanium 5k, weight x0.5)
        if cand:
            gate("v3t", cand)
        cand = train("v3k", "wsum_kataonly", cache)   # B: Kata rows only
        if cand:
            gate("v3k", cand)
    moon_ok = run("relabel_calib", [PY, "training/relabel_moonbird.py", "--part", "calib", "--nodes", str(NODES),
                                    "--workers", "8"], "moon_calib.txt") == 0
    moon_ok = moon_ok and run("calibrate", [PY, "training/calibrate_moonbird.py", "--nodes", str(NODES)],
                              "moon_calibrate.txt") == 0
    if moon_ok:
        for l in open(ROOT + "/logs/moon_calibrate.txt", errors="ignore"):
            log("  calib: " + l.rstrip())
    moon_ok = moon_ok and run("relabel_train", [PY, "training/relabel_moonbird.py", "--part", "train",
                                                "--nodes", str(NODES), "--workers", "8"], "moon_train.txt") == 0
    state["moon_ok"] = moon_ok
    if not moon_ok:
        log("MOONBIRD STAGE FAILED -> no C run")
    else:
        cache = build("_v3m", moon=True)
        if cache:
            cand = train("v3m", "wsum3", cache)   # C: Kata TD + own rows relabeled by Moonbird@1k (weight x1)
            if cand:
                gate("v3m", cand)
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log("overnight chain finished")


if __name__ == "__main__":
    main()
