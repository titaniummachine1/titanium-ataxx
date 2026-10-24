"""Stockfish recipe on the fixed dataset (user order 2026-10-24). Only the DATA changes vs the proven runs; every
training setting is copied from abB_twostage / v2 (outcome weight 0.25, Adam 1e-3, full weights, warm start from main).

Stage 1  sf_base : Stockfish's initial stage = the engine's own search labels on the WHOLE dataset.
                   All 32.5M rows, target 0.75*sigmoid(clip(titanium_cp,+-1500)/400) + 0.25*result, weight wsum, init S57.
Stage 2  sf_mix  : Stockfish's later stage = mixed own + Leela data. init = stage-1 net, all rows of v3:
                   KataGo rows -> KataGo's 200-visit search value (teacher_p3), own rows -> Titanium 5k label,
                   target 0.75*teacher_p3 + 0.25*result, weight wsum.
Schedule per stage: 6 epochs, ONE cosine decay 1e-3 -> ~7e-5 (no warm restarts, no plateau stop: that pair was buggy).
Gates (net-only change -> 5k fixed nodes) vs S57: 8 legs x 200 games, both seats, book 1000-1799, + determinism tripwire.
Nothing is merged: verdicts are the user's.  Logs: logs/twostage.txt, logs/train_sf_*.txt, logs/gate_sf_*_p*.txt
"""
import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight_v3m as o  # noqa: E402  (run/gate/log helpers)

ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL, inherited
o.LOG = o.ROOT + "/logs/twostage.txt"
o.STATUS = o.ROOT + "/logs/twostage_status.json"
EPOCHS = "6"
COMMON = ["--cache", "data/nnue/final/final_train_v3.npz", "--sidecar-dir", "data/nnue/final/sidecar",
          "--wdl-loss", "--wdl-scale", "400", "--outcome-col", "outcome_p", "--blend", "1.0",
          "--outcome-weight", "0.25", "--weight-col", "wsum", "--dirs", "0,1,2,3", "--chunk-rows", "800000",
          "--opt", "adam", "--opt-lr", "0.001", "--conv-lr", "0.003", "--wd", "0", "--freq-decay", "0.0",
          "--cyclic-T0", EPOCHS, "--cyclic-mult", "2.0", "--cyclic-lr-min", "1e-5",
          "--epochs", EPOCHS, "--max-epochs", EPOCHS, "--plateau", "0",
          "--seed", "7", "--threads", "4", "--min-free-mb", "1000", "--resume"]


def stage(tag, extra, init):
    out = "data/nnue/final_%s.tup" % tag
    rc = o.run("train_" + tag, [o.PY, "training/train_stream.py"] + COMMON + extra +
               ["--init", init, "--ckpt", "data/nnue/final_%s_ep{ep}.tup" % tag, "--out", out], "train_%s.txt" % tag)
    txt = open(o.ROOT + "/logs/train_%s.txt" % tag, errors="ignore").read()
    if "RAM GUARD" in txt:
        o.log("STAGE %s stopped by RAM guard -> partial, chain stops (rerun resumes)" % tag)
        return None
    if rc != 0 or not os.path.exists(o.ROOT + "/" + out) or ("epoch %s " % EPOCHS) not in txt:
        o.log("STAGE %s FAILED (rc=%d) -> chain stops" % (tag, rc))
        return None
    return out


o.log("=" * 70)
o.log("Stockfish recipe start: stage 1 Titanium base on all rows, stage 2 mixed Kata+Titanium, init S57")
base = stage("sf_base", ["--tcol", "titanium_cp"], o.CHAMP)
if base:
    mix = stage("sf_mix", ["--tcol", "teacher", "--teacher-prob-col", "teacher_p3"], base)
    if mix:
        o.gate("sf_mix", mix)       # the candidate
    o.gate("sf_base", base)         # diagnostic: what the base stage alone does
o.log("Stockfish recipe run finished")
