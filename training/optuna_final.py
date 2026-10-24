"""Optuna HPO on the FINAL dataset (port of optuna_stage1.py, 2026-10-22).

What changed vs the original (nothing else):
- CACHE/VAL: stage1_base_* -> data/nnue/final/final_{train,val}.npz (33.9M rows).
- Trials run FULL rows (rows=0): user order "final dataset, nothing else matters".
- Added WDL truth columns: --outcome-col outcome_p --teacher-prob-col teacher_p --blend 1.0.
- Added --sidecar-dir when trial dirs == manifest dirs (else full recompute; values
  identical, slower trial). Sidecar grid is dirs-specific; other combos recompute.
- Fresh OUTDIR / STUDY_DB / LOG: nothing old touched, everything shelved per trial.
- Per-phase timeouts (full-data trials are ~40min/ep): phase1 2h, phase2/PBT 3h, final 6h.
- Added --trials (default 32, original design). --trials 1 = single verification trial.
- Selection unchanged from original: val screens, nightly.py gates crown, all .tup kept.

Launch: NOTHING runs on import. Needs explicit user order. Verify first:
  python training/optuna_final.py --trials 1   (then STOP, report, await order)
Full study:
  python training/optuna_final.py --trials 32
"""
import argparse
import json
import os
import re
import subprocess
import time

import optuna

ROOT = r"C:\gitProjects\titanium-ataxx"
TRAIN = os.path.join(ROOT, "training", "train_stream.py")
CACHE = os.path.join(ROOT, "data", "nnue", "final", "final_train.npz")
VAL = os.path.join(ROOT, "data", "nnue", "final", "final_val.npz")
INIT = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")
SIDECAR = os.path.join(ROOT, "data", "nnue", "final", "sidecar")
OUTDIR = os.path.join(ROOT, "data", "nnue", "optuna_final")
LOG = os.path.join(ROOT, "logs", "optuna_final.log")
STUDY_DB = os.path.join(ROOT, "data", "nnue", "optuna_final.db")
os.makedirs(OUTDIR, exist_ok=True)

DIR_CHOICES = [
    "0", "1", "2", "3",
    "0,1", "0,2", "0,3", "1,2", "1,3", "2,3",
    "0,1,2", "0,1,3", "0,2,3", "1,2,3",
    "0,1,2,3",
]
SIDECAR_DIRS = "0,1,2,3"


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def config_from_trial(trial):
    cfg = {
        "opt": trial.suggest_categorical("opt", ["adamw", "nadam", "radam", "adam"]),
        "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
        "conv_ratio": trial.suggest_float("conv_ratio", 0.5, 4.0, log=True),
        "wd": trial.suggest_float("wd", 1e-8, 1e-4, log=True),
        "outcome": trial.suggest_float("outcome", 0.05, 0.95),
        "wdl_scale": trial.suggest_float("wdl_scale", 200.0, 800.0, log=True),
        "dirs": trial.suggest_categorical("dirs", DIR_CHOICES),
        "lookahead": trial.suggest_categorical("lookahead", [False, True]),
        "cyclic_t0": trial.suggest_categorical("cyclic_t0", [0, 2, 4]),
    }
    if cfg["lookahead"]:
        cfg["lookahead_k"] = trial.suggest_int("lookahead_k", 2, 10)
        cfg["lookahead_alpha"] = trial.suggest_float("lookahead_alpha", 0.2, 0.8)
    return cfg


def run_config(tag, rows, epochs, cfg, init=INIT, timeout=7200):
    out = os.path.join(OUTDIR, tag + ".tup")
    cmd = [
        "python", TRAIN,
        "--cache", CACHE,
        "--rows", str(rows),
        "--epochs", str(epochs),
        "--max-epochs", str(epochs),
        "--plateau", "0",
        "--init", init,
        "--out", out,
        "--val-probe", VAL,
        "--tcol", "teacher",
        "--blend", "1.0",
        "--outcome-col", "outcome_p",
        "--teacher-prob-col", "teacher_p",
        "--outcome-weight", str(cfg["outcome"]),
        "--wdl-loss",
        "--wdl-scale", str(cfg["wdl_scale"]),
        "--dirs", cfg["dirs"],
        "--opt", cfg["opt"],
        "--opt-lr", str(cfg["lr"]),
        "--conv-lr", str(cfg["lr"] * cfg["conv_ratio"]),
        "--wd", str(cfg["wd"]),
        "--freq-decay", "0",
        "--threads", "4",
    ]
    if cfg["dirs"] == SIDECAR_DIRS:
        cmd += ["--sidecar-dir", SIDECAR]
    else:
        log("NOTE %s: dirs %s != sidecar grid, full recompute (slower, same values)"
            % (tag, cfg["dirs"]))
    if cfg.get("lookahead"):
        cmd += ["--lookahead", "--lookahead-k", str(cfg["lookahead_k"]),
                "--lookahead-alpha", str(cfg["lookahead_alpha"])]
    if cfg.get("cyclic_t0"):
        cmd += ["--cyclic-T0", str(cfg["cyclic_t0"]), "--cyclic-mult", "2",
                "--cyclic-lr-min", "1e-5"]
    log("START %s rows=%d epochs=%d cfg=%s" % (tag, rows, epochs, json.dumps(cfg)))
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, timeout=timeout)
    text = p.stdout or ""
    with open(os.path.join(OUTDIR, tag + ".log"), "w", encoding="utf-8") as f:
        f.write(text)
    vals = re.findall(r"val_mse\s+([0-9.]+)", text)
    if p.returncode != 0 or not vals:
        log("FAIL %s rc=%d vals=%s tail=%s" % (
            tag, p.returncode, vals, text[-500:].replace("\n", " | ")))
        return None, out, time.time() - t0
    val = float(vals[-1])
    log("DONE %s val_mse=%.8f elapsed=%.0fs out=%s" % (
        tag, val, time.time() - t0, out))
    return val, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=32)
    n_trials = ap.parse_args().trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, constant_liar=True,
        n_startup_trials=12, seed=12345,
    )
    study = optuna.create_study(
        study_name="titanium_final",
        storage="sqlite:///" + STUDY_DB.replace("\\", "/"),
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    log("study trials before: %d" % len(study.trials))

    def objective(trial):
        cfg = config_from_trial(trial)
        val, _, _ = run_config("opt_%04d" % trial.number, 0, 1, cfg, timeout=7200)
        if val is None:
            return float("inf")
        trial.set_user_attr("dirs", cfg["dirs"])
        return val

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    complete = [t for t in study.trials if t.value is not None and t.value < float("inf")]
    complete.sort(key=lambda t: t.value)
    log("phase1 complete=%d best=%.8f" % (len(complete), complete[0].value if complete else -1))
    if not complete:
        return
    top = []
    for t in complete[:5]:
        cfg = dict(t.params)
        cfg["outcome"] = cfg.pop("outcome")
        cfg.setdefault("lookahead", False)
        cfg.setdefault("cyclic_t0", 0)
        val, out, _ = run_config("phase2_t%04d" % t.number, 0, 2, cfg, timeout=10800)
        if val is not None:
            top.append((val, cfg, out, t.number))
    top.sort(key=lambda x: x[0])
    if not top:
        return
    best_val, best_cfg, best_out, best_num = top[0]
    log("phase2 best val=%.8f trial=%d cfg=%s" % (
        best_val, best_num, json.dumps(best_cfg)))

    pbt = []
    for i, factor in enumerate((0.8, 1.2, 1.6)):
        cfg = dict(best_cfg)
        cfg["lr"] = max(1e-5, cfg["lr"] * factor)
        val, out, _ = run_config("pbt_%02d" % i, 0, 1, cfg, init=best_out, timeout=10800)
        if val is not None:
            pbt.append((val, cfg, out))
    pbt.sort(key=lambda x: x[0])
    if pbt and pbt[0][0] < best_val:
        best_val, best_cfg, best_out = pbt[0]
        log("PBT wins val=%.8f cfg=%s" % (best_val, json.dumps(best_cfg)))

    with open(os.path.join(OUTDIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "phase1_best": complete[0].value if complete else None,
            "phase2_top": [{"val": v, "trial": n, "out": o, "cfg": c} for v, c, o, n in top],
            "pbt": [{"val": v, "out": o, "cfg": c} for v, c, o in pbt],
            "best": {"val": best_val, "cfg": best_cfg, "out": best_out},
        }, f, indent=2)

    log("FINAL full 6 epochs from best")
    val, out, _ = run_config("final_stage1", 0, 6, best_cfg, init=best_out, timeout=21600)
    log("FINAL val=%.8f out=%s" % (val if val is not None else -1, out))
    log("optuna complete")


if __name__ == "__main__":
    main()
