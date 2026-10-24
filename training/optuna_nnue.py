"""Phase A of the NNUE training plan (docs/ML_TRAINING_RESEARCH.md section 4): map the hyperparameter space.

Optuna TPE (multivariate: models which regions are good instead of sampling blindly) + Hyperband pruning
(successive halving: weak trials stop after 1 epoch) = the BOHB idea. Fixed 2.5M-row subset made of 10 contiguous
blocks spread over the file (both data sources, one sequential read each), held in RAM; fixed held-out val split.
Objective: best val mse vs the stage target (live or EMA weights). Study stored in SQLite -> resumable; the top
configs seed the PBT population (pbt_nnue.py). Training stage: base (Stockfish stage 1 labels).
Usage: python training/optuna_nnue.py --trials 30
"""
import argparse
import json
import os
import sys

import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nnue as T  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
OUT = ROOT + "/data/nnue/nnue_hpo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--subset", type=int, default=2_500_000)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--study", default="nnue_base_v1")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tr, va = T.load_splits("base", "")
    n = len(tr.rows)
    blocks = 10
    per = a.subset // blocks
    starts = np.linspace(0, n - per, blocks).astype(np.int64)
    rows = np.concatenate([np.arange(s, s + per) for s in starts])
    sub = tr.compact(rows)
    print("subset %d rows in RAM (%.0f MB idx)" % (len(sub.rows), sub.idx.nbytes / 1e6), flush=True)
    T._CACHE[("base", "")] = (sub, va)          # trials train on the subset

    def objective(trial):
        args = T.parser().parse_args([
            "--stage", "base", "--out", "%s/t%03d" % (OUT, trial.number),
            "--epochs", str(a.max_epochs), "--chunk", "500000", "--save-every", "1000000",
            "--opt", trial.suggest_categorical("opt", ["ranger", "adamw", "radam"]),
            "--lr", str(trial.suggest_float("lr", 1e-4, 5e-3, log=True)),
            "--wd", str(trial.suggest_float("wd", 1e-7, 1e-3, log=True)),
            "--bs", str(trial.suggest_categorical("bs", [4096, 8192, 16384])),
            "--la-k", str(trial.suggest_int("la_k", 4, 10)),
            "--la-alpha", str(trial.suggest_float("la_alpha", 0.3, 0.8)),
            "--ema", str(trial.suggest_categorical("ema", [0.0, 0.99, 0.999, 0.9999])),
            "--width", str(trial.suggest_categorical("width", [128, 256])),
            "--seed", str(7 + trial.number)])
        return T.run(args, trial=trial)

    study = optuna.create_study(
        study_name=a.study, storage="sqlite:///%s/optuna.db" % OUT, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=8, seed=1),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=a.max_epochs, reduction_factor=3))
    study.optimize(objective, n_trials=a.trials, gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value)
    top = [dict(number=t.number, value=t.value, params=t.params) for t in done[:8]]
    with open(OUT + "/top.json", "w") as f:
        json.dump(top, f, indent=1)
    print("TOP:", json.dumps(top[:4], indent=1), flush=True)


if __name__ == "__main__":
    main()
