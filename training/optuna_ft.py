"""Hyperparameter search for CONTINUING from a saturated net (D83): Optuna TPE + Hyperband pruning, every trial starts from
--init (the stage-1 result) and trains --epochs full epochs on the whole block-mixed training set (train_nnue.run, same
code as the real run), objective = best val mse vs the stage target (live or EMA). Only optimizer settings are searched:
the target (lambda, labels) stays fixed, so the val numbers are comparable (D39).
A trial is only worth keeping if it beats the init net's val by > 0.2% (the auto-lr min-rel); short trials favour settings
that drop the loss fast (short-horizon bias), so a small win is suspect.
Usage (inside the Modal container, see modal_train.py hpo): python optuna_ft.py --init NET.pt --out DIR --trials 16
"""
import argparse
import json
import os
import sys

import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nnue as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="base")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--study", default="ft_base_v4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def objective(trial):
        argv = ["--stage", a.stage, "--init", a.init, "--out", "%s/t%03d" % (a.out, trial.number),
                "--epochs", str(a.epochs), "--save-every", "1000000", "--block", "1024", "--threads", str(a.threads),
                "--opt", trial.suggest_categorical("opt", ["ranger", "adamw"]),
                "--lr", str(trial.suggest_float("lr", 5e-5, 4e-3, log=True)),
                "--lr-min", "1e-6",
                "--wd", str(trial.suggest_float("wd", 1e-8, 1e-4, log=True)),
                "--bs", str(trial.suggest_categorical("bs", [4096, 8192, 16384])),
                "--la-k", str(trial.suggest_int("la_k", 4, 10)),
                "--la-alpha", str(trial.suggest_float("la_alpha", 0.3, 0.8)),
                "--ema", str(trial.suggest_categorical("ema", [0.999, 0.9995, 0.9999])),
                "--width", "256", "--seed", str(7 + trial.number), "--min-free-gb", "1.5"]
        if trial.suggest_categorical("hypergrad", [False, True]):
            argv.append("--hypergrad")
        v = T.run(T.parser().parse_args(argv), trial=trial)
        print("TRIAL %d -> %.6f %s" % (trial.number, v, json.dumps(trial.params)), flush=True)
        return v

    study = optuna.create_study(
        study_name=a.study, storage="sqlite:///%s/optuna.db" % a.out, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=6, seed=1),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=a.epochs, reduction_factor=3))
    left = a.trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left, gc_after_trial=True)
    done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
    top = [dict(number=t.number, value=t.value, params=t.params) for t in done[:8]]
    with open(a.out + "/top.json", "w") as f:
        json.dump(top, f, indent=1)
    print("TOP:", json.dumps(top[:4], indent=1), flush=True)


if __name__ == "__main__":
    main()
