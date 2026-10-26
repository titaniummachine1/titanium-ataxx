"""Optuna search for the futility small net (D114, user 2026-10-26: "optimise until saturated"; "adjust the net to take the
least time for maximum fitness"). Objective = expected microseconds per futility call at a FIXED wrong-prune rate (1.43%,
the deployed net's): small-net cost + (1 - skip) * big-net cost. So a bigger net must pay for its time with skipped big-net
evals. Each trial = one smallnet_train.py run in bound mode to saturation, scored by smallnet_eval2.py (best head, shifted).
Cost model (us, PC Haswell, re-measure in the engine after): hand eval 0.07; extras 0.05; small net 0.18 at W32 H16,
scaled 0.3 fixed + 0.7 * (W * H) / 512; big net 6.6.
Stops when the best value has not improved by > 0.5% for --patience completed trials (saturated) or at --trials.
Usage (sandbox): SN_DATA=/tmp/sn python smallnet_optuna.py --jobs 4 --threads 4 [--extra] [--alpha-in]
"""
import argparse
import json
import os
import subprocess
import sys
import threading

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
BIG_US = 6.6


def sn_cost(w, h, extra):
    return 0.07 + (0.05 if extra else 0.0) + 0.18 * (0.3 + 0.7 * w * h / 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--alpha-in", action="store_true")
    ap.add_argument("--db", default="sqlite:///sn_optuna.db")
    a = ap.parse_args()
    lock = threading.Lock()
    state = {"best": float("inf"), "since": 0}
    study = optuna.create_study(study_name="sn", storage=a.db, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=8))

    if not study.trials:
        # trial 0 = arm D of D114 trained from scratch (the warm-started version scored 72.7% skipped at 1.43% wrong prunes)
        study.enqueue_trial({"width": 32, "hidden": 16, "lr": 1e-3, "wd": 1e-6, "bs": 16384, "focus": 0.5})

    def objective(tr):
        w = tr.suggest_categorical("width", [16, 24, 32, 48, 64])
        h = tr.suggest_categorical("hidden", [8, 16, 24, 32])
        lr = tr.suggest_float("lr", 5e-4, 6e-3, log=True)
        wd = tr.suggest_float("wd", 1e-7, 1e-4, log=True)
        bs = tr.suggest_categorical("bs", [8192, 16384, 32768])
        focus = tr.suggest_categorical("focus", [0.0, 0.25, 0.5, 1.0])
        out = "opt_%03d.pt" % tr.number
        cmd = [sys.executable, os.path.join(HERE, "smallnet_train.py"), "--out", out, "--width", str(w), "--hidden", str(h),
               "--lr", repr(lr), "--wd", repr(wd), "--bs", str(bs), "--focus", str(focus), "--threads", str(a.threads),
               "--quantiles", "0.01,0.02,0.05,0.1", "--seed", str(tr.number + 1)]
        if a.extra:
            cmd.append("--extra")
        if a.alpha_in:
            cmd.append("--alpha-in")
        with open(out + ".out", "w") as f:
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode:
                raise optuna.TrialPruned()
        r = subprocess.run([sys.executable, os.path.join(HERE, "smallnet_eval2.py"), out, "--json"],
                           capture_output=True, text=True)
        best = json.loads(r.stdout.strip().splitlines()[-1])
        skip = best["skip"] / 100
        v = sn_cost(w, h, a.extra) + (1 - skip) * BIG_US
        tr.set_user_attr("skip", best["skip"])
        tr.set_user_attr("tau", best["tau"])
        tr.set_user_attr("all", best["all"])
        print("trial %d W %d H %d lr %.2e wd %.1e bs %d focus %g -> skip %.1f%% (tau %g) cost %.3f us/call"
              % (tr.number, w, h, lr, wd, bs, focus, best["skip"], best["tau"], v), flush=True)
        with lock:
            if v < state["best"] * 0.995:
                state["best"], state["since"] = v, 0
            else:
                state["since"] += 1
            if state["since"] >= a.patience:
                print("SATURATED: no gain > 0.5%% in %d trials, best %.3f us/call" % (a.patience, state["best"]), flush=True)
                study.stop()
        return v

    study.optimize(objective, n_trials=a.trials, n_jobs=a.jobs)
    b = study.best_trial
    print("BEST trial %d value %.3f params %s attrs %s" % (b.number, b.value, b.params, b.user_attrs), flush=True)


if __name__ == "__main__":
    main()
