"""Phase B of the NNUE training plan: Population Based Training (Jaderberg 2017; AlphaZero-PBT: copy top, perturb
x0.8/1.2) on the FULL data, seeded from the Optuna top configs (data/nnue/nnue_hpo/top.json).

Each round every member trains `--interval` rows (contiguous 1M-row chunks, its own order), then all are scored on
a fixed 300k-row slice of the held-out val split. Exploit: the bottom member copies weights + optimizer + EMA of the
best one. Explore: the copy perturbs lr / wd x0.8 or x1.25, and nudges lookahead alpha and the EMA decay.
Stage 1 (base labels, all rows) runs until the best score has not improved by > 0.1% for --patience rounds, then the
population switches to stage 2 (KataGo teacher labels, Kata rows), continuing from the best weights (never restart).
Best-ever net per stage -> data/nnue/nnue_pbt/best_{stage}.pt. State saved every round -> resumable; STOP file
C:/gitProjects/logs/STOP ends the run cleanly after the current round.
"""
import argparse
import copy
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nnue as T  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
OUT = ROOT + "/data/nnue/nnue_pbt"
STOP = "C:/gitProjects/logs/STOP"


def log(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(OUT + "/pbt_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Member:
    def __init__(self, hp, width, seed):
        self.hp = dict(hp)
        self.model = T.NNUE(W=width)
        self.ema = T.NNUE(W=width)
        self.ema.load_state_dict(self.model.state_dict())
        self.opt = T.make_opt(self.model, argparse.Namespace(**self.args_ns()))
        self.rng = np.random.default_rng(seed)
        self.order, self.pos, self.steps = None, 0, 0
        self.score = float("inf")

    def args_ns(self):
        return dict(opt=self.hp["opt"], lr=self.hp["lr"], wd=self.hp["wd"], la_k=self.hp["la_k"],
                    la_alpha=self.hp["la_alpha"])

    def set_lr_wd(self):
        groups = self.opt.param_groups
        for g in groups:
            g["lr"], g["weight_decay"] = self.hp["lr"], self.hp["wd"]
        if isinstance(self.opt, T.Lookahead):
            self.opt.alpha = self.hp["la_alpha"]


def train_interval(m, tr, rows_per_round, chunk):
    nch = int(np.ceil(len(tr.rows) / chunk))
    done = 0
    while done < rows_per_round:
        if m.order is None or m.pos >= len(m.order):
            m.order, m.pos = m.rng.permutation(nch), 0
        ci = int(m.order[m.pos])
        m.pos += 1
        ch = tr.compact(tr.rows[ci * chunk:(ci + 1) * chunk])     # chunk in private RAM via file reads
        rows = ch.rows[m.rng.permutation(len(ch.rows))]
        bs = int(m.hp["bs"])
        for i in range(0, len(rows), bs):
            warm = min(1.0, (m.steps + 1) / 200)
            for g in m.opt.param_groups:
                g["lr"] = m.hp["lr"] * warm
            ix, off, bk, t, w = ch.batch(rows[i:i + bs])
            p = torch.sigmoid(m.model(ix, off, bk))
            loss = ((p - t).abs().pow(2.5) * w).sum() / w.sum()
            m.opt.zero_grad()
            loss.backward()
            m.opt.step()
            if m.hp["ema"] > 0:
                T.ema_update(m.ema, m.model, m.hp["ema"])
            m.steps += 1
        done += len(rows)


def score(m, va):
    vm = T.evaluate(m.model, va)
    ve = T.evaluate(m.ema, va) if m.hp["ema"] > 0 else None
    if ve and ve["mse_target"] < vm["mse_target"]:
        return ve["mse_target"], "ema", ve
    return vm["mse_target"], "live", vm


def save_state(pop, stage, rnd, best, stall):
    st = dict(stage=stage, round=rnd, best=best, stall=stall, members=[
        dict(hp=m.hp, model=m.model.state_dict(), ema=m.ema.state_dict(), opt=m.opt.state_dict(),
             order=None if m.order is None else m.order.tolist(), pos=m.pos, steps=m.steps, score=m.score,
             rng=m.rng.bit_generator.state) for m in pop])
    torch.save(st, OUT + "/state.pt.tmp")
    os.replace(OUT + "/state.pt.tmp", OUT + "/state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--interval", type=int, default=2_000_000)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max-rounds", type=int, default=100000)
    ap.add_argument("--val-rows", type=int, default=300_000)
    a = ap.parse_args()
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    top = json.load(open(ROOT + "/data/nnue/nnue_hpo/top.json"))
    width = int(top[0]["params"]["width"])
    stage, rnd, best, stall = "base", 0, float("inf"), 0
    pop = []
    if os.path.exists(OUT + "/state.pt"):
        st = torch.load(OUT + "/state.pt", weights_only=False)
        stage, rnd, best, stall = st["stage"], st["round"], st["best"], st["stall"]
        for i, ms in enumerate(st["members"]):
            m = Member(ms["hp"], width, i)
            m.model.load_state_dict(ms["model"])
            m.ema.load_state_dict(ms["ema"])
            m.opt.load_state_dict(ms["opt"])
            m.order = None if ms["order"] is None else np.array(ms["order"])
            m.pos, m.steps, m.score = ms["pos"], ms["steps"], ms["score"]
            m.rng.bit_generator.state = ms["rng"]
            pop.append(m)
        log("resumed stage %s round %d best %.6f" % (stage, rnd, best))
    else:
        seeds = [t["params"] for t in top if int(t["params"]["width"]) == width][:a.pop]
        while len(seeds) < a.pop:
            seeds.append(dict(seeds[len(seeds) % max(len(seeds), 1)]))
        pop = [Member(dict(hp, ema=hp["ema"] or 0.999), width, i) for i, hp in enumerate(seeds)]
        log("new population of %d, width %d, seeds %s" % (a.pop, width, json.dumps([m.hp for m in pop])))
    while rnd < a.max_rounds:
        tr, va_full = T.load_splits(stage, "")
        va = va_full if len(va_full.rows) <= a.val_rows else va_full.compact(
            np.sort(np.random.default_rng(5).choice(va_full.rows, a.val_rows, replace=False)))
        rnd += 1
        t0 = time.time()
        for m in pop:
            m.set_lr_wd()
            train_interval(m, tr, a.interval, a.chunk)
            m.score, which, metrics = score(m, va)
            m.which, m.metrics = which, metrics
        pop.sort(key=lambda m: m.score)
        lead = pop[0]
        improved = lead.score < best * (1 - 1e-3)
        if lead.score < best:
            net = lead.ema if lead.which == "ema" else lead.model
            torch.save(dict(model=net.state_dict(), hp=lead.hp, stage=stage, round=rnd, metrics=lead.metrics,
                            width=width), OUT + "/best_%s.pt" % stage)
            best = lead.score
        stall = 0 if improved else stall + 1
        log("stage %s round %d (%.0fs) scores %s | best %.6f stall %d | lead hp %s | kata_td %.5f" % (
            stage, rnd, time.time() - t0, ["%.6f/%s" % (m.score, m.which) for m in pop], best, stall,
            json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in lead.hp.items()}),
            lead.metrics.get("mse_kata_td", float("nan"))))
        # exploit + explore: bottom member copies the leader, then perturbs
        loser = pop[-1]
        loser.model.load_state_dict(lead.model.state_dict())
        loser.ema.load_state_dict(lead.ema.state_dict())
        loser.hp = dict(lead.hp)
        loser.opt = T.make_opt(loser.model, argparse.Namespace(**loser.args_ns()))
        loser.opt.load_state_dict(copy.deepcopy(lead.opt.state_dict()))
        r = loser.rng
        loser.hp["lr"] *= float(r.choice([0.8, 1.25]))
        loser.hp["wd"] *= float(r.choice([0.8, 1.25]))
        loser.hp["la_alpha"] = float(np.clip(loser.hp["la_alpha"] + r.choice([-0.05, 0.05]), 0.2, 0.9))
        loser.hp["ema"] = float(r.choice([0.999, 0.9995, 0.9999]))
        if stall >= a.patience and stage == "base":
            log("stage 1 plateau -> switching the population to stage 2 (KataGo teacher), from best weights")
            stage, best, stall = "teacher", float("inf"), 0
            for m in pop:
                m.order, m.pos = None, 0
        save_state(pop, stage, rnd, best, stall)
        if os.path.exists(STOP):
            log("STOP file -> ending after round %d" % rnd)
            break


if __name__ == "__main__":
    main()
