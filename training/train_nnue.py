"""NNUE trainer (exp/nnue, user order 2026-10-24: "train it, then ablation to prune, then restart training").

Features: written once by the ENGINE's own code (`titanium-cli nndump`, crates/titanium/src/nnfeat.rs) into
data/nnue/nnfeat/{split}.cnt (u8, 2 per row: side to move, then opponent) and {split}.idx (u16 indices).
Rows = final_{split}_v3.npz rows (same order).

Net (Stockfish layout): shared feature transformer NUM_FEATS -> W (+bias), one accumulator per perspective,
CReLU [0,1], concat (side to move first) -> 8 phase buckets (by empties) of 2W -> L1 -> L2 -> 1 with CReLU,
plus a PSQT-style linear skip (NUM_FEATS -> 8 buckets, (us - them) / 2). Output y; win prob = sigmoid(y),
engine cp = 400 * y.
Loss: weighted mean |sigmoid(y) - target| ** 2.5 (Stockfish's power).
Stages:
  base    : all rows, target 0.75 * sigmoid(clip(titanium_cp, +-1500) / 400) + 0.25 * outcome_p, weight wsum
  teacher : KataGo rows only (teacher_kind == 1), target 0.75 * teacher_p3 + 0.25 * outcome_p, weight wsum
Optimizer: --opt ranger = RAdam + Lookahead (the nnue-pytorch choice; Lookahead slow weights favour flat basins),
or adam / adamw / radam. --ema keeps an exponential moving average of the weights; live and EMA weights are both
scored on val every epoch and the better one is kept. Population based training: training/pbt_nnue.py drives
run_interval() (explore/exploit over hyperparameters, weights copied from the best members).
Ablation: --drop G1,G2,G3,G5,G6 removes those index ranges at load time (no re-dump).
Validation on the separate held-out val split, reported vs: the stage target, KataGo TD (Kata rows), the game result.
"""
import argparse
import ctypes
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("TI_ROOT", "C:/gitProjects/titanium-ataxx")
FEAT = ROOT + "/data/nnue/nnfeat"
FINAL = ROOT + "/data/nnue/final"
# v4 = v3 with titanium_cp replaced by the 2026-10-25 relabel (big NNUE c4829350, 5000 nodes); TI_DATA_TAG=v3 gives the old labels
DATA_TAG = os.environ.get("TI_DATA_TAG", "v4")
# TI_DEVICE=cuda trains on a GPU (Modal, D80); data prep stays numpy on the CPU, batches are moved per step
DEV = torch.device(os.environ.get("TI_DEVICE", "cpu"))
# must match crates/titanium/src/nnfeat.rs
SQ_STATES = 262
G1_OFF = 0
G2_OFF = 49 * SQ_STATES
G2_NONE = G2_OFF + 49 * 49 * 2
G3_OFF = G2_NONE + 1
G5_OFF = G3_OFF + 49 * 2 * 81
G6_OFF = G5_OFF + 49 * 9
G6_PAR_OFF = G6_OFF + 2 * 8 * 10
NUM_FEATS = G6_PAR_OFF + 16
GROUPS = {"G1": (G1_OFF, G2_OFF), "G2": (G2_OFF, G3_OFF), "G3": (G3_OFF, G5_OFF), "G5": (G5_OFF, G6_OFF),
          "G6": (G6_OFF, NUM_FEATS)}
NB = 8


def free_gb():
    if os.name != "nt":
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
        return 99.0
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                    ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2 ** 30


class Split:
    """Feature indices + targets of one split, gathered per batch."""

    def __init__(self, split, stage, keep_mask):
        z = np.load("%s/final_%s_%s.npz" % (FINAL, split, DATA_TAG))
        self.cnt = np.fromfile("%s/%s.cnt" % (FEAT, split), dtype=np.uint8)
        self.idx = np.memmap("%s/%s.idx" % (FEAT, split), dtype=np.uint16, mode="r")
        n = len(self.cnt) // 2
        assert n == len(z["stm"]), "feature dump rows != dataset rows"
        self.off = np.zeros(2 * n + 1, np.int64)
        np.cumsum(self.cnt, out=self.off[1:])
        assert self.off[-1] == len(self.idx), "cnt/idx mismatch"
        occ = z["occ0"] | z["occ1"]
        emp = 49 - np.bitwise_count(occ).astype(np.int64)
        self.bucket = np.minimum(emp * NB // 50, NB - 1).astype(np.int64)
        out = z["outcome_p"].astype(np.float32)
        kind = z["teacher_kind"]
        self.kata = (z["kata_td_win"][:, 0].astype(np.float32) + 0.5 * z["kata_td_draw"][:, 0].astype(np.float32))
        self.is_kata = kind == 1
        self.outcome = out
        ti = 1 / (1 + np.exp(-np.clip(z["titanium_cp"].astype(np.float32), -1500, 1500) / 400))
        if stage == "base":
            self.target = (0.75 * ti + 0.25 * out).astype(np.float32)
            self.rows = np.arange(n)
        else:
            self.target = (0.75 * z["teacher_p3"].astype(np.float32) + 0.25 * out).astype(np.float32)
            self.rows = np.nonzero(self.is_kata)[0]
        self.w = z["wsum"].astype(np.float32)
        self.keep = keep_mask
        # stage-target part without the outcome mix, for the lambda schedule (target = lam * tpart + (1 - lam) * outcome)
        self.tpart = (ti if stage == "base" else z["teacher_p3"].astype(np.float32)).astype(np.float32)
        # S57 eval per row EXACTLY as the engine computes it (logit units), for the v2 skip term:
        # `titanium-cli s57dump --in data/nnue/nnfeat/<split> --out data/nnue/s57/<split>.f32 --tuple <S57 .tup>`.
        # (The Python tuple forward differs from the engine's by ~66 cp mean, docs/NET_ANALYSIS.md section 27.)
        s57p = "%s/../s57/%s.f32" % (FEAT, split)
        self.s57 = np.fromfile(s57p, dtype="<f4") if os.path.exists(s57p) else np.zeros(n, np.float32)
        assert len(self.s57) == n, "s57 sidecar rows != dataset rows"

    def compact(self, rows):
        """In-RAM copy restricted to `rows` (sorted), so random batches never page-fault the 13 GB index file.
        Use contiguous blocks: each block is one sequential read."""
        rows = np.sort(rows)
        ent = np.empty(2 * len(rows), np.int64)
        ent[0::2], ent[1::2] = 2 * rows, 2 * rows + 1
        st, ln = self.off[ent], self.cnt[ent].astype(np.int64)
        # read each maximal run of consecutive entries in one slice
        brk = np.nonzero(np.diff(ent) != 1)[0] + 1
        parts = []
        # plain file reads, NOT the memmap: mapped pages stay pinned in this process's working set (10+ GB after a
        # few PBT rounds, 2026-10-24); read() pages go to the OS standby cache, which Windows can always reclaim
        with open(self.idx.filename, "rb") as fh:
            for a_, b_ in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(ent)]])):
                lo, hi = int(st[a_]), int(st[b_ - 1] + ln[b_ - 1])
                fh.seek(2 * lo)
                parts.append(np.fromfile(fh, dtype=np.uint16, count=hi - lo))
        c = Split.__new__(Split)
        c.idx = np.concatenate(parts)
        c.cnt = ln.astype(np.uint8)
        c.off = np.zeros(len(ln) + 1, np.int64)
        np.cumsum(ln, out=c.off[1:])
        assert c.off[-1] == len(c.idx)
        for k in ("bucket", "target", "w", "kata", "is_kata", "outcome", "tpart", "s57"):
            setattr(c, k, getattr(self, k)[rows])
        c.keep = self.keep
        c.rows = np.arange(len(rows)) if len(self.rows) == len(self.target) else np.nonzero(np.isin(rows, self.rows))[0]
        return c

    def batch(self, rows):
        ent = np.empty(2 * len(rows), np.int64)
        ent[0::2], ent[1::2] = 2 * rows, 2 * rows + 1
        st, ln = self.off[ent], self.cnt[ent].astype(np.int64)
        tot = int(ln.sum())
        excl = np.concatenate([[0], np.cumsum(ln)[:-1]])
        pos = np.repeat(st - excl, ln) + np.arange(tot)
        ix = np.asarray(self.idx[pos], dtype=np.int64)
        if self.keep is not None:
            k = self.keep[ix]
            seg = np.repeat(np.arange(len(ln)), ln)
            ln = np.bincount(seg[k], minlength=len(ln)).astype(np.int64)
            ix = ix[k]
        offs = np.concatenate([[0], np.cumsum(ln)[:-1]])
        return (torch.from_numpy(ix), torch.from_numpy(offs), torch.from_numpy(self.bucket[rows]),
                torch.from_numpy(self.target[rows]), torch.from_numpy(self.w[rows]))

    def extra(self, rows):
        """lambda-schedule parts and the S57 skip input for the same rows as batch()"""
        return (torch.from_numpy(self.tpart[rows]), torch.from_numpy(self.outcome[rows]), torch.from_numpy(self.s57[rows]))


class NNUE2(nn.Module):
    """v2 (D65): feature transformer W per perspective (shared weights), CReLU, Stockfish pairwise product (W -> W/2
    per perspective), concat (side to move first) -> 8 phase buckets: L1 -> L2, output reads BOTH hidden layers (skip),
    + PSQT-style linear skip over the features + S57 tuple eval (logit units) times a per-bucket weight.
    Output layer and PSQT start at zero and the S57 weight at 1, so the untrained net IS S57 and training learns the
    residual (what S57 misses)."""

    def __init__(self, W=1024, L1=16, L2=32, pairwise=True):
        super().__init__()
        self.W, self.L1, self.L2, self.pairwise = W, L1, L2, pairwise
        self.ft = nn.EmbeddingBag(NUM_FEATS, W, mode="sum")
        nn.init.normal_(self.ft.weight, 0.0, 0.02)
        self.ftb = nn.Parameter(torch.full((W,), 0.3 if not pairwise else 0.5))
        self.psqt = nn.EmbeddingBag(NUM_FEATS, NB, mode="sum")
        nn.init.zeros_(self.psqt.weight)
        din = W if pairwise else 2 * W
        self.l1 = nn.Linear(din, L1 * NB)
        self.l2 = nn.Linear(L1, L2 * NB)
        self.out = nn.Linear(L1 + L2, NB)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.s57w = nn.Parameter(torch.ones(NB))

    def forward(self, ix, offs, bucket, s57=None):
        n = len(bucket)
        a = (self.ft(ix, offs) + self.ftb).clamp(0, 1)
        if self.pairwise:
            h = self.W // 2
            a = a[:, :h] * a[:, h:]
        x = torch.cat([a[0::2], a[1::2]], 1)
        ar = torch.arange(n)
        h1 = self.l1(x).view(n, NB, self.L1)[ar, bucket].clamp(0, 1)
        h2 = self.l2(h1).view(n, NB, self.L2)[ar, bucket].clamp(0, 1)
        o = self.out(torch.cat([h1, h2], 1))[ar, bucket]
        ps = self.psqt(ix, offs)
        y = o + 0.5 * (ps[0::2] - ps[1::2])[ar, bucket]
        if s57 is not None:
            y = y + self.s57w[bucket] * s57
        return y


class NNUE(nn.Module):
    def __init__(self, W=256, L1=16, L2=32):
        super().__init__()
        self.W, self.L1, self.L2 = W, L1, L2
        self.ft = nn.EmbeddingBag(NUM_FEATS, W, mode="sum")
        nn.init.normal_(self.ft.weight, 0.0, 0.02)
        self.ftb = nn.Parameter(torch.full((W,), 0.3))
        self.psqt = nn.EmbeddingBag(NUM_FEATS, NB, mode="sum")
        nn.init.zeros_(self.psqt.weight)
        self.l1 = nn.Linear(2 * W, L1 * NB)
        self.l2 = nn.Linear(L1, L2 * NB)
        self.out = nn.Linear(L2, NB)

    def forward(self, ix, offs, bucket, s57=None):
        n = len(bucket)
        a = self.ft(ix, offs) + self.ftb
        x = torch.cat([a[0::2], a[1::2]], 1).clamp(0, 1)
        ar = torch.arange(n)
        h = self.l1(x).view(n, NB, self.L1)[ar, bucket].clamp(0, 1)
        h = self.l2(h).view(n, NB, self.L2)[ar, bucket].clamp(0, 1)
        o = self.out(h)[ar, bucket]
        ps = self.psqt(ix, offs)
        return o + 0.5 * (ps[0::2] - ps[1::2])[ar, bucket]


class Lookahead:
    """Lookahead(k, alpha) over any base optimizer; slow weights are part of the state (resume-safe)."""

    def __init__(self, base, k=6, alpha=0.5):
        self.base, self.k, self.alpha, self.n = base, k, alpha, 0
        self.slow = [p.detach().clone() for g in base.param_groups for p in g["params"]]
        self.param_groups = base.param_groups

    def zero_grad(self):
        self.base.zero_grad()

    def step(self):
        self.base.step()
        self.n += 1
        if self.n % self.k == 0:
            with torch.no_grad():
                for s_, p in zip(self.slow, (p for g in self.base.param_groups for p in g["params"])):
                    s_.add_(p - s_, alpha=self.alpha)
                    p.copy_(s_)

    def state_dict(self):
        return dict(base=self.base.state_dict(), slow=self.slow, n=self.n)

    def load_state_dict(self, st):
        self.base.load_state_dict(st["base"])
        self.n = st["n"]
        for s_, v in zip(self.slow, st["slow"]):
            s_.copy_(v)


def make_opt(model, a):
    if a.opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    if a.opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    base = torch.optim.RAdam(model.parameters(), lr=a.lr, weight_decay=a.wd, decoupled_weight_decay=True)
    return base if a.opt == "radam" else Lookahead(base, k=a.la_k, alpha=a.la_alpha)


def ema_update(ema, model, decay):
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def evaluate(model, sp, bs=65536):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sp.rows), bs):
            r = sp.rows[i:i + bs]
            ix, off, bk, _, _ = sp.batch(r)
            preds.append(torch.sigmoid(model(ix.to(DEV), off.to(DEV), bk.to(DEV), sp.extra(r)[2].to(DEV))).cpu().numpy())
    model.train()
    p = np.concatenate(preds)
    r = sp.rows
    w = sp.w[r]
    res = dict(mse_target=float(np.average((p - sp.target[r]) ** 2, weights=w)),
               mse_outcome=float(np.average((p - sp.outcome[r]) ** 2, weights=w)))
    k = sp.is_kata[r]
    if k.any():
        res["mse_kata_td"] = float(np.average((p[k] - sp.kata[r][k]) ** 2, weights=w[k]))
    return res


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "teacher"], required=True)
    ap.add_argument("--init", default="", help="torch checkpoint to start from (stage 2 = stage 1 output)")
    ap.add_argument("--out", required=True, help="checkpoint prefix, e.g. data/nnue/nnue_base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--block", type=int, default=1024,
                    help="rows per block: each training chunk = chunk/block random blocks from the whole dataset, so a "
                         "chunk mixes sources, phases and games (the final_*.npz files are in source then game order; "
                         "Stockfish draws every batch from a 1M shuffle window over interleaved data). 0 = old "
                         "contiguous chunks (runs resumed from a pre-block resume file keep 0)")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--drop", default="", help="feature groups to remove, e.g. G5,G3")
    ap.add_argument("--rows", type=int, default=0, help="subset of training rows (ablation / smoke)")
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke test only: stop after N chunks")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=8, help="chunks between resume checkpoints")
    ap.add_argument("--opt", choices=["adam", "adamw", "radam", "ranger"], default="ranger")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--la-k", type=int, default=6)
    ap.add_argument("--la-alpha", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay per step (0 = off)")
    ap.add_argument("--warmup", type=int, default=200, help="linear LR warmup steps")
    ap.add_argument("--metric", default="mse_target", help="val metric that decides the best checkpoint")
    ap.add_argument("--patience", type=int, default=0, help="stop after N epochs without a min-rel gain (0 = off)")
    ap.add_argument("--min-rel", type=float, default=0.002, help="relative metric gain that counts as progress")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1", help="v2 = D65 (pairwise, hidden skips, S57 skip)")
    ap.add_argument("--no-pairwise", action="store_true", help="v2 without the pairwise product (ablation arm)")
    ap.add_argument("--lam-start", type=float, default=0.75, help="weight of the stage target vs outcome at step 0")
    ap.add_argument("--lam-end", type=float, default=0.75, help="... at the last step (linear schedule)")
    ap.add_argument("--auto-lr", action="store_true",
                    help="no fixed schedule: the LR is held per epoch and driven by val (backtracking line search over "
                         "epochs + SGDR warm restarts to look for another basin), see AutoLR")
    ap.add_argument("--auto-down", type=float, default=0.5, help="auto-lr: LR factor after an epoch without a min-rel gain")
    ap.add_argument("--auto-restarts", type=int, default=2,
                    help="auto-lr: stop after this many warm-restart cycles IN A ROW found no better basin")
    ap.add_argument("--hypergrad", action="store_true",
                    help="per-STEP learning-rate adaptation (hypergradient, Baydin et al. 2018): cosine of consecutive "
                         "gradients, tracked online as mean + variance (Kalman steady-state family, train_full.py "
                         "--step-kalman); consistently aligned -> LR scale up (still going downhill), consistently "
                         "opposed -> LR scale down (overshoot); multiplies the auto-lr / cosine LR")
    ap.add_argument("--hd-beta", type=float, default=0.02, help="hypergrad: log-LR change per step per unit cosine")
    ap.add_argument("--hd-ema", type=float, default=0.01, help="hypergrad: tracker weight (0.01 = ~100-step memory)")
    ap.add_argument("--hd-k", type=float, default=2.0, help="hypergrad: act only when |mean| > k sigma of the tracker")
    ap.add_argument("--hd-min", type=float, default=0.1, help="hypergrad: scale floor")
    ap.add_argument("--hd-max", type=float, default=10.0, help="hypergrad: scale ceiling")
    ap.add_argument("--force-restart", action="store_true",
                    help="auto-lr: end the current cycle NOW (basin saved) and warm-restart from the global best; for a "
                         "cycle that has clearly bottomed out. Applied once at startup, then the run continues normally.")
    ap.add_argument("--cos-epochs", type=int, default=0,
                    help="auto-lr switch-over from a cosine run: epochs of that cosine (its LR at the resume step is the "
                         "starting LR). 0 = start at --lr")
    return ap


_CACHE = {}


def load_splits(stage, drop):
    """Splits are cached so optuna / PBT members share one copy in RAM."""
    key = (stage, drop)
    if key not in _CACHE:
        _CACHE.clear()
        keep = None
        if drop:
            keep = np.ones(NUM_FEATS, bool)
            for g in drop.split(","):
                lo, hi = GROUPS[g.strip()]
                keep[lo:hi] = False
        _CACHE[key] = (Split("train", stage, keep), Split("val", stage, keep))
    return _CACHE[key]


class AutoLR:
    """Val-driven LR control over CYCLES (replaces the fixed cosine and the plateau STOP; no hand-set LR after the start).

    Ranger already adapts every parameter's step; this decides the global LR from what the val metric does.
    Inside a cycle (= one basin):
      * an epoch with a min-rel gain over the cycle's best keeps the LR and becomes the cycle's best state
        (weights + EMA + optimizer);
      * an epoch without one is a failed step: back to the cycle's best state and LR x auto-down
        (backtracking line search over epochs, train_full.py --backtrack).
    The first epoch after a warm restart is always accepted as the new cycle's start, so the restart really explores
    instead of snapping back to the old optimum.
    When the LR falls below lr-min the basin is exhausted: its best state is kept as a basin file, and a warm restart
    (SGDR) from the GLOBAL best state looks for a better basin at lr * 0.5**(fails+1). A cycle that beats the global
    best by min-rel resets the failure count; training stops after auto-restarts cycles IN A ROW found nothing.
    At the end, swa() averages the basins (SWA across restarts) and keeps the average only if val says it is better.
    Any val improvement anywhere is saved as the global best, so a worse basin can never replace a better one."""

    def __init__(self, a, lr0, best_v):
        self.a, self.lr = a, lr0
        self.best_v = best_v            # global best (any cycle)
        self.cyc_v = best_v             # this cycle's best
        self.cyc_entry = best_v         # global best when this cycle started
        self.cycle, self.fails, self.fresh = 0, 0, False
        self.basins = []                # [(val, file)]
        self.path = a.out + "_autolr_best.pt"
        self.cpath = a.out + "_autolr_cycle.pt"

    def state(self):
        return dict(lr=self.lr, best_v=self.best_v, cyc_v=self.cyc_v, cyc_entry=self.cyc_entry, cycle=self.cycle,
                    fails=self.fails, fresh=self.fresh, basins=self.basins)

    def load(self, st):
        for k, v in st.items():
            setattr(self, k, v)

    @staticmethod
    def _save(path, model, ema, opt):
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict() if ema else None, opt=opt.state_dict()),
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    @staticmethod
    def _load(path, model, ema, opt):
        if not os.path.exists(path):
            return False
        st = torch.load(path, weights_only=False, map_location=DEV)
        model.load_state_dict(st["model"])
        if ema is not None and st["ema"]:
            ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        return True

    def seed(self, model, ema, opt):
        self._save(self.path, model, ema, opt)
        self._save(self.cpath, model, ema, opt)

    def end_epoch(self, v, model, ema, opt):
        """Returns False when training should stop."""
        a = self.a
        if v < self.best_v:
            self.best_v = v
            self._save(self.path, model, ema, opt)
        if self.fresh or v < self.cyc_v * (1 - a.min_rel):
            why = "restart epoch accepted as the cycle start" if self.fresh else "gain"
            self.fresh, self.cyc_v = False, v
            self._save(self.cpath, model, ema, opt)
            print("auto-lr c%d: %s -> cycle best %.6f (global %.6f), keep lr %.2e"
                  % (self.cycle, why, v, self.best_v, self.lr), flush=True)
            return True
        if v < self.cyc_v:  # a gain below min-rel still becomes the cycle's best, then step down from it
            self.cyc_v = v
            self._save(self.cpath, model, ema, opt)
        back = self._load(self.cpath, model, ema, opt)
        self.lr *= a.auto_down
        msg = "auto-lr c%d: no %.2f%% gain (%.6f vs cycle best %.6f) -> %s, lr -> %.2e" % (
            self.cycle, 100 * a.min_rel, v, self.cyc_v, "back to the cycle's best state" if back else "no state", self.lr)
        if self.lr >= a.lr_min:
            print(msg, flush=True)
            return True
        # basin exhausted
        bf = "%s_basin%d.pt" % (a.out, self.cycle)
        if os.path.exists(self.cpath):
            import shutil
            shutil.copy(self.cpath, bf)
            self.basins.append((self.cyc_v, bf))
        found = self.best_v < self.cyc_entry * (1 - a.min_rel)
        self.fails = 0 if found else self.fails + 1
        msg += " | basin %d exhausted (best %.6f, %s)" % (
            self.cycle, self.cyc_v, "beat the global best" if found else "no better than the global best")
        return self._restart(model, ema, opt, msg)

    def _restart(self, model, ema, opt, msg):
        a = self.a
        if self.fails >= a.auto_restarts:
            print(msg + " | %d cycle(s) in a row found nothing: SATURATED" % self.fails, flush=True)
            return False
        self.cycle += 1
        self.lr = a.lr * 0.5 ** max(self.fails, 1)  # first jump = lr/2, smaller after each failed cycle in a row
        self._load(self.path, model, ema, opt)
        self._save(self.cpath, model, ema, opt)
        self.cyc_v = self.cyc_entry = self.best_v
        self.fresh = True
        print(msg + " | warm restart -> cycle %d from the global best %.6f at lr %.2e" % (
            self.cycle, self.best_v, self.lr), flush=True)
        return True

    def force_restart(self, model, ema, opt):
        """End the current cycle now: keep its best state as a basin, then warm-restart (same rules as end_epoch)."""
        a = self.a
        bf = "%s_basin%d.pt" % (a.out, self.cycle)
        if os.path.exists(self.cpath):
            import shutil
            shutil.copy(self.cpath, bf)
            self.basins.append((self.cyc_v, bf))
        found = self.best_v < self.cyc_entry * (1 - a.min_rel)
        self.fails = 0 if found else self.fails + 1
        msg = "auto-lr: FORCED end of cycle %d (best %.6f, %s)" % (
            self.cycle, self.cyc_v, "beat the global best" if found else "no better than the global best")
        return self._restart(model, ema, opt, msg)

    def swa(self, mk, evaluate, va):
        """Average the basins' weights (live and EMA separately) within 1% of the best; returns (val, file) or None."""
        a = self.a
        near = [f for v, f in self.basins if v <= self.best_v * 1.01 and os.path.exists(f)]
        if len(near) < 2:
            print("auto-lr swa: %d basin(s) near the best, nothing to average" % len(near), flush=True)
            return None
        out = None
        for key in ("model", "ema"):
            sts = [torch.load(f, weights_only=False, map_location=DEV)[key] for f in near]
            if any(x is None for x in sts):
                continue
            avg = {k: sum(x[k].float() for x in sts) / len(sts) for k in sts[0]}
            net = mk()
            net.load_state_dict(avg)
            m = evaluate(net, va)
            print("auto-lr swa(%s) over %d basins: %s %.6f vs best %.6f" % (key, len(near), a.metric, m[a.metric],
                                                                             self.best_v), flush=True)
            if m[a.metric] < self.best_v and (out is None or m[a.metric] < out[0]):
                f = "%s_swa_%s.pt" % (a.out, key)
                torch.save(dict(model=net.state_dict(), args=vars(a), swa=near, val=m), f)
                out = (m[a.metric], f)
        pl = self.plane(mk, evaluate, va)
        if pl is not None and (out is None or pl[0] < out[0]):
            out = pl
        return out

    def plane(self, mk, evaluate, va):
        """Map the 2D slice of weight space through the global best B and the two best other basins P, Q
        (mode-connectivity / loss-landscape scan): w(x, y) = B + x (P - B) + y (Q - B) on a triangle grid of step 0.25
        (x, y >= 0, x + y <= 1.5; corners already known). EMA weights. Keeps the best point if val says it beats B."""
        a = self.a
        if not os.path.exists(self.path):
            return None
        bst = torch.load(self.path, weights_only=False, map_location=DEV)
        key = "ema" if bst.get("ema") else "model"
        B = bst[key]
        def dist(x, y):
            return sum(float((x[k].float() - y[k].float()).norm() ** 2) for k in x) ** 0.5
        others = sorted([(v, f) for v, f in self.basins if os.path.exists(f)])
        pts = []
        for v, f in others:
            w = torch.load(f, weights_only=False, map_location=DEV)[key]
            # P, Q must span a real plane: each distinct from B and from each other (> 5% of the B-P distance)
            if dist(w, B) > 0 and all(dist(w, q) > 0.05 * max(dist(q, B), 1e-12) for _, q in pts):
                pts.append((v, w))
            if len(pts) == 2:
                break
        if len(pts) < 2:
            print("auto-lr plane: %d basin(s) distinct from the best, no plane to map" % len(pts), flush=True)
            return None
        P, Q = pts[0][1], pts[1][1]
        grid = [(x / 4, y / 4) for x in range(7) for y in range(7) if x + y <= 6 and (x, y) not in ((0, 0), (4, 0), (0, 4))]
        best, out, net = self.best_v, None, mk()
        rows = []
        for x, y in grid:
            net.load_state_dict({k: B[k].float() + x * (P[k].float() - B[k].float()) + y * (Q[k].float() - B[k].float())
                                 for k in B})
            m = evaluate(net, va)[a.metric]
            rows.append((x, y, m))
            if m < best:
                best = m
                f = "%s_plane.pt" % a.out
                torch.save(dict(model=net.state_dict(), args=vars(a), plane=(x, y), val=m), f)
                out = (m, f)
        print("auto-lr plane map (x toward basin P %.6f, y toward basin Q %.6f, B = global best %.6f):"
              % (pts[0][0], pts[1][0], self.best_v), flush=True)
        for x, y, m in rows:
            print("  x %.2f y %.2f  %s %.6f%s" % (x, y, a.metric, m, "  <- better than B" if m < self.best_v else ""),
                  flush=True)
        json.dump(dict(B=self.best_v, P=pts[0][0], Q=pts[1][0], grid=rows), open(a.out + "_plane.json", "w"), indent=1)
        return out


def run(a, trial=None):
    """Train; returns the best val metric. `trial` = optuna trial for per-epoch pruning."""
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    tr, va = load_splits(a.stage, a.drop)
    tr.rows_all = getattr(tr, "rows_all", tr.rows)
    tr.rows = tr.rows_all
    if a.rows:
        tr.rows = np.sort(rng.choice(tr.rows_all, min(a.rows, len(tr.rows_all)), replace=False))
    print("stage %s | train rows %d | val rows %d | drop [%s] | load %.0fs | free %.1f GB"
          % (a.stage, len(tr.rows), len(va.rows), a.drop, time.time() - t0, free_gb()), flush=True)
    mk0 = (lambda: NNUE2(W=a.width, pairwise=not a.no_pairwise)) if a.arch == "v2" else (lambda: NNUE(W=a.width))
    mk = lambda: mk0().to(DEV)
    print("device", DEV, flush=True)
    model = mk()
    opt = make_opt(model, a)
    ema = mk() if a.ema > 0 else None
    if a.arch == "v2":
        print("arch v2: W %d pairwise %s | lambda %.2f -> %.2f | s57 skip mean %.3f std %.3f (zeros = sidecar missing)"
              % (a.width, not a.no_pairwise, a.lam_start, a.lam_end, float(tr.s57.mean()), float(tr.s57.std())), flush=True)
    st = None
    ep0, done_chunks, hist = 1, set(), []
    resume = a.out + ".resume.pt"
    if os.path.exists(resume):
        st = torch.load(resume, weights_only=False, map_location=DEV)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ep0, done_chunks, hist = st["epoch"], set(st["done"]), st["hist"]
        rng = np.random.default_rng(st["rng_seed"])
        a.block = st.get("block", 0)  # chunk layout must match the one the done_chunks ids refer to
        print("resumed epoch %d with %d chunks done (block %d)" % (ep0, len(done_chunks), a.block), flush=True)
    elif a.init:
        model.load_state_dict(torch.load(a.init, weights_only=False, map_location=DEV)["model"])
        print("init from", a.init, flush=True)
    if ema is not None:
        ema.load_state_dict(model.state_dict())
        if st is not None and st.get("ema"):
            ema.load_state_dict(st["ema"])
    best = (float("inf"), None)
    if not hist:
        hist.append(dict(epoch=0, **evaluate(model, va)))
        print("epoch 0 val", hist[-1], flush=True)
    steps_per_epoch = math.ceil(len(tr.rows) / a.bs)
    total_steps = steps_per_epoch * a.epochs
    hd = None
    if a.hypergrad:
        hd = dict(scale=1.0, mean=0.0, var=1e-4, prev=None, ups=0, downs=0)
        if st is not None and st.get("hd"):
            hd.update(st["hd"])
            hd["prev"] = None
        print("hypergrad: per-step LR scale %.3f (beta %.3g, tracker %.3g, k %.1f, range %.2g..%.2g)"
              % (hd["scale"], a.hd_beta, a.hd_ema, a.hd_k, a.hd_min, a.hd_max), flush=True)
    hds = lambda: {k: v for k, v in hd.items() if k != "prev"} if hd is not None else None
    auto = None
    if a.auto_lr:
        best_v = min([min(h[a.metric], (h.get("ema") or {}).get(a.metric, float("inf")))
                      for h in hist if h.get("epoch", 0) >= 1] or [float("inf")])
        lr0 = a.lr
        if a.cos_epochs:  # continue a cosine run at the LR it had reached
            s0 = (ep0 - 1) * steps_per_epoch + len(done_chunks) * a.chunk // a.bs
            lr0 = a.lr_min + 0.5 * (a.lr - a.lr_min) * (
                1 + math.cos(math.pi * min(s0 / (steps_per_epoch * a.cos_epochs), 1.0)))
        auto = AutoLR(a, lr0, best_v)
        if st is not None and st.get("autolr"):
            auto.load(st["autolr"])
        elif best_v < float("inf") and not os.path.exists(auto.path):
            # switch-over: seed the best state with the best epoch file so far (optimizer state = the current one)
            for h in hist:
                for tag, m in (("", h), ("_ema", h.get("ema"))):
                    f = "%s_ep%d%s.pt" % (a.out, h.get("epoch", 0), tag)
                    if h.get("epoch", 0) >= 1 and m and m.get(a.metric) == best_v and os.path.exists(f):
                        bm = mk()
                        bm.load_state_dict(torch.load(f, weights_only=False, map_location=DEV)["model"])
                        auto.seed(bm, bm if ema is not None else None, opt)
        elif st is None and a.init and not os.path.exists(auto.path):
            # warm start: the init net (epoch 0) is the best state to beat, so an epoch that makes it worse is rolled
            # back (LR halved) instead of being accepted as the first cycle best
            auto.best_v = auto.cyc_v = auto.cyc_entry = hist[0][a.metric]
            auto.seed(model, ema, opt)
            print("auto-lr: seeded with the init net, %s %.6f" % (a.metric, auto.best_v), flush=True)
        if a.force_restart:
            if hd is not None:
                hd.update(scale=1.0, mean=0.0, var=1e-4, prev=None)  # same rule as an automatic warm restart
            if not auto.force_restart(model, ema, opt):
                print("auto-lr: forced restart -> SATURATED, nothing left to explore", flush=True)
                return None
            done_chunks = set()  # the restarted cycle begins with a fresh epoch
        print("auto-lr: cycle %d lr %.2e best %.6f cycle best %.6f fails %d/%d basins %d" % (
            auto.cycle, auto.lr, auto.best_v, auto.cyc_v, auto.fails, a.auto_restarts, len(auto.basins)), flush=True)
    for ep in range(ep0, a.epochs + 1):
        ep_seed = a.seed * 1000 + ep
        nchunks = math.ceil(len(tr.rows) / a.chunk)
        if a.block > 0:
            nblk = math.ceil(len(tr.rows) / a.block)
            bperm = np.random.default_rng(ep_seed + 10 ** 6).permutation(nblk)
            per = max(1, a.chunk // a.block)
            groups = [np.sort(bperm[j:j + per]) for j in range(0, nblk, per)]
            nchunks = len(groups)
        order = np.random.default_rng(ep_seed).permutation(nchunks)
        tl, tw, te = 0.0, 0.0, time.time()
        for ci_n, ci in enumerate(order):
            if int(ci) in done_chunks:
                continue
            if free_gb() < a.min_free_gb:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, done=sorted(done_chunks),
                                hist=hist, rng_seed=ep_seed, ema=ema.state_dict() if ema else None,
                                autolr=auto.state() if auto else None, hd=hds(), block=a.block), resume)
                print("RAM GUARD: %.1f GB free < %.1f -> saved resume, exit" % (free_gb(), a.min_free_gb), flush=True)
                return
            if a.block > 0:  # sorted blocks of the sorted tr.rows -> rows stay sorted (compact needs that)
                rows = np.concatenate([tr.rows[b * a.block:(b + 1) * a.block] for b in groups[ci]])
            else:
                rows = tr.rows[ci * a.chunk:(ci + 1) * a.chunk]
            perm = np.random.default_rng(ep_seed + int(ci)).permutation(len(rows))
            # chunk -> RAM with plain file reads (compact), never batches through the 13 GB memmap: mapped pages pin
            # in the working set (law 21, MISTAKES 19). tr.rows is sorted, so sub row j == rows[j] and sub.batch(perm)
            # yields exactly the batches tr.batch(rows[perm]) did (resume-consistent).
            sub = tr.compact(rows)
            for i in range(0, len(rows), a.bs):
                step = (ep - 1) * steps_per_epoch + (ci_n * a.chunk + i) // a.bs
                if auto is not None:
                    lr = auto.lr
                else:
                    lr = a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                lr *= min(1.0, (step + 1) / max(a.warmup, 1))
                lr_eff = lr * (hd["scale"] if hd is not None else 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_eff
                ix, off, bk, t, w = (x.to(DEV, non_blocking=True) for x in sub.batch(perm[i:i + a.bs]))
                tp, oc, s57 = (x.to(DEV, non_blocking=True) for x in sub.extra(perm[i:i + a.bs]))
                if a.lam_start != 0.75 or a.lam_end != 0.75:
                    frac = min(step / max(total_steps, 1), 1.0)
                    lam = a.lam_start + (a.lam_end - a.lam_start) * frac
                    t = lam * tp + (1 - lam) * oc
                p = torch.sigmoid(model(ix, off, bk, s57))
                loss = ((p - t).abs().pow(2.5) * w).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                if hd is not None:
                    # Baydin et al. 2018 for ANY update rule: d loss / d lr = g_t . d theta_t / d lr = -g_t . u_(t-1) / lr,
                    # with u = the step the optimizer actually took. hd["prev"] = -(theta_t - theta_(t-1)) (the descent
                    # direction taken), so cos(g_t, prev) > 0 = the gradient still agrees with the last step -> LR up,
                    # < 0 = overshoot -> LR down. (Before D77 prev was g_(t-1): right for plain SGD only; under Ranger's
                    # Adam preconditioning it stayed positive and the LR ran away.)
                    params = list(model.parameters())
                    gf = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1) for q in params])
                    if hd["prev"] is not None:
                        c = float(torch.dot(gf, hd["prev"]) / (gf.norm() * hd["prev"].norm() + 1e-12))
                        k = a.hd_ema
                        d = c - hd["mean"]
                        hd["mean"] += k * d
                        hd["var"] = (1 - k) * hd["var"] + k * d * d
                        sig = math.sqrt(max(hd["var"], 1e-12) * k / (2 - k))  # std of the tracked mean
                        if abs(hd["mean"]) > a.hd_k * sig:
                            hd["scale"] = min(max(hd["scale"] * math.exp(a.hd_beta * hd["mean"]), a.hd_min), a.hd_max)
                            hd["ups" if hd["mean"] > 0 else "downs"] += 1
                            lr_eff = lr * hd["scale"]
                            for g in opt.param_groups:
                                g["lr"] = lr_eff
                    theta0 = torch.cat([q.detach().reshape(-1) for q in params])
                opt.step()
                if hd is not None:
                    synced = isinstance(opt, Lookahead) and opt.n % opt.k == 0  # this step also jumped to the slow weights
                    hd["prev"] = None if synced else theta0 - torch.cat([q.detach().reshape(-1) for q in params])
                if ema is not None:
                    ema_update(ema, model, a.ema)
                tl += float(loss) * float(w.sum())
                tw += float(w.sum())
            del sub
            done_chunks.add(int(ci))
            print("  ep %d chunk %d/%d loss %.6f lr %.2e %.0fs free %.1fGB%s" % (
                ep, len(done_chunks), len(order), tl / max(tw, 1e-9), lr_eff, time.time() - te, free_gb(),
                " | hd scale %.3f cos %+.4f up/down %d/%d" % (hd["scale"], hd["mean"], hd["ups"], hd["downs"])
                if hd is not None else ""), flush=True)
            if len(done_chunks) % a.save_every == 0:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, done=sorted(done_chunks),
                                hist=hist, rng_seed=ep_seed, ema=ema.state_dict() if ema else None,
                                autolr=auto.state() if auto else None, hd=hds(), block=a.block), resume + ".tmp")
                os.replace(resume + ".tmp", resume)
            if a.max_chunks and len(done_chunks) >= a.max_chunks:
                print("SMOKE: stopping after %d chunks (not a real run)" % len(done_chunks), flush=True)
                print("val", evaluate(model, va), "| ema", evaluate(ema, va) if ema else "-", flush=True)
                return None
        vm = evaluate(model, va)
        ve = evaluate(ema, va) if ema is not None else None
        hist.append(dict(epoch=ep, train_loss=tl / max(tw, 1e-9), secs=round(time.time() - te), **vm, ema=ve))
        print("epoch %d val %s" % (ep, json.dumps(hist[-1])), flush=True)
        for tag, net, m in (("", model, vm), ("_ema", ema, ve)):
            if net is None:
                continue
            torch.save(dict(model=net.state_dict(), hist=hist, args=vars(a)), "%s_ep%d%s.pt" % (a.out, ep, tag))
            if m[a.metric] < best[0]:
                best = (m[a.metric], "%s_ep%d%s.pt" % (a.out, ep, tag))
        if trial is not None:
            trial.report(best[0], ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        done_chunks = set()
        go_on = True
        if auto is not None:
            cyc_before = auto.cycle
            go_on = auto.end_epoch(min(vm[a.metric], ve[a.metric] if ve else float("inf")), model, ema, opt)
            if hd is not None and auto.cycle != cyc_before:
                # warm restart (SGDR): the restart LR must not be shrunk by the hypergrad scale the old basin ended with
                # (greedy per-step LR adaptation drifts down = short-horizon bias, Wu et al. 2018); backtracks keep it
                hd.update(scale=1.0, mean=0.0, var=1e-4, prev=None)
                print("hypergrad: scale reset to 1.0 for the warm restart (cycle %d)" % auto.cycle, flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep + 1, done=[], hist=hist,
                        rng_seed=a.seed * 1000 + ep + 1, ema=ema.state_dict() if ema else None,
                        autolr=auto.state() if auto else None, hd=hds(), block=a.block), resume + ".tmp")
        os.replace(resume + ".tmp", resume)
        if not go_on:
            break
        if a.patience and auto is None:
            # plateau stop (saturation): epochs since the last relative improvement > min_rel on the metric
            best_v, stall = float("inf"), 0
            for h in hist:
                if h.get("epoch", 0) < 1:
                    continue
                v = min(h[a.metric], (h.get("ema") or {}).get(a.metric, float("inf")))
                if v < best_v * (1 - a.min_rel):
                    best_v, stall = v, 0
                else:
                    stall += 1
            print("plateau check: best %s %.6f, %d epoch(s) without a %.2f%% gain (patience %d)"
                  % (a.metric, best_v, stall, 100 * a.min_rel, a.patience), flush=True)
            if stall >= a.patience:
                print("PLATEAU: saturated, stopping after epoch %d" % ep, flush=True)
                break
    import shutil
    if auto is not None:
        sw = auto.swa(mk, evaluate, va)
        if sw is not None:
            best = min(best, sw)
    # best over ALL epochs in hist (also those trained before a resume), live or EMA
    for h in hist:
        if h.get("epoch", 0) < 1:
            continue
        for tag, m in (("", h), ("_ema", h.get("ema"))):
            f = "%s_ep%d%s.pt" % (a.out, h["epoch"], tag)
            if m and m.get(a.metric, float("inf")) < best[0] and os.path.exists(f):
                best = (m[a.metric], f)
    if best[1]:
        shutil.copy(best[1], a.out + ".pt")
    with open(a.out + "_hist.json", "w") as f:
        json.dump(dict(hist=hist, best=best, autolr=auto.state() if auto is not None else None), f, indent=1)
    if os.path.exists(resume):
        os.remove(resume)
    print("done -> %s (best %s = %.6f from %s)" % (a.out + ".pt", a.metric, best[0], best[1]), flush=True)
    return best[0]


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
