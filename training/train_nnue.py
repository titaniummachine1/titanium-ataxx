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

ROOT = "C:/gitProjects/titanium-ataxx"
FEAT = ROOT + "/data/nnue/nnfeat"
FINAL = ROOT + "/data/nnue/final"
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
        z = np.load("%s/final_%s_v3.npz" % (FINAL, split))
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
        for k in ("bucket", "target", "w", "kata", "is_kata", "outcome"):
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

    def forward(self, ix, offs, bucket):
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
            preds.append(torch.sigmoid(model(ix, off, bk)).numpy())
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
    model = NNUE(W=a.width)
    opt = make_opt(model, a)
    ema = NNUE(W=a.width) if a.ema > 0 else None
    st = None
    ep0, done_chunks, hist = 1, set(), []
    resume = a.out + ".resume.pt"
    if os.path.exists(resume):
        st = torch.load(resume, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ep0, done_chunks, hist = st["epoch"], set(st["done"]), st["hist"]
        rng = np.random.default_rng(st["rng_seed"])
        print("resumed epoch %d with %d chunks done" % (ep0, len(done_chunks)), flush=True)
    elif a.init:
        model.load_state_dict(torch.load(a.init, weights_only=False)["model"])
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
    for ep in range(ep0, a.epochs + 1):
        ep_seed = a.seed * 1000 + ep
        order = np.random.default_rng(ep_seed).permutation(math.ceil(len(tr.rows) / a.chunk))
        tl, tw, te = 0.0, 0.0, time.time()
        for ci_n, ci in enumerate(order):
            if int(ci) in done_chunks:
                continue
            if free_gb() < a.min_free_gb:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, done=sorted(done_chunks),
                                hist=hist, rng_seed=ep_seed, ema=ema.state_dict() if ema else None), resume)
                print("RAM GUARD: %.1f GB free < %.1f -> saved resume, exit" % (free_gb(), a.min_free_gb), flush=True)
                return
            rows = tr.rows[ci * a.chunk:(ci + 1) * a.chunk]
            rows = rows[np.random.default_rng(ep_seed + int(ci)).permutation(len(rows))]
            for i in range(0, len(rows), a.bs):
                step = (ep - 1) * steps_per_epoch + (ci_n * a.chunk + i) // a.bs
                lr = a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
                lr *= min(1.0, (step + 1) / max(a.warmup, 1))
                for g in opt.param_groups:
                    g["lr"] = lr
                ix, off, bk, t, w = tr.batch(rows[i:i + a.bs])
                p = torch.sigmoid(model(ix, off, bk))
                loss = ((p - t).abs().pow(2.5) * w).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if ema is not None:
                    ema_update(ema, model, a.ema)
                tl += float(loss) * float(w.sum())
                tw += float(w.sum())
            done_chunks.add(int(ci))
            print("  ep %d chunk %d/%d loss %.6f lr %.2e %.0fs free %.1fGB" % (
                ep, len(done_chunks), len(order), tl / max(tw, 1e-9), lr, time.time() - te, free_gb()), flush=True)
            if len(done_chunks) % a.save_every == 0:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, done=sorted(done_chunks),
                                hist=hist, rng_seed=ep_seed, ema=ema.state_dict() if ema else None), resume + ".tmp")
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
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep + 1, done=[], hist=hist,
                        rng_seed=a.seed * 1000 + ep + 1, ema=ema.state_dict() if ema else None), resume + ".tmp")
        os.replace(resume + ".tmp", resume)
    import shutil
    if best[1]:
        shutil.copy(best[1], a.out + ".pt")
    with open(a.out + "_hist.json", "w") as f:
        json.dump(dict(hist=hist, best=best), f, indent=1)
    if os.path.exists(resume):
        os.remove(resume)
    print("done -> %s (best %s = %.6f from %s)" % (a.out + ".pt", a.metric, best[0], best[1]), flush=True)
    return best[0]


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
