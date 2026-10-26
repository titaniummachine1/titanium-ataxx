"""Futility small net (D109): distil the big NNUE's eval of search CHILDREN (exp/fut-probe records, smallnet_pack.py)
into a tiny net the engine can run in ~0.1-0.2 us, so the futility step only calls the big net near the cut line.
Inputs (side to move relative, two perspectives like the big net, shared weights): own / enemy / blocker on each of
the 49 squares = 147 binary features -> W per perspective -> CReLU -> concat (stm first) -> 8 empties buckets:
L1 (2W -> H) CReLU -> 1, plus per-bucket linear skips of the parent static eval and (optionally) the child hand eval.
Output = cp / 400 (the big net's scale). Loss MSE on the big net's cp / 400 clipped to +-5.
LR: AdamW, val-driven: halve on a non-improving val (min rel gain --min-rel), stop after --patience halvings
(saturation); the best-val weights are kept. CPU only (Docker sandboxes / the PC; no Modal, user order 2026-10-26).
BOUND mode (--quantiles, user order 2026-10-26 "bound output not eval"): Q output heads, each trained with the pinball
loss at its quantile tau, so head q is a LOWER BOUND of the big net's child eval that holds with probability 1 - tau
(per position: tight where the position is easy, wide where it is not). The engine prunes without the big net when the
move is pruned even at that bound. --init warm-starts from a saturated eval net (all heads start at its output).
Usage: python training/smallnet_train.py --out data/nnue/smallnet/sn_w32.pt [--width 32 --hidden 16 --no-hand]
       python training/smallnet_train.py --out sn_q.pt --quantiles 0.002,0.005,0.01,0.02 --init sn_w32.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

D = os.environ.get("SN_DATA", "C:/gitProjects/titanium-ataxx/data/nnue/smallnet")
NB = 8
CLIP = 5.0


def bits(x):
    """(n,) uint64 -> (n, 49) float32 bit planes"""
    b = np.unpackbits(x.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")[:, :49]
    return b.astype(np.float32)


class Rows:
    def __init__(self, z, idx):
        self.own, self.opp, self.blk = z["own"][idx], z["opp"][idx], z["blk"][idx]
        self.nn = z["nn"][idx].astype(np.float32) / 400
        self.pst = z["pst"][idx].astype(np.float32) / 400
        self.hand = z["hand"][idx].astype(np.float32) / 400
        occ = self.own | self.opp | self.blk
        emp = 49 - np.bitwise_count(occ).astype(np.int64)
        self.bucket = np.minimum(emp * NB // 50, NB - 1)

    def __len__(self):
        return len(self.nn)

    def batch(self, r):
        o, e, k = bits(self.own[r]), bits(self.opp[r]), bits(self.blk[r])
        us = np.concatenate([o, e, k], 1)
        them = np.concatenate([e, o, k], 1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return (t(us), t(them), t(self.bucket[r]), t(self.pst[r]), t(self.hand[r]),
                t(np.clip(self.nn[r], -CLIP, CLIP)))


class SmallNet(nn.Module):
    def __init__(self, W=32, H=16, hand=True, Q=1):
        super().__init__()
        self.W, self.H, self.use_hand, self.Q = W, H, hand, Q
        self.ft = nn.Linear(147, W)
        self.l1 = nn.Linear(2 * W, H * NB)
        self.out = nn.Linear(H, NB * Q)
        self.skip = nn.Parameter(torch.zeros(2, NB))  # parent eval, hand eval (per bucket)
        with torch.no_grad():
            self.skip[0].fill_(-0.5)  # the child eval is roughly minus the parent's

    def forward(self, us, them, bk, pst, hand):
        a = torch.cat([self.ft(us), self.ft(them)], 1).clamp(0, 1)
        n = len(bk)
        ar = torch.arange(n)
        h = self.l1(a).view(n, NB, self.H)[ar, bk].clamp(0, 1)
        y = self.out(h).view(n, NB, self.Q)[ar, bk] + (self.skip[0, bk] * pst)[:, None]
        if self.use_hand:
            y = y + (self.skip[1, bk] * hand)[:, None]
        return y if self.Q > 1 else y[:, 0]


def evaluate(m, rows, bs=65536):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            b = rows.batch(np.arange(i, min(i + bs, len(rows))))
            out.append(m(*b[:5]).numpy())
    m.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--epoch-rows", type=int, default=8_000_000, help="rows per 'epoch' (val check)")
    ap.add_argument("--min-rel", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=4, help="LR halvings without val gain before stopping")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quantiles", default="", help="comma list of tau: bound mode (pinball loss per head)")
    ap.add_argument("--init", default="", help="warm start from an eval-mode checkpoint (same W/H/hand)")
    a = ap.parse_args()
    taus = [float(x) for x in a.quantiles.split(",")] if a.quantiles else []
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    z = dict(np.load(D + "/sn_train.npz"))
    n = len(z["nn"])
    perm = rng.permutation(n)
    nval = min(500_000, n // 50)
    tr, va = Rows(z, np.sort(perm[nval:])), Rows(z, np.sort(perm[:nval]))
    del z
    m = SmallNet(a.width, a.hidden, not a.no_hand, max(1, len(taus)))
    if a.init:
        src = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        st = m.state_dict()
        for k_, v_ in src.items():
            if k_.startswith("out."):
                # [NB, H] -> [NB * Q, H]: every head starts at the eval net's output
                st[k_] = v_.view(NB, 1, -1).expand(NB, m.Q, v_.shape[-1] if v_.dim() > 1 else 1).reshape(st[k_].shape)                     if v_.dim() > 1 else v_.view(NB, 1).expand(NB, m.Q).reshape(-1)
            else:
                st[k_] = v_
        m.load_state_dict(st)
    tq = torch.tensor(taus) if taus else None

    def lossf(y, t):
        if tq is None:
            return ((y - t) ** 2).mean()
        d = t[:, None] - y
        return torch.maximum(tq * d, (tq - 1) * d).mean()

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    lr, best, bad, ep = a.lr, float("inf"), 0, 0
    best_state = None
    log = open(a.out + ".log", "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say("args %s | train %d val %d" % (json.dumps(vars(a)), len(tr), len(va)))
    t0 = time.time()
    while ep < a.max_epochs:
        ep += 1
        idx = rng.choice(len(tr), size=a.epoch_rows, replace=False)
        tl, k = 0.0, 0
        for i in range(0, len(idx), a.bs):
            us, them, bk, pst, hand, t = tr.batch(np.sort(idx[i:i + a.bs]))
            y = m(us, them, bk, pst, hand)
            loss = lossf(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss)
            k += 1
        p = evaluate(m, va)
        tv = np.clip(va.nn, -CLIP, CLIP)
        if tq is None:
            v = float(((np.clip(p, -CLIP, CLIP) - tv) ** 2).mean())
            sd = float(np.std((p - va.nn)[np.abs(va.nn) < CLIP]) * 400)
            stat = "resid sd %.1f cp" % sd
        else:
            d = tv[:, None] - p
            ta = np.array(taus)
            v = float(np.maximum(ta * d, (ta - 1) * d).mean())
            stat = "below-bound rate " + " ".join("%.4f/%g" % (float((d[:, i] < 0).mean()), t_) for i, t_ in enumerate(taus))
        if v < best * (1 - a.min_rel):
            best, bad = v, 0
            best_state = {k_: v_.clone() for k_, v_ in m.state_dict().items()}
            torch.save({"model": best_state, "args": vars(a), "val": best}, a.out)
            tag = "best"
        else:
            bad += 1
            lr /= 2
            for g in opt.param_groups:
                g["lr"] = lr
            m.load_state_dict(best_state)
            tag = "no gain -> lr %.2e, back to best (%d/%d)" % (lr, bad, a.patience)
        say("ep %d %.0fs train %.5f val %.5f %s | %s" % (ep, time.time() - t0, tl / k, v, stat, tag))
        if bad >= a.patience:
            say("SATURATED: best val %.5f" % best)
            break


if __name__ == "__main__":
    main()
