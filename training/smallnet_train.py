"""Futility small net (D109): distil the big NNUE's eval of search CHILDREN (exp/fut-probe records, smallnet_pack.py)
into a tiny net the engine can run in ~0.1-0.2 us, so the futility step only calls the big net near the cut line.
Inputs (side to move relative, two perspectives like the big net, shared weights): own / enemy / blocker on each of
the 49 squares = 147 binary features -> W per perspective -> CReLU -> concat (stm first) -> 8 empties buckets:
L1 (2W -> H) CReLU -> 1, plus per-bucket linear skips of the parent static eval and (optionally) the child hand eval.
Output = cp / 400 (the big net's scale). Loss MSE on the big net's cp / 400 clipped to +-5.
LR: AdamW, val-driven: halve on a non-improving val (min rel gain --min-rel), stop after --patience halvings
(saturation); the best-val weights are kept. CPU only (Docker sandboxes / the PC; no Modal, user order 2026-10-26).
Usage: python training/smallnet_train.py --out data/nnue/smallnet/sn_w32.pt [--width 32 --hidden 16 --no-hand]
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
    def __init__(self, W=32, H=16, hand=True):
        super().__init__()
        self.W, self.H, self.use_hand = W, H, hand
        self.ft = nn.Linear(147, W)
        self.l1 = nn.Linear(2 * W, H * NB)
        self.out = nn.Linear(H, NB)
        self.skip = nn.Parameter(torch.zeros(2, NB))  # parent eval, hand eval (per bucket)
        with torch.no_grad():
            self.skip[0].fill_(-0.5)  # the child eval is roughly minus the parent's

    def forward(self, us, them, bk, pst, hand):
        a = torch.cat([self.ft(us), self.ft(them)], 1).clamp(0, 1)
        n = len(bk)
        ar = torch.arange(n)
        h = self.l1(a).view(n, NB, self.H)[ar, bk].clamp(0, 1)
        y = self.out(h)[ar, bk] + self.skip[0, bk] * pst
        if self.use_hand:
            y = y + self.skip[1, bk] * hand
        return y


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
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    z = dict(np.load(D + "/sn_train.npz"))
    n = len(z["nn"])
    perm = rng.permutation(n)
    nval = min(500_000, n // 50)
    tr, va = Rows(z, np.sort(perm[nval:])), Rows(z, np.sort(perm[:nval]))
    del z
    m = SmallNet(a.width, a.hidden, not a.no_hand)
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
            loss = ((y - t) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss)
            k += 1
        p = evaluate(m, va)
        v = float(((np.clip(p, -CLIP, CLIP) - np.clip(va.nn, -CLIP, CLIP)) ** 2).mean())
        sd = float(np.std((p - va.nn)[np.abs(va.nn) < CLIP]) * 400)
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
        say("ep %d %.0fs train %.5f val %.5f resid sd %.1f cp | %s" % (ep, time.time() - t0, tl / k, v, sd, tag))
        if bad >= a.patience:
            say("SATURATED: best val %.5f" % best)
            break


if __name__ == "__main__":
    main()
