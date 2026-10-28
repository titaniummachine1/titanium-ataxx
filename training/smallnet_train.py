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
v2 options (D114, user order 2026-10-26 "optimise it until saturated before the next relabel"):
  --extra: 10 cheap tactical scalars of the child (clone / jump-only targets of both sides; best capture and the number
           of squares capturing >= 2 / >= 3 for both sides), fed into L1 and a per-bucket linear skip;
  --alpha-in: the futility call's alpha (cp / 400) as one more scalar (search knowledge of the position's value);
  --focus S: weight each row by 0.1 + exp(-|t - T| / S), T = (57 * new_depth - alpha) / 400 = the prune threshold of
           that call (only errors near the cut line change a pruning decision).
  Extras are precomputed once: python training/smallnet_train.py --make-extras --out x (writes sn_{train,eval}_ex.npy).
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
NEX = 10
FULL = np.uint64((1 << 49) - 1)


def _file(f):
    return np.uint64(sum(1 << (r * 7 + f) for r in range(7)))


FA, FB, FF, FG = _file(0), _file(1), _file(5), _file(6)


def _sr(x, k):
    return x >> np.uint64(k)


def _sl(x, k):
    return x << np.uint64(k)


def dil1(b):
    """crate::board::dist_union(b, 1): the 8 neighbours"""
    na, ng = b & ~FA, b & ~FG
    return (_sr(b, 7) | _sl(b, 7) | _sr(na, 1) | _sl(ng, 1) | _sr(na, 8) | _sr(ng, 6) | _sl(na, 6) | _sl(ng, 8)) & FULL


def jump(b):
    """crate::board::jump_union: the 16 squares at Chebyshev distance 2"""
    nab, nfg = b & ~(FA | FB), b & ~(FF | FG)
    up, dn, lf, rt = _sr(b, 14), _sl(b, 14) & FULL, _sr(nab, 2), _sl(nfg, 2)
    ul, ur, dl, dr = _sr(nab, 16), _sr(nfg, 12), _sl(nab, 12), _sl(nfg, 16)
    v2, h2 = (up | dn) & FULL, lf | rt
    n1 = _sr(v2 & ~FA, 1) | _sl(v2 & ~FG, 1)
    n2 = _sr(h2, 7) | (_sl(h2, 7) & FULL)
    return (up | dn | lf | rt | ul | ur | dl | dr | n1 | n2) & FULL


ADJ = np.zeros((49, 49), np.float32)
for _q in range(49):
    _r, _c = divmod(_q, 7)
    for _dr in (-1, 0, 1):
        for _dc in (-1, 0, 1):
            if (_dr or _dc) and 0 <= _r + _dr < 7 and 0 <= _c + _dc < 7:
                ADJ[(_r + _dr) * 7 + _c + _dc, _q] = 1


def pc(x):
    return np.bitwise_count(x).astype(np.float32)


def extras(own, opp, blk):
    """(n,) uint64 x 3 -> (n, NEX) float32, child side to move = own. Same arithmetic as the engine port."""
    emp = ~(own | opp | blk) & FULL
    out = np.empty((len(own), NEX), np.float32)
    for k, (a, b) in enumerate(((own, opp), (opp, own))):
        c1 = dil1(a) & emp
        j = jump(a) & emp & ~c1
        cnt = bits(b) @ ADJ  # stones of the other side around each square
        g = cnt * bits(c1 | j)
        out[:, 5 * k + 0] = pc(c1) / 16
        out[:, 5 * k + 1] = pc(j) / 16
        out[:, 5 * k + 2] = g.max(1) / 8
        out[:, 5 * k + 3] = (g >= 2).sum(1) / 8
        out[:, 5 * k + 4] = (g >= 3).sum(1) / 8
    return out


def make_extras():
    for name in ("train", "eval"):
        z = np.load(D + "/sn_%s.npz" % name)
        own, opp, blk = z["own"], z["opp"], z["blk"]
        ex = np.empty((len(own), NEX), np.float16)
        for i in range(0, len(own), 1 << 20):
            ex[i:i + (1 << 20)] = extras(own[i:i + (1 << 20)], opp[i:i + (1 << 20)], blk[i:i + (1 << 20)])
        np.save(D + "/sn_%s_ex.npy" % name, ex)
        print("extras", name, ex.shape, flush=True)


def bits(x):
    """(n,) uint64 -> (n, 49) float32 bit planes"""
    b = np.unpackbits(x.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")[:, :49]
    return b.astype(np.float32)


class Rows:
    def __init__(self, z, idx, ex=None, alpha_in=False, focus=0.0):
        self.own, self.opp, self.blk = z["own"][idx], z["opp"][idx], z["blk"][idx]
        self.nn = z["nn"][idx].astype(np.float32) / 400
        self.pst = z["pst"][idx].astype(np.float32) / 400
        self.hand = z["hand"][idx].astype(np.float32) / 400
        al = z["alpha"][idx].astype(np.float32)
        self.thr = (57 * z["nd0"][idx].astype(np.float32) - al) / 400
        cols = []
        if ex is not None:
            cols.append(np.asarray(ex[idx], np.float32))
        if alpha_in:
            cols.append(np.clip(al / 400, -CLIP, CLIP)[:, None])
        self.ex = np.concatenate(cols, 1) if cols else np.zeros((len(idx), 0), np.float32)
        self.focus = focus
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
        tg = np.clip(self.nn[r], -CLIP, CLIP)
        w = 0.1 + np.exp(-np.abs(self.nn[r] - self.thr[r]) / self.focus) if self.focus > 0 else np.ones_like(tg)
        return (t(us), t(them), t(self.bucket[r]), t(self.pst[r]), t(self.hand[r]), t(self.ex[r]),
                t(tg), t(w.astype(np.float32)))


class SmallNet(nn.Module):
    def __init__(self, W=32, H=16, hand=True, Q=1, K=0):
        super().__init__()
        self.W, self.H, self.use_hand, self.Q, self.K = W, H, hand, Q, K
        self.ft = nn.Linear(147, W)
        self.l1 = nn.Linear(2 * W + K, H * NB)
        if K:
            self.exskip = nn.Parameter(torch.zeros(NB, K))
        self.out = nn.Linear(H, NB * Q)
        self.skip = nn.Parameter(torch.zeros(2, NB))  # parent eval, hand eval (per bucket)
        with torch.no_grad():
            self.skip[0].fill_(-0.5)  # the child eval is roughly minus the parent's

    def forward(self, us, them, bk, pst, hand, ex=None):
        a = torch.cat([self.ft(us), self.ft(them)], 1).clamp(0, 1)
        if self.K:
            a = torch.cat([a, ex], 1)
        n = len(bk)
        ar = torch.arange(n)
        h = self.l1(a).view(n, NB, self.H)[ar, bk].clamp(0, 1)
        y = self.out(h).view(n, NB, self.Q)[ar, bk] + (self.skip[0, bk] * pst)[:, None]
        if self.K:
            y = y + (self.exskip[bk] * ex).sum(1)[:, None]
        if self.use_hand:
            y = y + (self.skip[1, bk] * hand)[:, None]
        return y if self.Q > 1 else y[:, 0]


def evaluate(m, rows, bs=65536):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            b = rows.batch(np.arange(i, min(i + bs, len(rows))))
            out.append(m(*b[:6]).numpy())
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
    ap.add_argument("--extra", action="store_true", help="10 tactical scalars (precomputed by --make-extras)")
    ap.add_argument("--alpha-in", action="store_true", help="the call's alpha as an input")
    ap.add_argument("--focus", type=float, default=0.0, help="weight rows near the prune threshold (cp/400 scale)")
    ap.add_argument("--data-name", default="sn_train", help="npz dataset name inside SN_DATA")
    ap.add_argument("--make-extras", action="store_true")
    a = ap.parse_args()
    if a.make_extras:
        make_extras()
        return
    taus = [float(x) for x in a.quantiles.split(",")] if a.quantiles else []
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    z = dict(np.load(D + f"/{a.data_name}.npz"))
    ex = np.load(D + "/sn_train_ex.npy", mmap_mode="r") if a.extra else None
    n = len(z["nn"])
    perm = rng.permutation(n)
    nval = min(500_000, n // 50)
    kw = dict(ex=ex, alpha_in=a.alpha_in, focus=a.focus)
    tr, va = Rows(z, np.sort(perm[nval:]), **kw), Rows(z, np.sort(perm[:nval]), **kw)
    del z, ex
    K = tr.ex.shape[1]
    m = SmallNet(a.width, a.hidden, not a.no_hand, max(1, len(taus)), K)
    if a.init:
        src = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        st = m.state_dict()
        for k_, v_ in src.items():
            if k_.startswith("out."):
                # [NB, H] -> [NB * Q, H]: every head starts at the eval net's output
                st[k_] = v_.view(NB, 1, -1).expand(NB, m.Q, v_.shape[-1] if v_.dim() > 1 else 1).reshape(st[k_].shape)                     if v_.dim() > 1 else v_.view(NB, 1).expand(NB, m.Q).reshape(-1)
            elif k_ == "l1.weight" and v_.shape != st[k_].shape:
                # warm start without the extra inputs: copy the 2W activation columns, the extras start at zero
                st[k_] = torch.zeros_like(st[k_])
                st[k_][:, :v_.shape[1]] = v_
            else:
                st[k_] = v_
        m.load_state_dict(st)
    tq = torch.tensor(taus) if taus else None

    def lossf(y, t, w):
        if tq is None:
            return (w * (y - t) ** 2).sum() / w.sum()
        d = t[:, None] - y
        return (w[:, None] * torch.maximum(tq * d, (tq - 1) * d)).sum() / (w.sum() * len(taus))

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
            us, them, bk, pst, hand, ex_, t, w = tr.batch(np.sort(idx[i:i + a.bs]))
            y = m(us, them, bk, pst, hand, ex_)
            loss = lossf(y, t, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss)
            k += 1
        p = evaluate(m, va)
        tv = np.clip(va.nn, -CLIP, CLIP)
        wv = 0.1 + np.exp(-np.abs(va.nn - va.thr) / a.focus) if a.focus > 0 else np.ones_like(tv)
        if tq is None:
            v = float((wv * (np.clip(p, -CLIP, CLIP) - tv) ** 2).sum() / wv.sum())
            sd = float(np.std((p - va.nn)[np.abs(va.nn) < CLIP]) * 400)
            stat = "resid sd %.1f cp" % sd
        else:
            d = tv[:, None] - p
            ta = np.array(taus)
            v = float((wv[:, None] * np.maximum(ta * d, (ta - 1) * d)).sum() / (wv.sum() * len(taus)))
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
