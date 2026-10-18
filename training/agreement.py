#!/usr/bin/env python3
"""Agreement tracking: net static eval vs search eval, per tier.
pred_full = net_forward + base; agreement = |pred_full - teacher|.
Tiers: regret (wsum>2) vs base; common (top-half pattern freq) vs rare.
Answers: does weighting emphasis change agreement where it matters?
Usage: python training/agreement.py --tups <a.tup,b.tup> --data <npz>
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, 'training')
import train_tuple as T
import prio_queue as Q

SP_OFF = {'s': 0, 'H': 2916, 'V': 9477, 'D1': 16038, 'D2': 22599,
          'q2': 29160, 'envus': 37921, 'envthem': 73642,
          'mobus': 109363, 'mobthem': 111863, 'pa': 114818}
DE_OFF = {'capnus': 0, 'capnthem': 9, 'convus': 18, 'convthem': 274}
SP_N = 114818 + 80 * 6561
DE_N = 274 + 256


def load_net(path):
    bags = [torch.nn.EmbeddingBag(SP_N, 1, mode='sum', sparse=True),
            torch.nn.EmbeddingBag(DE_N, 1, mode='sum', sparse=True)]
    bias = torch.zeros(1)
    b0, ws0, wps0, rest0, cc0, pa0 = T.load_tup(path)
    with torch.no_grad():
        W, D = bags[0].weight, bags[1].weight
        bias.copy_(torch.tensor([b0]))
        W[SP_OFF['s']:SP_OFF['s'] + 36 * 81].copy_(torch.from_numpy(ws0.reshape(-1, 1)))
        for d, nm in enumerate(['H', 'V', 'D1', 'D2']):
            W[SP_OFF[nm]:SP_OFF[nm] + 6561].copy_(torch.from_numpy(wps0[d].reshape(-1, 1)))
        if rest0 is not None:
            wq, weu, wet, wmu, wmt = rest0
            W[SP_OFF['q2']:SP_OFF['q2'] + 36 * 256].copy_(torch.from_numpy(wq.reshape(-1, 1)))
            W[SP_OFF['envus']:SP_OFF['envus'] + 35721].copy_(torch.from_numpy(weu.reshape(-1, 1)))
            W[SP_OFF['envthem']:SP_OFF['envthem'] + 35721].copy_(torch.from_numpy(wet.reshape(-1, 1)))
            W[SP_OFF['mobus']:SP_OFF['mobus'] + 2500].copy_(torch.from_numpy(wmu.reshape(-1, 1)))
            W[SP_OFF['mobthem']:SP_OFF['mobthem'] + 2500].copy_(torch.from_numpy(wmt.reshape(-1, 1)))
        if cc0 is not None:
            wcu, wct, wvu, wvt = cc0
            D[DE_OFF['capnus']:DE_OFF['capnus'] + 9].copy_(torch.from_numpy(wcu.reshape(-1, 1)))
            D[DE_OFF['capnthem']:DE_OFF['capnthem'] + 9].copy_(torch.from_numpy(wct.reshape(-1, 1)))
            D[DE_OFF['convus']:DE_OFF['convus'] + 256].copy_(torch.from_numpy(wvu.reshape(-1, 1)))
            D[DE_OFF['convthem']:DE_OFF['convthem'] + 256].copy_(torch.from_numpy(wvt.reshape(-1, 1)))
        if pa0 is not None:
            W[SP_OFF['pa']:SP_OFF['pa'] + 80 * 6561].copy_(torch.from_numpy(pa0.reshape(-1, 1)))
    for bg in bags:
        bg.eval()
    return bags, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tups', required=True)
    ap.add_argument('--data', default='data/nnue/regretloop_r9.npz')
    a = ap.parse_args()
    d = np.load(a.data)
    keep = np.abs(d['teacher'].astype(np.float64)) <= 5000.0
    d = {k: (v[keep] if len(v) == len(keep) else v) for k, v in dict(d).items()}
    n = int(d['occ0'].shape[0])
    print('rows after mate filter: %d' % n, flush=True)
    teach = d['teacher'].astype(np.float64)
    w = d['wsum'].astype(np.float64) if 'wsum' in d else np.ones(n)
    base = T.base_eval_stm(d['occ0'], d['occ1'], d['blk'], d['stm']).astype(np.float64)
    print('materializing %d rows' % n, flush=True)
    F, Qb, EUS, ETH, MUS, MTH, _, _, _, _ = Q.build_feats(
        d['occ0'], d['occ1'], d['blk'], d['stm'])
    SP = Q.materialize_sp(F.astype(np.int64), Qb, EUS, ETH, MUS, MTH)
    del F, Qb, EUS, ETH, MUS, MTH
    counts = np.bincount(SP.reshape(-1).astype(np.int64), minlength=SP_N).astype(np.float64)
    rel = counts[SP].mean(axis=1)
    rel /= rel.max() if rel.max() > 0 else 1.0
    del SP, counts
    is_reg = w > 2.0
    is_com = rel >= np.median(rel)
    print('tiers: regret=%d base=%d common=%d rare=%d' % (
        is_reg.sum(), (~is_reg).sum(), is_com.sum(), (~is_com).sum()), flush=True)
    CH = 2000
    for tp in a.tups.split(','):
        bags, bias = load_net(tp.strip())
        errs = []
        with torch.no_grad():
            F2, Q2, EU2, ET2, MU2, MT2, CU2, CT2, VU2, VT2 = Q.build_feats(
                d['occ0'], d['occ1'], d['blk'], d['stm'])
            import torch as _t
            for s in range(0, n, CH):
                e = min(n, s + CH)
                SPc = Q.materialize_sp(F2[s:e].astype(np.int64), Q2[s:e], EU2[s:e], ET2[s:e],
                                        MU2[s:e], MT2[s:e])
                DEc = np.concatenate([
                    (DE_OFF['capnus'] + CU2[s:e].astype(np.int64)).reshape(e - s, -1),
                    (DE_OFF['capnthem'] + CT2[s:e].astype(np.int64)).reshape(e - s, -1),
                    (DE_OFF['convus'] + VU2[s:e].astype(np.int64)),
                    (DE_OFF['convthem'] + VT2[s:e].astype(np.int64))], axis=1)
                Xsp = _t.from_numpy(SPc)
                Xde = _t.from_numpy(DEc)
                pred = (bags[0](Xsp).squeeze(1) + bags[1](Xde).squeeze(1) + float(bias)).numpy()
                errs.append(np.abs(pred + base[s:e] - teach[s:e]))
                del SPc, DEc, Xsp, Xde, pred
        errs = np.concatenate(errs)
        for tol in (50, 100, 200):
            a = float((errs <= tol).mean() * 100)
            r = float((errs[is_reg] <= tol).mean() * 100) if is_reg.sum() else 0.0
            print('   within %3dcp: all=%.1f%% regret=%.1f%%' % (tol, a, r), flush=True)
        print('%-28s agree|err| all=%.0f regret=%.0f base=%.0f common=%.0f rare=%.0f' % (
            tp.split('/')[-1], errs.mean(), errs[is_reg].mean(), errs[~is_reg].mean(),
            errs[is_com].mean(), errs[~is_com].mean()), flush=True)


if __name__ == '__main__':
    main()
