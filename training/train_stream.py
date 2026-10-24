import argparse
import ctypes
import gc
import os
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T

import torch


class Lookahead:
    def __init__(self, base, k=5, alpha=0.5):
        self.base = base
        self.k = max(1, int(k))
        self.alpha = float(alpha)
        self.n = 0
        self.slow = [p.detach().clone() for g in base.param_groups for p in g['params']]

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self):
        self.base.zero_grad()

    def step(self):
        self.base.step()
        self.n += 1
        if self.n % self.k:
            return
        i = 0
        for g in self.base.param_groups:
            for p in g['params']:
                s = self.slow[i]
                s.add_(self.alpha * (p.detach() - s))
                p.data.copy_(s)
                i += 1


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gb():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / (1024.0 ** 3)


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 3)
    except Exception:
        return float('nan')


SP_OFF = {'s': 0, 'H': 2916, 'V': 9477, 'D1': 16038, 'D2': 22599,
          'q2': 29160, 'envus': 38376, 'envthem': 74097,
          'mobus': 109818, 'mobthem': 112318, 'pa': 114818}
SP_N = 114818 + 80 * 6561
DE_OFF = {'capnus': 0, 'capnthem': 9, 'convus': 18, 'convthem': 274}
DE_N = 274 + 256
SP_DIR = {0: 'H', 1: 'V', 2: 'D1', 3: 'D2'}

AR = np.arange(36, dtype=np.int64)[None, :]
AR2 = np.arange(36, dtype=np.int64)[None, :] * 256
PAX = np.array([x for (dd, x, y) in T.PAIRS], dtype=np.int64)
PAY = np.array([y for (dd, x, y) in T.PAIRS], dtype=np.int64)
PAOFF = np.arange(80, dtype=np.int64)[None, :] * 6561
PAL = [[np.array([x for (dd, x, y) in T.PAIRS if dd == d], dtype=np.int64),
        np.array([y for (dd, x, y) in T.PAIRS if dd == d], dtype=np.int64)]
       for d in range(4)]


def sp_width(active):
    return 36 + sum(len(PAL[d][0]) for d in active) + 36 + 49 + 49 + 1 + 1 + 80


def de_width(active):
    return 1 + 1 + 49 + 49


def features_for(occ0c, occ1c, blkc, stmc):
    is_b = (stmc == 0)
    own = np.where(is_b, occ0c, occ1c)
    enm = np.where(is_b, occ1c, occ0c)
    F = T.block_indices(own, enm).astype(np.int8)
    Q = T.q2_indices(own, enm, blkc)
    EUS, ETH = T.env_tables(own, enm, blkc)
    MUS, MTH = T.mob_tables(own, enm, blkc)
    CUS, CTH = T.capn_tables(own, enm, blkc)
    VUS, VTH = T.conv_tables(own, enm, blkc)
    del own, enm
    return F, Q, EUS, ETH, MUS, MTH, CUS, CTH, VUS, VTH


def materialize32(F, Q, EUS, ETH, MUS, MTH, CUS, CTH, VUS, VTH, active):
    n = F.shape[0]
    F32 = F.astype(np.int32)
    WSP = sp_width(active)
    WDE = de_width(active)
    SP = np.empty((n, WSP), dtype=np.int32)
    if n:
        np.add(F32, AR * 81, out=SP[:, 0:36], casting='unsafe')
        c = 36
        for d in active:
            xa, ya = PAL[d]
            w = len(xa)
            np.add(F32[:, xa], 81 * F32[:, ya], out=SP[:, c:c + w], casting='unsafe')
            np.add(SP[:, c:c + w], SP_OFF[SP_DIR[d]], out=SP[:, c:c + w], casting='unsafe')
            c += w
        np.add(Q, AR2 + SP_OFF['q2'], out=SP[:, c:c + 36], casting='unsafe')
        c += 36
        np.add(EUS, SP_OFF['envus'], out=SP[:, c:c + 49], casting='unsafe')
        c += 49
        np.add(ETH, SP_OFF['envthem'], out=SP[:, c:c + 49], casting='unsafe')
        c += 49
        np.add(MUS.reshape(n, 1), SP_OFF['mobus'], out=SP[:, c:c + 1], casting='unsafe')
        c += 1
        np.add(MTH.reshape(n, 1), SP_OFF['mobthem'], out=SP[:, c:c + 1], casting='unsafe')
        c += 1
        pa = F32[:, PAX] + 81 * F32[:, PAY]
        np.add(pa, SP_OFF['pa'] + PAOFF, out=SP[:, c:c + 80], casting='unsafe')
        del pa
        DE = np.empty((n, WDE), dtype=np.int32)
        np.add(CUS.reshape(n, 1), DE_OFF['capnus'], out=DE[:, 0:1], casting='unsafe')
        np.add(CTH.reshape(n, 1), DE_OFF['capnthem'], out=DE[:, 1:2], casting='unsafe')
        np.add(VUS, DE_OFF['convus'], out=DE[:, 2:51], casting='unsafe')
        np.add(VTH, DE_OFF['convthem'], out=DE[:, 51:100], casting='unsafe')
        del F32
    else:
        DE = np.empty((0, WDE), dtype=np.int32)
    return SP, DE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--chunk-rows', type=int, default=800000)
    ap.add_argument('--opt', default='adam', choices=['sgd', 'adam', 'adamw', 'radam', 'nadam'])
    ap.add_argument('--lookahead', action='store_true')
    ap.add_argument('--lookahead-k', type=int, default=5)
    ap.add_argument('--lookahead-alpha', type=float, default=0.5)
    ap.add_argument('--opt-lr', type=float, default=0.001)
    ap.add_argument('--conv-lr', type=float, default=0.003)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--freq-decay', type=float, default=0.0)
    ap.add_argument('--init', default='')
    ap.add_argument('--out', default='data/nnue/tupStream.tup')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--plateau', type=float, default=1e-4)
    ap.add_argument('--max-epochs', type=int, default=8)
    ap.add_argument('--cyclic-T0', type=int, default=0)
    ap.add_argument('--cyclic-mult', type=float, default=2.0)
    ap.add_argument('--cyclic-lr-min', type=float, default=1e-5)
    ap.add_argument('--val-probe', default='')
    ap.add_argument('--agree-select', action='store_true')
    ap.add_argument('--tcol', default='teacher', choices=['sc', 'teacher', 'titanium_cp'],
                    help="target column: 'teacher' (precedence mix), 'sc' (legacy), "
                         "'titanium_cp' (Titanium 5k search labels = Stockfish-style base stage)")
    ap.add_argument('--blend', type=float, default=1.0)
    ap.add_argument('--outcome-weight', type=float, default=0.0)
    ap.add_argument('--wdl-loss', action='store_true')
    ap.add_argument('--wdl-scale', type=float, default=400.0)
    ap.add_argument('--outcome-col', default='out')
    ap.add_argument('--teacher-prob-col', default='')
    ap.add_argument('--weight-col', default='wsum', help='per-row sample weight column (v3: wsum3)')
    ap.add_argument('--outcome-weight-col', default='',
                    help='per-row outcome weight (lambda on the game result) column, e.g. ow_row; overrides --outcome-weight in the WDL target')
    ap.add_argument('--resume', action='store_true',
                    help='resume from a.out.resume.* files; bounds any kill loss to one chunk')
    ap.add_argument('--mock-epoch', action='store_true',
                    help='digest one full epoch forward-only (no backward/step/writes) and '
                         'report rows + loss. Needs --sidecar-dir. NOT a training run.')
    ap.add_argument('--dirs', default='0,1,2,3')
    ap.add_argument('--sidecar-dir', default='',
                    help='pre-materialized SP/DE sidecars (training/materialize_sidecar.py); '
                         'skips per-epoch features_for+materialize32. Must match --dirs and full rows.')
    ap.add_argument('--debug-chunks', type=int, default=0,
                    help='WIRING TEST ONLY: train only the first N chunks (partial epoch). '
                         'Never use for a real run/candidate.')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--min-free-mb', type=int, default=1200)
    a = ap.parse_args()

    if a.threads > 0:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    z = np.load(a.cache)
    outcol = a.outcome_col if a.outcome_col in z else 'out'
    if a.wdl_loss and outcol not in z:
        raise SystemExit('wdl-loss requires an outcome column (%s); no training without it' % a.outcome_col)
    occ0 = z['occ0'].astype(np.uint64)
    occ1 = z['occ1'].astype(np.uint64)
    blk = z['blk'].astype(np.uint64)
    stm = z['stm'].astype(np.uint8)
    MATE_CAP = 5000.0
    raw_t = z['teacher'].astype(np.float64) if 'teacher' in z else None
    keep = np.ones(len(occ0), dtype=bool)
    if raw_t is not None:
        bad = ~np.isfinite(raw_t) | (np.abs(raw_t) > MATE_CAP)
        keep = ~bad
        if bad.any():
            print('mate-spike filter: dropping %d/%d rows (|teacher|>%.0f)'
                  % (int(bad.sum()), len(raw_t), MATE_CAP), flush=True)
            occ0, occ1, blk, stm = occ0[~bad], occ1[~bad], blk[~bad], stm[~bad]
            z_teacher = raw_t[~bad]
            z_out = (z[outcol].astype(np.float64) if outcol in z
                     else np.zeros(len(raw_t)))[~bad]
            z_sc = (z['sc'].astype(np.float64) if 'sc' in z else np.zeros(len(raw_t)))[~bad]
        else:
            z_teacher = raw_t
            z_out = (z[outcol].astype(np.float64) if outcol in z
                     else np.zeros(len(raw_t)))
            z_sc = (z['sc'].astype(np.float64) if 'sc' in z else np.zeros(len(raw_t)))
    else:
        z_teacher = None
        z_out = (z[outcol].astype(np.float64) if outcol in z
                 else np.zeros(len(occ0)))
        z_sc = (z['sc'].astype(np.float64) if 'sc' in z else np.zeros(len(occ0)))
    n = len(occ0)
    print('rows %d' % n, flush=True)
    assert a.tcol in ('sc', 'teacher', 'titanium_cp')
    if a.tcol == 'teacher' and z_teacher is None:
        print('tcol=teacher requested but no teacher column: falling back to sc', flush=True)
        a.tcol = 'sc'
    if a.tcol == 'teacher':
        prime = np.clip(z_teacher * 1.0, -1500.0, 1500.0)
    elif a.tcol == 'titanium_cp':
        assert 'titanium_cp' in z, 'tcol=titanium_cp but cache has no titanium_cp column'
        z_tcp = z['titanium_cp'].astype(np.float64)
        if len(z_tcp) != n:
            z_tcp = z_tcp[keep]
        prime = np.clip(z_tcp, -1500.0, 1500.0)
        print('base-stage target titanium_cp: mean %+.1f clipped to [-1500,1500] (%.1f%% rows at rail)'
              % (prime.mean(), 100.0 * (np.abs(prime) >= 1500.0).mean()), flush=True)
        del z_tcp
    else:
        prime = np.clip(z_sc * 2000.0, -1500.0, 1500.0)
        prime = prime[:n] if len(prime) != n else prime
    if abs(a.blend - 1.0) > 1e-9:
        assert z_teacher is not None and a.tcol == 'teacher', 'blend needs teacher tcol + sc legacy'
        legacy = np.clip(z_sc * 2000.0, -1500.0, 1500.0)
        score_cp = (a.blend * prime + (1.0 - a.blend) * legacy).astype(np.float64)
    else:
        score_cp = prime.astype(np.float64)
    print('teacher column: %s blend %.2f' % (a.tcol, a.blend), flush=True)
    outcome_cp = np.clip((z_out * 2.0 - 1.0) * 1000.0, -1500.0, 1500.0)
    print('outcome stats: mean %+.3f -> cp mean %+.1f std %.1f'
          % (z_out.mean(), outcome_cp.mean(), outcome_cp.std()), flush=True)
    ow = float(a.outcome_weight)
    assert 0.0 <= ow <= 1.0
    best_cp = ((1.0 - ow) * score_cp + ow * outcome_cp).astype(np.float32) if ow > 0 else score_cp.astype(np.float32)
    wdl_target = None
    if a.wdl_loss:
        if a.teacher_prob_col and a.teacher_prob_col not in z:
            raise SystemExit('--teacher-prob-col %s not in cache (columns: %s); refusing the silent cp fallback'
                             % (a.teacher_prob_col, ','.join(z.files)))
        if a.teacher_prob_col:
            tp = np.asarray(z[a.teacher_prob_col], dtype=np.float64)
            if a.outcome_weight_col:
                if a.outcome_weight_col not in z:
                    raise SystemExit('--outcome-weight-col %s not in cache' % a.outcome_weight_col)
                owr = np.asarray(z[a.outcome_weight_col], dtype=np.float64)
                assert len(owr) == len(tp) == len(z_out) and (owr >= 0).all() and (owr <= 1).all()
                wdl_target = ((1.0 - owr) * tp + owr * z_out).astype(np.float32)
                print('per-row outcome weight %s: mean %.3f min %.3f max %.3f' % (
                    a.outcome_weight_col, owr.mean(), owr.min(), owr.max()), flush=True)
                del owr
            else:
                wdl_target = ((1.0 - ow) * tp + ow * z_out).astype(np.float32)
            print('wdl-loss ON teacher-prob-col %s lambda %.2f target mean %.4f' % (
                a.teacher_prob_col, 1.0 - ow, float(wdl_target.mean())), flush=True)
        else:
            wdl_target = ((1.0 - ow) * (1.0 / (1.0 + np.exp(-score_cp / a.wdl_scale)))
                          + ow * z_out).astype(np.float32)
        print('wdl-loss ON scale %.0f lambda %.2f target mean %.4f' % (
            a.wdl_scale, 1.0 - ow, float(wdl_target.mean())), flush=True)
    print('outcome-weight %.2f' % ow, flush=True)
    if a.weight_col != 'wsum' and a.weight_col not in z:
        raise SystemExit('--weight-col %s not in cache' % a.weight_col)
    wsum_full = z[a.weight_col].astype(np.float64) if a.weight_col in z else None
    if wsum_full is not None:
        wsum_full = wsum_full[keep]
    if 'teacher_kind' in z and wdl_target is not None and wsum_full is not None:
        tk = z['teacher_kind'][keep]
        for c, nm in ((1, 'kata'), (2, 'own/titanium5k'), (3, 'terminal')):
            m = tk == c
            if m.any():
                print('target kind %d %-15s rows %9d  target mean %.3f std %.3f  weight share %.3f'
                      % (c, nm, int(m.sum()), float(wdl_target[m].mean()), float(wdl_target[m].std()),
                         float(wsum_full[m].sum() / wsum_full.sum())), flush=True)
        del tk
    if a.rows:
        sel = np.random.choice(n, min(a.rows, n), replace=False)
        occ0, occ1, blk, stm, best_cp = occ0[sel], occ1[sel], blk[sel], stm[sel], best_cp[sel]
        if wdl_target is not None:
            wdl_target = wdl_target[sel]
        if wsum_full is not None:
            wsum_full = wsum_full[sel]
        sel = None
        n = len(occ0)
        print('rows subsample %d' % n, flush=True)
    z.close()
    del z, z_teacher, z_out, z_sc, score_cp, outcome_cp, prime, raw_t, keep
    gc.collect()

    base = None
    if a.sidecar_dir:
        import json as _jsonb
        _manb = _jsonb.load(open(os.path.join(a.sidecar_dir, 'manifest.json')))
        _bl = _manb.get('base', [])
        if (_manb.get('dirs') == a.dirs and _manb.get('rows') == n and len(_bl)
                and sum(c['hi'] - c['lo'] for c in _bl) == n
                and all(os.path.exists(c['path']) for c in _bl)):
            print('startup: loading precomputed base (%d chunks)...' % len(_bl), flush=True)
            t_base = time.time()
            base = np.concatenate([np.load(c['path'], mmap_mode='r')
                                   for c in _bl]).astype(np.float32)
            print('startup: precomputed base loaded (%.0fs)' % (time.time() - t_base), flush=True)
    if base is None:
        print('startup: base eval over %d rows...' % n, flush=True)
        t_base = time.time()
        base = T.base_eval_stm(occ0, occ1, blk, stm)
    assert abs(T.base_eval_stm(np.array([np.uint64(1 << 42)]), np.array([np.uint64(1 << 6)]),
                               np.array([np.uint64(0)]), np.array([0]))[0] - 150.0) < 1e-9
    print('startup: base eval done (%.0fs)' % (time.time() - t_base), flush=True)
    resid = (best_cp - base).astype(np.float32)
    print('residual mean %+.1f std %.1f' % (resid.mean(), resid.std()), flush=True)
    base_full = base if a.wdl_loss else None
    del base, best_cp
    gc.collect()
    if wsum_full is not None:
        sw = np.clip(wsum_full, 0, None)
        del wsum_full
        sw = sw / max(sw.mean(), 1e-9)
        print('wsum weights: mean 1.0 max %.1f frac>2x %.3f'
              % (sw.max(), (sw > 2).mean()), flush=True)
        samp_w = sw.astype(np.float32)
        del sw
    else:
        samp_w = np.ones(n, np.float32)
    gc.collect()

    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * 0.95)
    tr, va = idx[:cut], idx[cut:]
    del idx
    ytr_full = wdl_target if a.wdl_loss else resid
    wtr_full = samp_w
    base_tr = base_full[tr] if (base_full is not None and not a.sidecar_dir) else None
    base_va = base_full[va] if base_full is not None else None
    if not a.sidecar_dir:
        del base_full

    active = [int(x) for x in a.dirs.split(',')]
    assert set(active) <= {0, 1, 2, 3} and active, a.dirs
    WSP = sp_width(active)
    WDE = de_width(active)
    print('stream: dirs %s WSP %d WDE %d chunk %d' % (a.dirs, WSP, WDE, a.chunk_rows), flush=True)

    print('startup: materializing val set (%d rows)...' % len(va), flush=True)
    t0 = time.time()
    fa = features_for(occ0[va], occ1[va], blk[va], stm[va])
    SPva, DEva = materialize32(*fa, active)
    del fa
    gc.collect()
    yva = torch.from_numpy(ytr_full[va])
    wva = torch.from_numpy(wtr_full[va])
    print('val materialized %d rows %.1fMB (%.0fs, free %.1fGB)'
          % (len(va), (SPva.nbytes + DEva.nbytes) / 1e6, time.time() - t0, free_gb()), flush=True)

    AG = None
    if a.val_probe:
        _pz = np.load(a.val_probe)
        _po = _pz['occ0'].astype(np.uint64)
        _pe = _pz['occ1'].astype(np.uint64)
        _pb = _pz['blk'].astype(np.uint64)
        _ps = _pz['stm'].astype(np.uint8)
        _f = features_for(_po, _pe, _pb, _ps)
        SPp, DEp = materialize32(*_f, active)
        del _f
        AG = {'teach': _pz['teacher'].astype(np.float32),
              'base': T.base_eval_stm(_po, _pe, _pb, _ps).astype(np.float32),
              'SP': SPp, 'DE': DEp}
        print('val probe: %d rows deep-teacher mean %+.1f' % (
            len(AG['teach']), float(AG['teach'].mean())), flush=True)

    bags = torch.nn.ModuleList(
        [torch.nn.EmbeddingBag(SP_N, 1, mode='sum', sparse=False),
         torch.nn.EmbeddingBag(DE_N, 1, mode='sum', sparse=False)])
    for bg in bags:
        torch.nn.init.zeros_(bg.weight)
    bias = torch.nn.Parameter(torch.zeros(1))
    groups = [{'params': [bags[0].weight, bias]}, {'params': [bags[1].weight], 'lr': a.conv_lr}]
    if a.opt == 'sgd':
        opt = torch.optim.SGD(groups, lr=a.opt_lr)
    elif a.opt == 'adam':
        opt = torch.optim.Adam(groups, lr=a.opt_lr, weight_decay=0.0)
    elif a.opt == 'adamw':
        opt = torch.optim.AdamW(groups, lr=a.opt_lr, weight_decay=0.0)
    elif a.opt == 'radam':
        opt = torch.optim.RAdam(groups, lr=a.opt_lr, weight_decay=0.0)
    elif a.opt == 'nadam':
        opt = torch.optim.NAdam(groups, lr=a.opt_lr, weight_decay=0.0)
    else:
        raise ValueError(a.opt)
    if a.lookahead:
        opt = Lookahead(opt, a.lookahead_k, a.lookahead_alpha)
    fdec = float(a.freq_decay)
    freq = None
    if fdec > 0:
        freq = [torch.zeros(SP_N), torch.zeros(DE_N)]

    def zero_constrained():
        with torch.no_grad():
            sp, de = bags[0].weight.data, bags[1].weight.data
            sp[0].zero_()
            for d in range(4):
                sp[SP_OFF[['H', 'V', 'D1', 'D2'][d]]].zero_()
            sp[SP_OFF['q2']].zero_()
            for sq in range(49):
                sp[SP_OFF['envus'] + sq * 729].zero_()
                sp[SP_OFF['envthem'] + sq * 729].zero_()
            for ph in range(50):
                sp[SP_OFF['mobus'] + ph * 50].zero_()
                sp[SP_OFF['mobthem'] + ph * 50].zero_()
            for pi in range(80):
                sp[SP_OFF['pa'] + pi * 6561].zero_()
            de[DE_OFF['capnus']].zero_()
            de[DE_OFF['capnthem']].zero_()
            de[DE_OFF['convus']].zero_()
            de[DE_OFF['convthem']].zero_()

    if a.init:
        print('startup: loading warm-start %s...' % a.init, flush=True)
        t_init = time.time()
        b0, ws0, wps0, rest0, cc0, pa0 = T.load_tup(a.init)
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
        zero_constrained()
        print('warm start from %s (%.0fs)' % (a.init, time.time() - t_init), flush=True)

    def forward(Xsp, Xde):
        return bags[0](Xsp).squeeze(1) + bags[1](Xde).squeeze(1) + bias

    def dump_tup(path):
        W = bags[0].weight.detach().squeeze(1).numpy()
        D = bags[1].weight.detach().squeeze(1).numpy()
        ws = W[SP_OFF['s']:SP_OFF['s'] + 36 * 81].reshape(36, 81)
        wp = [W[SP_OFF[nm]:SP_OFF[nm] + 6561].reshape(6561) for nm in ['H', 'V', 'D1', 'D2']]
        wq = W[SP_OFF['q2']:SP_OFF['q2'] + 36 * 256].reshape(36 * 256)
        weu = W[SP_OFF['envus']:SP_OFF['envus'] + 35721]
        wet = W[SP_OFF['envthem']:SP_OFF['envthem'] + 35721]
        wmu = W[SP_OFF['mobus']:SP_OFF['mobus'] + 2500]
        wmt = W[SP_OFF['mobthem']:SP_OFF['mobthem'] + 2500]
        wcu = D[DE_OFF['capnus']:DE_OFF['capnus'] + 9]
        wct = D[DE_OFF['capnthem']:DE_OFF['capnthem'] + 9]
        wvu = D[DE_OFF['convus']:DE_OFF['convus'] + 256]
        wvt = D[DE_OFF['convthem']:DE_OFF['convthem'] + 256]
        wpa = W[SP_OFF['pa']:SP_OFF['pa'] + 80 * 6561]
        allw = np.concatenate([ws.reshape(-1)] + [p.reshape(-1) for p in wp]
                              + [wq, weu, wet, wmu, wmt, wcu, wct, wvu, wvt, wpa])
        print('weight spread: max|w| %.1f mean|w| %.2f frac_nonzero %.3f bias %+.1f'
              % (np.abs(allw).max(), np.abs(allw).mean(), (allw != 0).mean(), float(bias.item())),
              flush=True)
        counts = [24, 24, 16, 16]
        q = lambda v: max(-127, min(127, int(round(v))))
        payload = struct.pack('<i', int(round(float(bias.item()))))
        payload += bytes(bytearray([(q(v) & 0xFF) for v in ws.reshape(-1)]))
        intc = [counts[d] if d in active else 0 for d in range(4)]
        payload += struct.pack('<4i', *intc)
        for d in range(4):
            seg = wp[d].reshape(-1) if d in active else np.zeros(6561)
            payload += bytes(bytearray([(q(v) & 0xFF) for v in seg]))
        q16 = lambda arr: struct.pack('<%dh' % len(arr),
                                      *[max(-32767, min(32767, int(round(v)))) for v in arr])
        payload += q16(wq) + q16(weu) + q16(wet) + q16(wmu) + q16(wmt)
        payload += q16(wcu) + q16(wct) + q16(wvu) + q16(wvt)
        q8 = lambda arr: bytes(bytearray([(max(-127, min(127, int(round(v)))) & 0xFF) for v in arr]))
        payload += q8(wpa)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(b'TUP7' + payload)
        os.replace(tmp, path)
        print('wrote %s %.1fKB dirs %s TUP7' % (path, os.path.getsize(path) / 1024, a.dirs), flush=True)

    bs = 4096
    ntr = len(tr)
    bounds = [(lo, min(lo + a.chunk_rows, ntr)) for lo in range(0, ntr, a.chunk_rows)]
    print('chunks %d x ~%d rows, epochs %d' % (len(bounds), a.chunk_rows, a.epochs), flush=True)

    SC = None
    if a.sidecar_dir:
        import json as _json
        man = _json.load(open(os.path.join(a.sidecar_dir, 'manifest.json')))
        assert man['dirs'] == a.dirs, 'sidecar dirs %s != trainer dirs %s' % (man['dirs'], a.dirs)
        assert man['rows'] == n, 'sidecar rows %d != cache rows %d' % (man['rows'], n)
        assert man['WSP'] == WSP and man['WDE'] == WDE, 'sidecar width drift'
        assert len(man['chunks']) * 0 + sum(c['hi'] - c['lo'] for c in man['chunks']) == n, \
            'sidecar chunks do not cover all rows'
        SC = [(c['lo'], c['hi'],
               np.load(c['sp'], mmap_mode='r'), np.load(c['de'], mmap_mode='r'))
              for c in man['chunks']]
        print('sidecar: %d chunks mapped, per-epoch features_for+materialize32 OFF'
              % len(SC), flush=True)

    if SC is not None:
        # Sequential sidecar-chunk loop: shuffled chunk order per epoch + in-chunk
        # shuffle in train_pass. Random fancy-gather across the 56GB sidecar
        # page-faults (~320s/800k cold); sequential slices stream at disk speed.
        bounds = [(lo, hi) for (lo, hi, _, _) in SC]
        srt = np.sort(tr)
        print('sidecar loop: %d chunk-aligned bounds over sorted train rows' % len(bounds),
              flush=True)

    def sidecar_slice(lo, hi):
        SPm = DEm = None
        for (clo, chi, SPm_c, DEm_c) in SC:
            if clo == lo and chi == hi:
                SPm, DEm = SPm_c, DEm_c
                break
        assert SPm is not None, 'sidecar chunk %d-%d missing' % (lo, hi)
        # Whole-chunk sequential read (disk-speed) then in-RAM mask.
        # Fancy-indexing the mmap directly page-faults (~8s/40k rows cold).
        SPf = np.asarray(SPm)
        DEf = np.asarray(DEm)
        a = int(np.searchsorted(srt, lo))
        b = int(np.searchsorted(srt, hi))
        local = srt[a:b] - lo
        return SPf[local], DEf[local], srt[a:b]

    if a.debug_chunks:
        print('WARNING debug-chunks %d/%d: PARTIAL EPOCH, wiring test only'
              % (a.debug_chunks, len(bounds)), flush=True)
        bounds = bounds[:a.debug_chunks]

    cyc_on = a.cyclic_T0 > 0
    cyc_T, cyc_Ti = 0, max(a.cyclic_T0, 1)
    lr_max0 = float(a.opt_lr)
    lr_ratio = float(a.conv_lr) / max(lr_max0, 1e-12)
    max_ep = a.max_epochs if a.plateau > 0 else a.epochs
    prev_val, best_val, best_ep, stale = None, float('inf'), 0, 0
    best_state = None
    best_agree = [float('inf')]
    rng = np.random.RandomState(a.seed)
    # Crash-safe resume: manifest + float weights + best weights, all tmp+replace.
    rmeta, rw, rb = a.out + '.resume_pt', a.out + '.resume_w.pt', a.out + '.resume_best.pt'
    ep_start, resume_order, resume_done = 1, None, set()
    resume_tot, resume_cnt = 0.0, 0
    if a.resume and os.path.exists(rmeta):
        try:
            rm = torch.load(rmeta, map_location='cpu', weights_only=False)
        except Exception as e:
            rm = None
            print('resume files unreadable, starting clean: %s' % e, flush=True)
        if rm is not None and rm.get('cache') == a.cache and rm.get('seed') == a.seed and rm.get('dirs') == a.dirs and rm.get('sidecar', '') == a.sidecar_dir:
            ep_start = int(rm['ep'])
            resume_order = [int(x) for x in rm['order']]
            resume_done = set(int(x) for x in rm['done'])
            best_val, best_ep, prev_val, stale = rm['best_val'], rm['best_ep'], rm['prev_val'], rm['stale']
            cyc_T, cyc_Ti = int(rm.get('cyc_T', cyc_T)), int(rm.get('cyc_Ti', cyc_Ti))
            resume_tot, resume_cnt = float(rm.get('tot', 0.0)), int(rm.get('cnt', 0))
            w = torch.load(rw, map_location='cpu')
            with torch.no_grad():
                for bg, ww in zip(bags, w[:-1]):
                    bg.weight.copy_(ww)
                bias.copy_(w[-1])
            zero_constrained()
            if os.path.exists(rb):
                bw = torch.load(rb, map_location='cpu')
                best_state = ([t.clone() for t in bw[0]], bw[1].clone())
            print('resumed weights at ep %d (%d chunks already banked)' % (ep_start, len(resume_done)), flush=True)
        else:
            print('resume files present but args mismatch: starting clean', flush=True)

    if a.mock_epoch:
        assert SC is not None, '--mock-epoch needs --sidecar-dir'
        print('mock epoch: forward-only, no updates, no writes', flush=True)
        tot, cnt, t0 = 0.0, 0, time.time()
        with torch.no_grad():
            for gi, (lo, hi) in enumerate(bounds):
                SPc, DEc, rows = sidecar_slice(lo, hi)
                wc = torch.from_numpy(wtr_full[rows])
                yc = torch.from_numpy(ytr_full[rows])
                bc = torch.from_numpy(base_full[rows]) if base_full is not None else None
                for j in range(0, len(SPc), 100000):
                    e = min(len(SPc), j + 100000)
                    Xsp = torch.from_numpy(SPc[j:e].astype(np.int64))
                    Xde = torch.from_numpy(DEc[j:e].astype(np.int64))
                    pv = forward(Xsp, Xde)
                    if a.wdl_loss:
                        pv = torch.sigmoid((torch.from_numpy(bc[j:e]) + pv) / a.wdl_scale)
                    tot += (((pv - yc[j:e]) ** 2) * wc[j:e]).sum().item()
                    cnt += e - j
                del SPc, DEc, wc, yc, bc
                gc.collect()
                print('  mock chunk %d/%d rows %d digested %d/%d loss %.6f (%.0fs)'
                      % (gi + 1, len(bounds), len(rows), cnt, ntr, tot / max(cnt, 1),
                         time.time() - t0), flush=True)
        print('mock epoch done: digested %d/%d rows loss %.6f (%.0fs) NO UPDATES NO WRITES'
              % (cnt, ntr, tot / max(cnt, 1), time.time() - t0), flush=True)
        return

    def train_pass(SPc, DEc, wc, yc, basec=None):
        nc = len(SPc)
        perm = torch.randperm(nc)
        tot, cnt = 0.0, 0
        for i in range(0, nc, bs):
            b = perm[i:i + bs]
            bn = b.numpy()
            Xsp = torch.from_numpy(SPc[bn].astype(np.int64))
            Xde = torch.from_numpy(DEc[bn].astype(np.int64))
            if freq is not None:
                with torch.no_grad():
                    freq[0] += torch.bincount(Xsp.flatten(), minlength=SP_N).float()
                    freq[1] += torch.bincount(Xde.flatten(), minlength=DE_N).float()
            opt.zero_grad()
            pred = forward(Xsp, Xde)
            l2 = sum((bg.weight ** 2).sum() for bg in bags) if a.wd else 0.0
            wb = wc[b]
            if a.wdl_loss:
                bc = torch.from_numpy(basec[bn])
                p = torch.sigmoid((bc + pred) / a.wdl_scale)
                loss = ((p - yc[b]) ** 2 * wb).mean() + a.wd * l2
            else:
                loss = ((pred - yc[b]) ** 2 * wb).mean() + a.wd * l2
            loss.backward()
            opt.step()
            zero_constrained()
            if freq is not None:
                with torch.no_grad():
                    lr_now = opt.param_groups[0].get('lr', 0.0)
                    for bi, bg in enumerate(bags):
                        fr = freq[bi]
                        fmax = float(fr.max()) if float(fr.max()) > 0 else 1.0
                        rarity = torch.sqrt(fmax / (fr + 1.0))
                        shrink = (lr_now * fdec * rarity).clamp(max=0.05)
                        bg.weight.mul_(1.0 - shrink.view(-1, 1))
            tot += loss.item() * len(b)
            cnt += len(b)
        return tot, cnt

    print('startup: complete, entering epoch loop (ep %d-%d)' % (ep_start, max_ep), flush=True)
    for ep in range(ep_start, max_ep + 1):
        t0 = time.time()
        for bg in bags:
            bg.train()
        if a.resume and resume_order is not None and ep == ep_start:
            order = np.asarray(resume_order)
            done = set(resume_done)
            tot, cnt = resume_tot, resume_cnt
            print('resuming ep %d, skipping %d/%d chunks' % (ep, len(done), len(order)), flush=True)
        else:
            tot, cnt = 0.0, 0
            order = rng.permutation(len(bounds))
            done = set()
        t_feat, t_mat, t_tr = 0.0, 0.0, 0.0
        n_seen = len(done)
        for ci in order:
            ci = int(ci)
            if ci in done:
                continue
            lo, hi = bounds[ci]
            if free_gb() * 1024.0 < a.min_free_mb:
                print('RAM GUARD: free %.0fMB < %dMB before chunk %d/%d'
                      % (free_gb() * 1024.0, a.min_free_mb, ci + 1, len(bounds)), flush=True)
                if best_state is not None:
                    with torch.no_grad():
                        for bg, w in zip(bags, best_state[0]):
                            bg.weight.copy_(w)
                        bias.copy_(best_state[1])
                    zero_constrained()
                dump_tup(a.out)
                return
            if SC is not None:
                ts = time.time()
                SPc, DEc, rows = sidecar_slice(lo, hi)
                t_feat += time.time() - ts
            else:
                rows = tr[lo:hi]
                tf = time.time()
                feats = features_for(occ0[rows], occ1[rows], blk[rows], stm[rows])
                t_feat += time.time() - tf
                tm = time.time()
                SPc, DEc = materialize32(*feats, active)
                del feats
                gc.collect()
                t_mat += time.time() - tm
            tt = time.time()
            wc = torch.from_numpy(wtr_full[rows])
            yc = torch.from_numpy(ytr_full[rows])
            if SC is not None:
                bc = base_full[rows] if base_full is not None else None
            else:
                bc = base_tr[lo:hi] if base_tr is not None else None
            # Zero-weight rows (e.g. --weight-col wsum_kataonly) carry no gradient; training on them only
            # wastes time and lets Adam's momentum drift the weights on all-zero steps -> drop them.
            nz = wtr_full[rows] > 0
            if not nz.all():
                nzt = torch.from_numpy(nz)
                SPc, DEc, wc, yc = SPc[nz], DEc[nz], wc[nzt], yc[nzt]
                if bc is not None:
                    bc = bc[nz]
            t_, c_ = train_pass(SPc, DEc, wc, yc, bc)
            tot += t_
            cnt += c_
            t_tr += time.time() - tt
            del SPc, DEc, wc, yc
            gc.collect()
            n_seen += 1
            print('  ep %d chunk %d/%d rows %d feat %.0fs mat %.0fs train %.0fs free %.1fGB train_mse %.4f (ep_avg %.4f)'
                  % (ep, n_seen, len(bounds), len(rows), t_feat, t_mat, t_tr, free_gb(),
                     t_ / max(c_, 1), tot / max(cnt, 1)), flush=True)
            if a.resume:
                # Bank this chunk: current weights + manifest, both tmp+replace.
                # A kill can never cost more than the in-flight chunk.
                with torch.no_grad():
                    torch.save([bg.weight.detach().cpu() for bg in bags] + [bias.detach().cpu()], rw + '.tmp')
                os.replace(rw + '.tmp', rw)
                if best_state is not None:
                    with torch.no_grad():
                        torch.save([[t.detach().cpu() for t in best_state[0]], best_state[1].detach().cpu()], rb + '.tmp')
                    os.replace(rb + '.tmp', rb)
                done.add(ci)
                torch.save({'ep': ep, 'order': [int(x) for x in order], 'done': sorted(done),
                            'best_val': best_val, 'best_ep': best_ep, 'prev_val': prev_val, 'stale': stale,
                            'cyc_T': cyc_T, 'cyc_Ti': cyc_Ti, 'tot': tot, 'cnt': cnt,
                            'cache': a.cache, 'seed': a.seed, 'dirs': a.dirs,
                            'sidecar': a.sidecar_dir}, rmeta + '.tmp')
                os.replace(rmeta + '.tmp', rmeta)
        for bg in bags:
            bg.eval()
        with torch.no_grad():
            totv, cntv = 0.0, 0
            for j in range(0, len(SPva), 100000):
                nv = min(100000, len(SPva) - j)
                Xsp = torch.from_numpy(SPva[j:j + nv].astype(np.int64))
                Xde = torch.from_numpy(DEva[j:j + nv].astype(np.int64))
                pv = forward(Xsp, Xde)
                if a.wdl_loss:
                    pv = torch.sigmoid((torch.from_numpy(base_va[j:j + nv]) + pv) / a.wdl_scale)
                wv = wva[j:j + nv]
                totv += (((pv - yva[j:j + nv]) ** 2) * wv).sum().item()
                cntv += nv
            vmse = totv / max(cntv, 1)
        agree_msg = ''
        if AG is not None:
            with torch.no_grad():
                to = 0.0
                cn = 0
                for j in range(0, len(AG['SP']), 2000):
                    e = min(len(AG['SP']), j + 2000)
                    Xsp = torch.from_numpy(AG['SP'][j:e].astype(np.int64))
                    Xde = torch.from_numpy(AG['DE'][j:e].astype(np.int64))
                    pv = forward(Xsp, Xde)
                    to += torch.abs(pv + torch.from_numpy(AG['base'][j:e])
                                    - torch.from_numpy(AG['teach'][j:e])).sum().item()
                    cn += e - j
                agree = to / max(cn, 1)
            agree_msg = ' agree %.0f' % agree
            if agree < best_agree[0]:
                best_agree[0] = agree
                if a.agree_select:
                    best_val, best_ep = vmse, ep
                    with torch.no_grad():
                        best_state = ([bg.weight.detach().clone() for bg in bags], bias.detach().clone())
                    if a.ckpt:
                        dump_tup(a.ckpt.format(ep=ep))
                        shutil.copy(a.ckpt.format(ep=ep), a.out)
                    agree_msg += ' AGREE-BEST->out'
                else:
                    agree_msg += ' (best %.0f)' % best_agree[0]
        improv = (prev_val - vmse) / max(prev_val, 1e-9) if prev_val else 1.0
        tag = ''
        if vmse < best_val:
            best_val, best_ep = vmse, ep
            with torch.no_grad():
                best_state = ([bg.weight.detach().clone() for bg in bags], bias.detach().clone())
            if a.ckpt:
                dump_tup(a.ckpt.format(ep=ep))
                shutil.copy(a.ckpt.format(ep=ep), a.out)
                tag = ' BEST->out'
        cyc_msg = ''
        if cyc_on:
            cyc_T += 1
            if cyc_T >= cyc_Ti and best_state is not None:
                with torch.no_grad():
                    for bg, w in zip(bags, best_state[0]):
                        bg.weight.copy_(w)
                    bias.copy_(best_state[1])
                zero_constrained()
                cyc_T, cyc_Ti = 0, int(cyc_Ti * a.cyclic_mult)
                cyc_msg = ' RESTART(T=%d)' % cyc_Ti
            frac = min(cyc_T / max(cyc_Ti, 1), 1.0)
            import math
            lr_ep = a.cyclic_lr_min + 0.5 * (lr_max0 - a.cyclic_lr_min) * (
                1.0 + math.cos(math.pi * frac))
            opt.param_groups[0]['lr'] = lr_ep
            opt.param_groups[1]['lr'] = max(lr_ep * lr_ratio, a.cyclic_lr_min)
        if a.plateau > 0 and prev_val is not None and improv < a.plateau:
            stale += 1
        else:
            stale = 0
        print('epoch %d train_mse %.6f val_mse %.6f%s rel_improv %+.2e bias %+.1f lr %.5f (%.0fs)%s%s'
              % (ep, tot / max(cnt, 1), vmse, agree_msg, improv, bias.item(),
                 opt.param_groups[0]['lr'], time.time() - t0, tag, cyc_msg), flush=True)
        prev_val = vmse
        if a.plateau > 0 and stale >= 2:
            print('plateau: rel improv < %.0e twice -> stop at ep %d (best ep %d val %.1f)'
                  % (a.plateau, ep, best_ep, best_val), flush=True)
            break
    if not a.ckpt:
        dump_tup(a.out)
    else:
        print('stream: best ep %d val %.1f -> %s' % (best_ep, best_val, a.out), flush=True)
    for _f in (rmeta, rw, rb):
        try:
            os.remove(_f)
        except OSError:
            pass


if __name__ == '__main__':
    main()
