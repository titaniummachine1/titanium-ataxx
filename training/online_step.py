import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T
import train_stream as S

import torch

SP_OFF = S.SP_OFF
SP_N = S.SP_N
DE_OFF = S.DE_OFF
DE_N = S.DE_N
AR = S.AR
AR2 = S.AR2
PAX = S.PAX
PAY = S.PAY
PAOFF = S.PAOFF
PAL = S.PAL


def dump_tup(path, W, D, bias, active):
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
    q = lambda v: max(-127, min(127, int(round(v))))
    payload = struct.pack('<i', int(round(float(bias))))
    payload += bytes(bytearray([(q(v) & 0xFF) for v in ws.reshape(-1)]))
    counts = [24, 24, 16, 16]
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
    with open(path, 'wb') as f:
        f.write(b'TUP7' + payload)


def board_str(occ0, occ1, blk, stm):
    out = []
    for i in range(49):
        bit = 1 << i
        out.append('#' if blk & bit else 'x' if occ0 & bit else 'o' if occ1 & bit else '.')
    return ''.join(out) + (' b' if stm == 0 else ' w')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/nnue/cache12M.npz')
    ap.add_argument('--init', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--out', default='data/nnue/online.tup')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--nodes', type=int, default=5000)
    ap.add_argument('--outcome-weight', type=float, default=0.5)
    ap.add_argument('--regret-cap', type=float, default=5.0)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--conv-lr', type=float, default=0.003)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--tmp', default='data/nnue/online_cur.tup')
    ap.add_argument('--seed', type=int, default=71)
    ap.add_argument('--log', default='logs/online_step.log')
    ap.add_argument('--dirs', default='0,1,2,3')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    z = np.load(a.cache, mmap_mode='r')
    n = len(z['occ0'])
    active = [int(x) for x in a.dirs.split(',')]
    print('cache %d rows, steps %d batch %d nodes %d' % (n, a.steps, a.batch, a.nodes), flush=True)

    bags = torch.nn.ModuleList(
        [torch.nn.EmbeddingBag(SP_N, 1, mode='sum', sparse=True),
         torch.nn.EmbeddingBag(DE_N, 1, mode='sum', sparse=True)])
    for bg in bags:
        torch.nn.init.zeros_(bg.weight)
    bias = torch.nn.Parameter(torch.zeros(1))
    groups = [{'params': [bags[0].weight, bias]}, {'params': [bags[1].weight], 'lr': a.conv_lr}]
    opt = torch.optim.Adam(groups, lr=a.lr, weight_decay=0.0)

    b0, ws0, wps0, rest0, cc0, pa0 = T.load_tup(a.init)
    zero_idx = [0]
    zero_idx += [SP_OFF[nm] for nm in ['H', 'V', 'D1', 'D2', 'q2']]
    zero_idx += [SP_OFF['envus'] + sq * 729 for sq in range(49)]
    zero_idx += [SP_OFF['envthem'] + sq * 729 for sq in range(49)]
    zero_idx += [SP_OFF['mobus'] + ph * 50 for ph in range(50)]
    zero_idx += [SP_OFF['mobthem'] + ph * 50 for ph in range(50)]
    zero_idx += [SP_OFF['pa'] + pi * 6561 for pi in range(80)]
    zero_idx += [DE_OFF['capnus'], DE_OFF['capnthem'], DE_OFF['convus'], DE_OFF['convthem']]
    sp_idx = [0] + [SP_OFF[nm] for nm in ['H', 'V', 'D1', 'D2', 'q2']]
    sp_idx += [SP_OFF['envus'] + sq * 729 for sq in range(49)]
    sp_idx += [SP_OFF['envthem'] + sq * 729 for sq in range(49)]
    sp_idx += [SP_OFF['mobus'] + ph * 50 for ph in range(50)]
    sp_idx += [SP_OFF['mobthem'] + ph * 50 for ph in range(50)]
    sp_idx += [SP_OFF['pa'] + pi * 6561 for pi in range(80)]
    sp_t = torch.tensor(sp_idx, dtype=torch.long)
    de_t = torch.tensor([DE_OFF['capnus'], DE_OFF['capnthem'], DE_OFF['convus'],
                         DE_OFF['convthem']], dtype=torch.long)

    def zero_constrained():
        with torch.no_grad():
            bags[0].weight.data[sp_t].zero_()
            bags[1].weight.data[de_t].zero_()

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
    print('warm start %s' % a.init, flush=True)

    perm = np.random.permutation(n)
    pos = 0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        if pos + a.batch > n:
            perm = np.random.permutation(n)
            pos = 0
        idx = perm[pos:pos + a.batch]
        pos += a.batch
        o0 = np.array(z['occ0'][idx], dtype=np.uint64)
        o1 = np.array(z['occ1'][idx], dtype=np.uint64)
        bl = np.array(z['blk'][idx], dtype=np.uint64)
        st = np.array(z['stm'][idx], dtype=np.uint8)
        out = np.array(z['out'][idx], dtype=np.float64)

        Wnp = bags[0].weight.detach().squeeze(1).numpy()
        Dnp = bags[1].weight.detach().squeeze(1).numpy()
        dump_tup(a.tmp, Wnp, Dnp, float(bias.item()), active)

        boards = [board_str(int(o0[i]), int(o1[i]), int(bl[i]), int(st[i])) for i in range(len(idx))]
        cmd = [a.exe, 'label', '--nodes', str(a.nodes), '--tuple', a.tmp, '--tuple-native']
        res = subprocess.run(cmd, input=('\n'.join(boards) + '\n').encode(), capture_output=True)
        scores = []
        for line in res.stdout.decode('utf-8', 'replace').splitlines():
            p = line.split()
            if p:
                try:
                    scores.append(float(p[0]))
                except ValueError:
                    pass
        if len(scores) != len(idx):
            print('step %d: search returned %d scores for %d positions' % (step, len(scores), len(idx)), flush=True)
            continue
        sc = np.clip(np.array(scores, dtype=np.float64), -1500, 1500)
        oc = np.clip((out * 2 - 1) * 1000.0, -1500, 1500)
        tgt = ((1.0 - a.outcome_weight) * sc + a.outcome_weight * oc).astype(np.float32)

        feats = S.features_for(o0, o1, bl, st)
        SPc, DEc = S.materialize32(*feats, active)
        del feats
        base = T.base_eval_stm(o0, o1, bl, st).astype(np.float32)
        y = tgt - base

        Xsp = torch.from_numpy(SPc.astype(np.int64))
        Xde = torch.from_numpy(DEc.astype(np.int64))
        yt = torch.from_numpy(y)
        opt.zero_grad()
        pred = bags[0](Xsp).squeeze(1) + bags[1](Xde).squeeze(1) + bias
        err = (pred - yt).detach().abs()
        wgt = 1.0 + torch.clamp(err / 100.0, 0.0, a.regret_cap)
        l2 = sum((bg.weight ** 2).sum() for bg in bags)
        loss = (((pred - yt) ** 2) * wgt).mean() + a.wd * l2
        loss.backward()
        opt.step()
        zero_constrained()
        if step % 25 == 0:
            print('step %d loss %.0f err %.1f mean_w %.2f (%.0fs)' % (
                step, float(loss.item()), float(err.mean()), float(wgt.mean()), time.time() - t0), flush=True)
            with open(a.log, 'a') as f:
                f.write('step %d loss %.0f err %.1f w %.2f\n' % (step, float(loss.item()), float(err.mean()), float(wgt.mean())))
    dump_tup(a.out, bags[0].weight.detach().squeeze(1).numpy(), bags[1].weight.detach().squeeze(1).numpy(),
             float(bias.item()), active)
    print('wrote %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
