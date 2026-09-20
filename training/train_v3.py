"""Own-brain v3 trainer: 147x64 FT (incremental, unchanged) + 128->16->1 head.

Why v3: the 128->1 linear head can't express "fine unless X" interactions —
both own-net v1+v2 gated 0-30 with val improving (capacity floor, not data
floor). H2=16 adds ~2k MAC to the forward (~17ns -> ~25ns est, still noise
vs the FT refresh); the incremental FT path is byte-identical and stays
blazing fast. FT init = net004 distill; head = random (v1 math has no exact
v3 mapping — relu placement differs — so distill the KNOWLEDGE via targets,
not the weights).

Data: dag.db nodes(fen, stm, visits, sum_outcome, sum_score, wsum, wout,
  best_score, best_nodes, margin_n). Full-data: wsum-weighted sampling over
  ALL rows (margin_n>0 = 7.6M carry signal), NOT visits>=4 (34k thin slice).
  Targets dual: outcome = sum_outcome/visits + score = best_score/2000.
  Score-only ablation available (--score-only).
Export: data/nnue/own_v3.s1, S1N_V3 layout (26KB) sancta.rs reads.
Gate: every epoch exports a gate-able .s1 immediately.

Usage: python3 training/train_v3.py --epochs 25 --batch 2048
Smoke: python3 training/train_v3.py --epochs 1 --batch 1024 --limit 8000 --out data/nnue/v3_smoke.s1
"""
import argparse
import os
import sqlite3
import struct
import time

import torch
import torch.nn as nn

L1 = 64
H2 = 16
NFEAT = 147
S1N_V3 = (147 * 64 + 64 + 128 * H2 + H2 + H2 + 1) * 2
S1N_V1 = (147 * 64 + 64 + 128 + 1) * 2
S1_NUM = 400
S1_DIV = 255 * 64
CLIP = 255

# ---------------- board parse (matches Board::from_str) ----------------

def parse_fen(fen, stm):
    """Ataxx FEN rows top->bottom: digits=empty run, x/o/# pieces."""
    occ0, occ1, blk = [0] * 49, [0] * 49, [0] * 49
    cells = 0
    for c in fen.strip().split()[0]:
        if c == '/':
            continue
        if c.isdigit():
            cells += int(c)
        elif c in '.xo#':
            if c == 'x':
                occ0[cells] = 1
            elif c == 'o':
                occ1[cells] = 1
            elif c == '#':
                blk[cells] = 1
            cells += 1
    assert cells == 49, (fen, cells)
    if stm == 0:
        return occ0, occ1, blk
    return occ1, occ0, blk


def sq_idx(sq):
    row, f = divmod(sq, 7)
    return (6 - row) * 7 + f


def active_ids(own, enemy, gap):
    """(ours_ids, theirs_ids): ours = own+gap feats, theirs = enemy+gap feats."""
    ours, theirs = [], []
    for sq in range(49):
        i = sq_idx(sq)
        if own[sq]:
            ours.append(i)
        if enemy[sq]:
            theirs.append(49 + i)
        if gap[sq]:
            ours.append(98 + i)
            theirs.append(98 + i)
    return ours, theirs

# ---------------- model: 147x64 FT + 128->16->1 head ----------------

class V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.Embedding(NFEAT, L1)
        self._bias = nn.Parameter(torch.zeros(L1))
        self.w1 = nn.Linear(2 * L1, H2)
        self.w2 = nn.Linear(H2, 1)

    def forward_dual(self, ours_ids, theirs_ids):
        """Batched manual gather (CPU-friendly): list-of-lists active ids."""
        B = len(ours_ids)
        acc_o = self._bias.unsqueeze(0).expand(B, L1).clone()
        acc_t = self._bias.unsqueeze(0).expand(B, L1).clone()
        for b in range(B):
            if ours_ids[b]:
                acc_o[b] += self.ft(torch.tensor(ours_ids[b])).sum(0)
            if theirs_ids[b]:
                acc_t[b] += self.ft(torch.tensor(theirs_ids[b])).sum(0)
        feat = torch.cat([acc_o.clamp(0, CLIP), acc_t.clamp(0, CLIP)], dim=1)
        h = torch.relu(self.w1(feat))
        return self.w2(h)


def load_v1_blob(path):
    d = open(path, 'rb').read()
    assert len(d) == S1N_V1, (len(d), S1N_V1)
    w = struct.unpack('<%dh' % (len(d) // 2), d)
    o = 0
    ft_w = [list(w[o:o + L1]) for _ in range(NFEAT)]
    o += NFEAT * L1
    ft_b = list(w[o:o + L1])
    return ft_w, ft_b


def export_v3_blob(path, ft_w, ft_b, w1, b1, w2, b2):
    """Quant: FT/bias i16 direct; W1 i16 direct; b1 i32 = f32*256;
    w2 i16 direct; b2 i32 = f32*256. Matches sancta.rs from_bytes."""
    vals = []
    for f in range(NFEAT):
        vals += [int(max(-32768, min(32767, round(v)))) for v in ft_w[f]]
    vals += [int(max(-32768, min(32767, round(v)))) for v in ft_b]
    for i in range(2 * L1):
        vals += [int(max(-32768, min(32767, round(v)))) for v in w1[i]]
    b1q = [int(v * 256) for v in b1]
    vals += b1q  # stored i16 range check below
    vals += [int(max(-32768, min(32767, round(v)))) for v in w2]
    vals += [int(b2 * 256)]
    assert len(vals) == S1N_V3 // 2, len(vals)
    with open(path, 'wb') as f:
        f.write(struct.pack('<%dh' % len(vals), *[max(-32768, min(32767, v)) for v in vals]))

# ---------------- dataset: full-data wsum-weighted ----------------

class DagDS(torch.utils.data.Dataset):
    def __init__(self, db, limit=None, score_only=False, min_margin=True):
        con = sqlite3.connect(db)
        con.execute('pragma journal_mode=off')
        q = ('select fen, stm, visits, sum_outcome, best_score, wsum from nodes '
             'where margin_n>0' if min_margin else
             'select fen, stm, visits, sum_outcome, best_score, wsum from nodes')
        if limit:
            q += ' limit %d' % limit
        rows = con.execute(q).fetchall()
        con.close()
        self.rows = rows
        self.score_only = score_only
        # sampling weights: wsum (search mass), fallback visits+1
        import math
        self.w = [max(1.0, r[5] if r[5] and r[5] > 0 else (r[2] + 1)) for r in rows]
        print('dataset rows: %d (margin>0)' % len(rows), flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fen, stm, vis, s_out, best, _ = self.rows[i]
        own, enemy, gap = parse_fen(fen, stm)
        o_ids, t_ids = active_ids(own, enemy, gap)
        out_t = s_out / max(1, vis)
        out_t = max(-1.0, min(1.0, out_t))
        sc = max(-2000.0, min(2000.0, best)) / 2000.0
        return o_ids, t_ids, torch.tensor([out_t]), torch.tensor([sc])

# ---------------- train ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/nnue/dag.db')
    ap.add_argument('--init', default='data/nnue/sancta_w.s1')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--out', default='data/nnue/own_v3.s1')
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print('threads:', torch.get_num_threads(), flush=True)

    ft_w0, ft_b0 = load_v1_blob(a.init)
    model = V3()
    with torch.no_grad():
        model.ft.weight.copy_(torch.tensor(ft_w0, dtype=torch.float32))
        model._bias.copy_(torch.tensor(ft_b0, dtype=torch.float32))
        # head random (Kaiming): v1 linear weights have no exact v3 mapping
        nn.init.kaiming_uniform_(model.w1.weight, nonlinearity='relu')
        nn.init.zeros_(model.w1.bias)
        nn.init.kaiming_uniform_(model.w2.weight, nonlinearity='linear')
        nn.init.zeros_(model.w2.bias)
    model.train()

    ds = DagDS(a.db, a.limit or None, a.score_only)
    n = len(ds)
    nval = max(2000, n // 20)  # 5% val on full data (still huge)
    ntr = n - nval
    tr, va = torch.utils.data.random_split(ds, [ntr, nval], generator=torch.Generator().manual_seed(7))
    # weighted sampler: wsum mass, not uniform over thin rows
    w = torch.tensor(ds.w, dtype=torch.double)
    samp = torch.utils.data.WeightedRandomSampler(w[tr.indices], num_samples=len(tr.indices), replacement=True)
    trl = torch.utils.data.DataLoader(tr, batch_size=a.batch, sampler=samp,
                                      collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b],
                                                             torch.stack([x[2] for x in b]),
                                                             torch.stack([x[3] for x in b])))
    vall = torch.utils.data.DataLoader(va, batch_size=a.batch,
                                       collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b],
                                                              torch.stack([x[2] for x in b]),
                                                              torch.stack([x[3] for x in b])))

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * max(1, len(trl)))

    def run_epoch(loader, train):
        tot, to, ts, cnt = 0.0, 0.0, 0.0, 0
        for o_ids, t_ids, out_t, sc_t in loader:
            B = len(o_ids)
            pred = model.forward_dual(o_ids, t_ids)
            pred_out = torch.tanh(pred / 500.0)
            pred_sc = pred / 2000.0
            if a.score_only:
                loss = (pred_sc - sc_t).pow(2).mean()
            else:
                loss = (pred_out - out_t).pow(2).mean() + 0.5 * (pred_sc - sc_t).pow(2).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            tot += loss.item() * B
            to += (pred_out - out_t).pow(2).sum().item()
            ts += (pred_sc - sc_t).pow(2).sum().item()
            cnt += B
        return tot / cnt, to / cnt, ts / cnt

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_o, tr_s = run_epoch(trl, True)
        with torch.no_grad():
            va_loss, va_o, va_s = run_epoch(vall, False)
        dt = time.time() - t0
        print('ep %d/%d tr=%.4f(out %.4f sc %.4f) va=%.4f(out %.4f sc %.4f) %.0fs lr=%.4f' % (
            ep, a.epochs, tr_loss, tr_o, tr_s, va_loss, va_o, va_s, dt, sched.get_last_lr()[0]), flush=True)
        with torch.no_grad():
            ft_w = model.ft.weight.detach().tolist()
            ft_b = model._bias.detach().tolist()
            # w1: Linear(256,16) weight is [16][256]; export row-major [256][16]
            w1t = model.w1.weight.detach().t().tolist()
            b1 = model.w1.bias.detach().tolist()
            w2 = model.w2.weight.detach()[0].tolist()
            b2 = float(model.w2.bias.detach()[0])
            # unfold output scale: trainer predicts cp directly; sancta
            # applies *400/16320 at the end, so divide weights by that scale
            s = S1_NUM / S1_DIV
            w1t = [[v / s for v in row] for row in w1t]
            b1 = [v / s for v in b1]
            w2 = [v / s for v in w2]
            b2 = b2 / s
            export_v3_blob(a.out, ft_w, ft_b, w1t, b1, w2, b2)
            torch.save(model.state_dict(), 'data/nnue/v3_ckpt_e%d.pt' % ep)
        print('exported', a.out, flush=True)


if __name__ == '__main__':
    main()
