"""Own-brain 147x64 trainer: distill net004 -> OUR weights on dag.db data.

Why this file exists: net004 is Ciekce's knowledge (-140 at 5k nodes vs our
classical). We keep his PROVEN-CHEAP arch (147 in, L1=64, 19KB, L1-resident,
swap ~free at 5M NPS) and saturate OUR weights on OUR data.

Data: dag.db nodes(fen, stm, visits, sum_outcome, sum_score, wsum,
  best_score, best_nodes). 8.7M positions, CPU torch (no GPU on box).
Targets: outcome = sum_outcome/visits (fallback: sign(best_score)),
  score = best_score clipped to +/-2000, mapped through the s1 scale.
Init: net004 weights (distill, not random) -> fine-tune.
Export: data/nnue/own_w.s1, same 19202B layout sancta.rs reads.
Stays BLAZING fast: same 147x64 arch, no growth, no LUT needed (19KB L1).

Usage: python3 training/train_own.py --epochs 20 --batch 4096
Resume: checkpoints data/nnue/own_ckpt_e{N}.pt
"""
import argparse
import math
import os
import sqlite3
import struct
import sys
import time

import torch
import torch.nn as nn

L1 = 64
NFEAT = 147
S1N = 19202  # (147*64+64+128+1)*2 bytes, matches sancta.rs from_bytes
S1_NUM = 400
S1_DIV = 255 * 64
CLIP = 255

# ---------------- board parse (matches Board::from_str) ----------------

def parse_fen(fen, stm):
    """Ataxx FEN rows top->bottom: digits=empty run, x/o/# pieces.
    Returns (own49, enemy49, gap49) in OUR sq order (rank*7+file)."""
    occ0, occ1, blk = [0]*49, [0]*49, [0]*49
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
    return occ1, occ0, blk  # stm-relative own/enemy


def sq_idx(sq):
    row, f = divmod(sq, 7)
    return (6 - row) * 7 + f


def featurize(own, enemy, gap):
    """147-bit vector in sancta feature order: own49/enemy49/gap49 by sq_idx."""
    x = [0.0] * NFEAT
    for sq in range(49):
        i = sq_idx(sq)
        if own[sq]:
            x[i] = 1.0
        if enemy[sq]:
            x[49 + i] = 1.0
        if gap[sq]:
            x[98 + i] = 1.0
    return x

# ---------------- model: exact 147x64 + 128->1 sancta arch ----------------

class S1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.Embedding(NFEAT, L1)  # sparse lookup, no dense matmul
        self.out = nn.Linear(2 * L1, 1)

    def forward_sparse(self, idx, stm_sign):
        """idx: (B,K) active feature ids, stm_sign: (B,) +1/-1 unused (wired for later)."""
        B, K = idx.shape
        acc = self.ft.weight.sum(0, keepdim=True).expand(B, L1) * 0  # bias added below
        acc = acc + self.ft_bias.expand(B, L1)
        # sparse add of active columns
        cols = self.ft(idx)  # (B,K,L1)
        acc = acc + cols.sum(1)
        return acc

    @property
    def ft_bias(self):
        return self._bias

    def set_bias(self, b):
        self._bias = nn.Parameter(b)


def load_s1_blob(path):
    d = open(path, 'rb').read()
    assert len(d) == S1N, (len(d), S1N)
    n = len(d) // 2
    w = struct.unpack('<%dh' % n, d)
    o = 0
    ft_w = [w[o:o + L1] for _ in range(NFEAT)]
    o += NFEAT * L1
    ft_b = list(w[o:o + L1])
    o += L1
    out_w = list(w[o:o + 2 * L1])
    o += 2 * L1
    out_b = w[o]
    return ft_w, ft_b, out_w, out_b


def export_s1_blob(path, ft_w, ft_b, out_w, out_b):
    vals = []
    for f in range(NFEAT):
        vals += [int(max(-32768, min(32767, round(v)))) for v in ft_w[f]]
    vals += [int(max(-32768, min(32767, round(v)))) for v in ft_b]
    vals += [int(max(-32768, min(32767, round(v)))) for v in out_w]
    vals += [int(out_b)]
    assert len(vals) == S1N // 2
    with open(path, 'wb') as f:
        f.write(struct.pack('<%dh' % len(vals), *vals))

# ---------------- dataset ----------------

class DagDS(torch.utils.data.Dataset):
    def __init__(self, db, min_visits=4, limit=None):
        con = sqlite3.connect(db)
        con.execute('pragma journal_mode=off')
        q = ('select fen, stm, visits, sum_outcome, best_score from nodes '
             'where visits>=%d' % min_visits)
        if limit:
            q += ' limit %d' % limit
        rows = con.execute(q).fetchall()
        con.close()
        self.rows = rows
        print('dataset rows: %d (visits>=%d)' % (len(rows), min_visits), flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fen, stm, vis, s_out, best = self.rows[i]
        own, enemy, gap = parse_fen(fen, stm)
        x = featurize(own, enemy, gap)
        out_t = s_out / max(1, vis)
        out_t = max(-1.0, min(1.0, out_t))
        if vis < 1:
            out_t = 1.0 if best > 0 else (-1.0 if best < 0 else 0.0)
        sc = max(-2000.0, min(2000.0, best)) / 2000.0
        # active feature ids (sparse)
        ids = [j for j in range(NFEAT) if x[j] > 0.5]
        return torch.tensor(ids, dtype=torch.long), torch.tensor([out_t], dtype=torch.float32), torch.tensor([sc], dtype=torch.float32)


def collate(batch):
    maxk = max(b[0].numel() for b in batch)
    ids = torch.stack([torch.nn.functional.pad(b[0], (0, maxk - b[0].numel()), value=0) for b in batch])
    mask = torch.stack([torch.nn.functional.pad(torch.ones(b[0].numel()), (0, maxk - b[0].numel())) for b in batch])
    out = torch.stack([b[1] for b in batch])
    sc = torch.stack([b[2] for b in batch])
    return ids, mask, out, sc

# ---------------- train ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='data/nnue/dag.db')
    ap.add_argument('--init', default='data/nnue/sancta_w.s1')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--min-visits', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='data/nnue/own_w.s1')
    ap.add_argument('--workers', type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print('threads:', torch.get_num_threads(), flush=True)

    ft_w0, ft_b0, out_w0, out_b0 = load_s1_blob(a.init)
    model = S1()
    with torch.no_grad():
        model.ft.weight.copy_(torch.tensor(ft_w0, dtype=torch.float32))
        model.set_bias(torch.tensor(ft_b0, dtype=torch.float32))
        # out layer: sancta out_w is i16 over clipped acc; init linear to match scale
        # forward: (out_b + clip(acc).out_w) * 400/16320. Fold scale into init.
        model.out.weight.copy_(torch.tensor([out_w0], dtype=torch.float32) * (S1_NUM / S1_DIV))
        model.out.bias.copy_(torch.tensor([out_b0 * (S1_NUM / S1_DIV)], dtype=torch.float32))
    model.train()

    ds = DagDS(a.db, a.min_visits, a.limit or None)
    print('NOTE: dag rows are stm-folded single-view; theirs=zeros ablation first.', flush=True)
    # 90/10 split by hash for stable val
    n = len(ds)
    nval = max(1000, n // 10)
    ntr = n - nval
    tr, va = torch.utils.data.random_split(ds, [ntr, nval], generator=torch.Generator().manual_seed(7))
    trl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=a.workers)
    val = torch.utils.data.DataLoader(va, batch_size=a.batch, collate_fn=collate, num_workers=a.workers)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * max(1, len(trl)))

    def run_epoch(loader, train):
        tot, to, ts, cnt = 0.0, 0.0, 0.0, 0
        for ids, mask, out_t, sc_t in loader:
            B, K = ids.shape
            cols = model.ft(ids) * mask.unsqueeze(-1)  # (B,K,L1)
            acc = cols.sum(1) + model.ft_bias.unsqueeze(0)  # (B,L1)
            cre = acc.clamp(0, CLIP)
            # dual perspective? dag rows are stm-relative already (own/enemy) but
            # single acc only sees OUR side. Theirs = enemy features live in same
            # 147 space with (1-p) offset — our featurize already folded stm, so
            # acc IS ours; theirs approx = zeros (ablation: single-view first).
            # DUAL perspective (matches sancta.rs forward_ready): the 147 space
            # holds BOTH views — ours features [0..64) AND theirs [64..128).
            # featurize folds stm into own/enemy, so: ours cols use own+gap
            # ids, theirs cols use enemy+gap ids. Gap (98..147) goes in BOTH.
            ours_ids, theirs_ids = [], []
            for b in range(B):
                o, t = [], []
                for k in range(K):
                    if mask[b, k] < 0.5:
                        continue
                    f = int(ids[b, k])
                    if f < 49 or f >= 98:
                        o.append(f)   # own + gap
                    if f >= 49:
                        t.append(f)   # enemy + gap
                ours_ids.append(o)
                theirs_ids.append(t)
            # vectorized: build masks
            acc_o = model.ft_bias.unsqueeze(0).expand(B, L1).clone()
            acc_t = model.ft_bias.unsqueeze(0).expand(B, L1).clone()
            for b in range(B):
                if ours_ids[b]:
                    acc_o[b] += model.ft(torch.tensor(ours_ids[b])).sum(0)
                if theirs_ids[b]:
                    # theirs features are (1-p)*49 indexed; map: enemy f in
                    # 49..98 -> theirs-view own-slot? NO — sancta theirs acc
                    # uses SAME feature ids (enemy stones stay enemy feats).
                    acc_t[b] += model.ft(torch.tensor(theirs_ids[b])).sum(0)
            feat = torch.cat([acc_o.clamp(0, CLIP), acc_t.clamp(0, CLIP)], dim=1)
            pred = model.out(feat)  # cp-ish
            pred_out = torch.tanh(pred / 500.0)
            pred_sc = pred / 2000.0
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
            va_loss, va_o, va_s = run_epoch(val, False)
        dt = time.time() - t0
        print('ep %d/%d tr=%.4f(out %.4f sc %.4f) va=%.4f(out %.4f sc %.4f) %.0fs lr=%.4f' % (
            ep, a.epochs, tr_loss, tr_o, tr_s, va_loss, va_o, va_s, dt, sched.get_last_lr()[0]), flush=True)
        # checkpoint export every epoch (gate-able .s1 immediately)
        with torch.no_grad():
            ft_w = model.ft.weight.detach().tolist()
            ft_b = model.ft_bias.detach().tolist()
            ow = (model.out.weight.detach()[0].tolist())
            # unfold scale back to i16 sancta convention
            ow = [v / (S1_NUM / S1_DIV) for v in ow]
            ob = float(model.out.bias.detach()[0]) / (S1_NUM / S1_DIV)
            export_s1_blob(a.out, ft_w, ft_b, ow, ob)
            torch.save(model.state_dict(), 'data/nnue/own_ckpt_e%d.pt' % ep)
        print('exported', a.out, flush=True)


if __name__ == '__main__':
    main()
