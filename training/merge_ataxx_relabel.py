"""Merge Ataxx relabel chunks into ONE final-dataset split (new file; original untouched).

SAFETY: row ids in relabel chunks are per-source-file. This script therefore
verifies the relabel manifest's `cache` matches the split being merged and
refuses to merge otherwise. Run one split at a time (train/val/test each get
their own relabel dir).

For every row with teacher_src==1 that has a relabel record:
  teacher      = q * 1000            (cp-scale, same convention as Ataxx rows)
  teacher_p    = p_win + 0.5*p_draw   (expected score, same semantics as outcome_p)
  teacher_src  unchanged             (dirs drive the sidecar feature layout: the
                                       20.66M rows stay in dir 1 = KataGo position
                                       provenance; which teacher scored them is
                                       recorded in the ataxx_* columns)
  ataxx_p_win/draw/loss, ataxx_ply    (provenance columns)

Row order and count preserved EXACTLY -> existing SP/DE and base sidecars stay
valid (position-keyed).
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
FINAL = os.path.join(ROOT, "data", "nnue", "final")


def load_relabel(relabel_dir):
    files = sorted(glob.glob(os.path.join(relabel_dir, "ataxx_c*.npz")))
    if not files:
        raise SystemExit("no relabel chunks under %s" % relabel_dir)
    row, q, pw, pd, pl, ply = [], [], [], [], [], []
    for f in files:
        z = np.load(f)
        row.append(z['row'])
        q.append(z['q'])
        pw.append(z['p_win'])
        pd.append(z['p_draw'])
        pl.append(z['p_loss'])
        ply.append(z['ply'])
    cat = lambda xs: np.concatenate(xs)
    r = cat(row)
    order = np.argsort(r)
    return tuple(x[order] for x in (r, cat(q), cat(pw), cat(pd), cat(pl), cat(ply)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--relabel-dir', required=True)
    ap.add_argument('--suffix', default='_v2')
    a = ap.parse_args()

    src = os.path.join(FINAL, 'final_%s.npz' % a.split)
    man_path = os.path.join(a.relabel_dir, 'manifest.json')
    if not os.path.exists(man_path):
        raise SystemExit('missing manifest %s' % man_path)
    man = json.load(open(man_path))
    if os.path.abspath(man.get('cache', '')) != os.path.abspath(src):
        raise SystemExit('REFUSING: relabel cache %r != split %r'
                         % (man.get('cache'), src))

    rr, rq, rpw, rpd, rpl, rply = load_relabel(a.relabel_dir)
    if len(np.unique(rr)) != len(rr):
        raise SystemExit('duplicate rows in relabel chunks')

    z = np.load(src)
    data = {k: z[k] for k in z.files}
    n = len(data['occ0'])
    ts = data['teacher_src'].copy()
    teacher = data['teacher'].astype(np.float32).copy()
    tprob = data['teacher_p'].astype(np.float32).copy()
    pw = np.full(n, np.nan, np.float32)
    pd = np.full(n, np.nan, np.float32)
    pl = np.full(n, np.nan, np.float32)
    ply = np.full(n, -1, np.int16)

    pos = np.searchsorted(rr, np.arange(n))
    inb = pos < len(rr)
    valid = np.zeros(n, bool)
    valid[inb] = rr[pos[inb]] == np.arange(n)[inb]
    valid &= (ts == 1)  # only outcome-only rows get overwritten
    idx = np.where(valid)[0]
    rpos = pos[idx]
    teacher[idx] = rq[rpos] * 1000.0
    tprob[idx] = rpw[rpos] + 0.5 * rpd[rpos]
    pw[idx] = rpw[rpos]
    pd[idx] = rpd[rpos]
    pl[idx] = rpl[rpos]
    ply[idx] = rply[rpos]

    out = dict(data)
    out['teacher'] = teacher
    out['teacher_p'] = tprob
    out['teacher_src'] = ts
    out['ataxx_p_win'] = pw
    out['ataxx_p_draw'] = pd
    out['ataxx_p_loss'] = pl
    out['ataxx_ply'] = ply
    dst = os.path.join(FINAL, 'final_%s%s.npz' % (a.split, a.suffix))
    tmp = dst + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, dst)

    mix = {int(k2): int((ts == k2).sum()) for k2 in np.unique(ts)}
    print('%s: %d/%d rows relabeled; src mix %s' % (a.split, len(idx), n, mix))
    print('  teacher_p mean %.4f; corr(teacher_p,outcome_p) on relabeled %.3f'
          % (float(tprob.mean()),
             float(np.corrcoef(tprob[idx], data['outcome_p'][idx])[0, 1]) if len(idx) > 1 else float('nan')))
    print('  wrote %s' % dst)


if __name__ == '__main__':
    main()
