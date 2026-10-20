import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T


def sections(path):
    b0, ws, wp, rest, cc, pa = T.load_tup(path)
    d = {}
    d['singles'] = ws.reshape(-1).astype(np.float64)
    for i, nm in enumerate(['pairsH', 'pairsV', 'pairsD1', 'pairsD2']):
        d[nm] = wp[i].reshape(-1).astype(np.float64)
    if rest is not None:
        d['q2'] = rest[0].reshape(-1).astype(np.float64)
        d['envus'] = rest[1].reshape(-1).astype(np.float64)
        d['envthem'] = rest[2].reshape(-1).astype(np.float64)
        d['mobus'] = rest[3].reshape(-1).astype(np.float64)
        d['mobthem'] = rest[4].reshape(-1).astype(np.float64)
    if cc is not None:
        d['capnus'] = cc[0].reshape(-1).astype(np.float64)
        d['capnthem'] = cc[1].reshape(-1).astype(np.float64)
        d['convus'] = cc[2].reshape(-1).astype(np.float64)
        d['convthem'] = cc[3].reshape(-1).astype(np.float64)
    if pa is not None:
        d['pa'] = pa.reshape(-1).astype(np.float64)
    return float(b0), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    a = ap.parse_args()
    ba, A = sections(a.a)
    bb, B = sections(a.b)
    print('bias %+.1f -> %+.1f' % (ba, bb))
    print('%-10s %8s %8s %8s %8s %8s' % ('section', 'nz_a', 'nz_b', 'L1_a', 'L1_b', 'cos'))
    for k in A:
        x, y = A[k], B[k]
        nza = float((x != 0).mean())
        nzb = float((y != 0).mean())
        l1a = float(np.abs(x).sum())
        l1b = float(np.abs(y).sum())
        xu = x - x.mean()
        yu = y - y.mean()
        den = float(np.linalg.norm(xu) * np.linalg.norm(yu))
        cos = float(np.dot(xu, yu) / den) if den > 0 else 0.0
        print('%-10s %7.1f%% %7.1f%% %8.0f %8.0f %+8.4f' % (k, 100 * nza, 100 * nzb, l1a, l1b, cos))
    ma = max(np.abs(A[k]).max() for k in A)
    mb = max(np.abs(B[k]).max() for k in B)
    print('max|w| %.1f -> %.1f' % (ma, mb))


if __name__ == '__main__':
    main()
