import argparse
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_full as T

OFF = {
    'bias': (4, 8),
    'singles': (8, 8 + 2916),
    'dircounts': (8 + 2916, 8 + 2916 + 16),
    'pairs': (8 + 2916 + 16, 8 + 2916 + 16 + 4 * 6561),
}
_p = OFF['pairs'][1]
OFF['q2'] = (_p, _p + 9216 * 2)
_p = OFF['q2'][1]
OFF['envus'] = (_p, _p + 35721 * 2)
_p = OFF['envus'][1]
OFF['envthem'] = (_p, _p + 35721 * 2)
_p = OFF['envthem'][1]
OFF['mobus'] = (_p, _p + 2500 * 2)
_p = OFF['mobus'][1]
OFF['mobthem'] = (_p, _p + 2500 * 2)
_p = OFF['mobthem'][1]
OFF['capnus'] = (_p, _p + 9 * 2)
_p = OFF['capnus'][1]
OFF['capnthem'] = (_p, _p + 9 * 2)
_p = OFF['capnthem'][1]
OFF['convus'] = (_p, _p + 256 * 2)
_p = OFF['convus'][1]
OFF['convthem'] = (_p, _p + 256 * 2)
_p = OFF['convthem'][1]
OFF['pa'] = (_p, _p + 80 * 6561)

assert OFF['pa'][1] == 726440, OFF['pa']

GROUPS = {
    'pa': ['pa'],
    'capconv': ['capnus', 'capnthem', 'convus', 'convthem'],
    'q2': ['q2'],
    'env': ['envus', 'envthem'],
    'mob': ['mobus', 'mobthem'],
    'singles': ['singles'],
    'pairs': ['pairs'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--zero', default='capconv,pa',
                    help='comma list of groups: pa,capconv,q2,env,mob,singles,pairs')
    a = ap.parse_args()
    buf = bytearray(open(a.inp, 'rb').read())
    assert buf[0:4] == b'TUP7' and len(buf) == 726440, (buf[0:4], len(buf))
    groups = [g.strip() for g in a.zero.split(',') if g.strip()]
    for g in groups:
        assert g in GROUPS, g
        for name in GROUPS[g]:
            lo, hi = OFF[name]
            buf[lo:hi] = b'\x00' * (hi - lo)
            print('zeroed %s [%d:%d] %d bytes' % (name, lo, hi, hi - lo))
    open(a.out, 'wb').write(buf)
    b0, ws0, wps0, rest0, cc0, pa0 = T.load_tup(a.out)
    checks = {}
    checks['singles'] = float(np.abs(ws0).sum())
    checks['pairs'] = float(sum(np.abs(p).sum() for p in wps0))
    if rest0 is not None:
        wq, weu, wet, wmu, wmt = rest0
        checks['q2'] = float(np.abs(wq).sum())
        checks['env'] = float(np.abs(weu).sum() + np.abs(wet).sum())
        checks['mob'] = float(np.abs(wmu).sum() + np.abs(wmt).sum())
    if cc0 is not None:
        wcu, wct, wvu, wvt = cc0
        checks['capconv'] = float(np.abs(wcu).sum() + np.abs(wct).sum() + np.abs(wvu).sum() + np.abs(wvt).sum())
    if pa0 is not None:
        checks['pa'] = float(np.abs(pa0).sum())
    print('post-load section L1:', ' '.join('%s %.0f' % (k, v) for k, v in checks.items()))
    for k in ('singles', 'pairs', 'q2', 'env', 'mob', 'capconv', 'pa'):
        if k in checks and k in groups:
            assert checks[k] == 0.0, ('not zero', k, checks[k])
    print('wrote %s %d bytes; groups zeroed: %s' % (a.out, len(buf), ','.join(groups)))


if __name__ == '__main__':
    main()
