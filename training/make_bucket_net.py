import argparse
import re
import struct


def parse_table(path):
    vals = {}
    started = False
    for line in open(path, encoding='utf-8', errors='replace'):
        if '== teacher residual' in line:
            started = True
            continue
        if started and line.startswith('buckets'):
            break
        if started:
            m = re.match(r'\s*(\d+)\s+(\d+)\s+([+-]?\d+\.?\d*)', line)
            if m:
                pc = int(m.group(1))
                n = int(m.group(2))
                mean = float(m.group(3))
                vals[pc] = (n, mean)
    if not vals:
        raise SystemExit('no teacher table found in ' + path)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='logs/bucket_audit2.log')
    ap.add_argument('--tup', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--out', default='data/nnue/tonightBucket.tup')
    ap.add_argument('--min-n', type=int, default=150)
    ap.add_argument('--zero-heads', action='store_true', default=True)
    a = ap.parse_args()

    vals = parse_table(a.log)
    table = [0] * 50
    last = 0
    for pc in range(1, 50):
        n, mean = vals.get(pc, (0, None))
        if mean is not None and n >= a.min_n:
            table[pc] = int(round(max(-32767, min(32767, mean))))
            last = table[pc]
        else:
            table[pc] = last
    print('bucket table:', ' '.join(str(x) for x in table[1:]))
    buf = bytearray(open(a.tup, 'rb').read())
    assert buf[0:4] == b'TUP7' and len(buf) == 726440, (buf[0:4], len(buf))
    if a.zero_heads:
        for lo, hi in ((200500, 201560), (29184, 47616)):
            buf[lo:hi] = b'\x00' * (hi - lo)
    payload = bytes(bytearray(struct.pack('<%dh' % 50, *table)))
    out = bytes(buf[0:4]).replace(b'TUP7', b'TUP8') + bytes(buf[4:]) + payload
    open(a.out, 'wb').write(out)
    print('wrote %s %d bytes' % (a.out, len(out)))


if __name__ == '__main__':
    main()
