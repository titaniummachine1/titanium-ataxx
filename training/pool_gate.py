import glob
import math
import random
import re

pat = re.compile(r"^-- game \d+: book (\d+) titanium=(\w) opp=(\w): (titanium|opponent) wins (\d+)-(\d+) \((\d+) plies\)")
draw = re.compile(r"^-- game \d+: book (\d+) titanium=(\w) opp=(\w): draw (\d+)-(\d+) \((\d+) plies\)")

pairs = []
files = sorted(glob.glob('logs/gate_v2_ep8_p*.txt'))
for f in files:
    by_line = {}
    for line in open(f, encoding='utf-8', errors='ignore'):
        line = line.strip()
        m = pat.match(line)
        if m:
            by_line.setdefault(int(m.group(1)), []).append(1.0 if m.group(4) == 'titanium' else 0.0)
            continue
        m = draw.match(line)
        if m:
            by_line.setdefault(int(m.group(1)), []).append(0.5)
    for ln, rs in sorted(by_line.items()):
        if len(rs) != 2:
            print('WARN %s book %d has %d games' % (f, ln, len(rs)))
        pairs.append(sum(rs) / len(rs))

n = len(pairs)
p = sum(pairs) / n
print('pairs %d  pair-score mean %.4f  (game score %.4f)' % (n, p, p))
random.seed(1)
bs = []
for _ in range(20000):
    s = sum(pairs[random.randrange(n)] for _ in range(n)) / n
    bs.append(s)
bs.sort()
lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]
elo = lambda x: -400.0 * math.log10(1.0 / x - 1.0)
print('paired-bootstrap 95%% scoreCI [%.4f, %.4f]  elo %+.0f [%+.0f, %+.0f]' % (lo, hi, elo(p), elo(lo), elo(hi)))
w = sum(1 for x in pairs if x == 1.0)
l = sum(1 for x in pairs if x == 0.0)
print('pairs won/lost/split %d/%d/%d' % (w, l, n - w - l))
print('VERDICT:', 'PASS (CI lower > 0.5)' if lo > 0.5 else 'INCONCLUSIVE -> EXTEND')
