# D1598 analysis only (no training): check proof-line rows against themselves and against Titanium's own proven labels.
import struct, sys
def rows(p):
    d = open(p, 'rb').read()
    for i in range(0, len(d) - len(d) % 48, 48):
        o0, o1, out, cp = struct.unpack_from('<QQff', d, i)
        stm = d[i + 40]
        me, op = (o0, o1) if stm == 0 else (o1, o0)
        yield (me, op), out, cp
lines, orig = sys.argv[1], sys.argv[2]
L = list(rows(lines))
bad = sum(1 for _, o, c in L if not ((c >= 98000 and o == 1.0) or (c <= -98000 and o == 0.0)))
keys = {}
dup_conf = 0
for k, o, c in L:
    if k in keys and keys[k] != o:
        dup_conf += 1
    keys[k] = o
print(f"proof-line rows {len(L)}, distinct positions {len(keys)}, outcome disagrees with the proven cp sign: {bad}, same position with opposite results: {dup_conf}")
agree = dis = only_ti = 0
O = list(rows(orig))
ti_proven = {k: (1.0 if c > 0 else 0.0) for k, o, c in O if abs(c) >= 98000}
for k, v in ti_proven.items():
    if k in keys:
        if keys[k] == v: agree += 1
        else: dis += 1
    else:
        only_ti += 1
game_out = {k: o for k, o, c in O}
g_agree = sum(1 for k in keys if k in game_out and game_out[k] == keys[k])
g_tot = sum(1 for k in keys if k in game_out)
print(f"Titanium-proven positions in the original games: {len(ti_proven)}; also proven by Zataxx: agree {agree}, DISAGREE {dis}; proven by Titanium only: {only_ti}")
print(f"proof-line positions that are also positions of the original games: {g_tot}; game result agrees with the proof on {g_agree}")
print(f"NEW positions (not in the original games): {len(keys) - g_tot}")
