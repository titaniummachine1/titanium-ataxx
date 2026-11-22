# D1602 test helper (analysis only): print the hub pos_key (gamedata::pos_key) of row N of a .tdat
import struct, sys
M = (1 << 64) - 1
def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
    return x ^ (x >> 31)
d = open(sys.argv[1], 'rb').read()
n = int(sys.argv[2])
o0, o1 = struct.unpack_from('<QQ', d, n * 48)
stm = d[n * 48 + 40]
m, o = (o0, o1) if stm == 0 else (o1, o0)
rot = ((o << 17) | (o >> 47)) & M
print(f"{splitmix(splitmix(m) ^ rot):016x}")
