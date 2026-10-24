import sys, glob, numpy as np
sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P

KDIR = "C:/gitProjects/titanium-ataxx/data/datasets/KataGo_Ataxx_202604"
rng = np.random.default_rng(21)
fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
rows = []
for f in [fs[i] for i in rng.choice(len(fs), 3, replace=False)]:
    z = np.load(f)
    pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
    gi = z["globalInputNC"]
    q = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
    emp = 49 - pl[:, 1].sum(1) - pl[:, 2].sum(1)
    ok = np.nonzero((gi[:, 0] == 0) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0) & (emp >= 14) & (emp <= 30))[0]
    for i in rng.choice(ok, 400, replace=False):
        rows.append((pl[i], q[i] / q[i].sum()))

def bb(v):
    return int(sum(int(v[s]) << s for s in range(49)))

def conv(us, en, s):
    return bin(en & int(P.R1[s])).count("1")

def recapture(us, en, s, origin=None):
    full = int(P.FULL)
    u = us & ~(1 << origin) if origin is not None else us
    c = en & int(P.R1[s])
    u2 = u | (1 << s) | c
    e2 = en & ~c
    emp2 = ~(u2 | e2) & full
    le = 0
    x = e2
    while x:
        b = x & -x
        k = b.bit_length() - 1
        le |= int(P.R1[k]) | int(P.R2[k])
        x ^= b
    le &= emp2
    best, bs, y = 0, -1, le
    while y:
        b = y & -y
        k = b.bit_length() - 1
        n = bin(u2 & int(P.R1[k])).count("1")
        if n > best:
            best, bs = n, k
        y ^= b
    return bin(c).count("1"), best, bs

shown = 0
stats = []
for pl, q in rows:
    us, en = bb(pl[1]), bb(pl[2])
    empty = ~(us | en) & int(P.FULL)
    clones = [s for s in range(49) if (empty >> s) & 1 and (int(P.R1[s]) & us)]
    origins = [s for s in range(49) if (us >> s) & 1 and (int(P.R2[s]) & empty)]
    if not clones:
        continue
    top = int(q[:49].argmax())
    is_clone = top in clones
    info = {s: (conv(us, en, s),) + recapture(us, en, s)[1:] for s in clones}
    greedy = max(clones, key=lambda s: (info[s][0], -info[s][1]))
    net2 = {s: 1 + info[s][0] - info[s][1] for s in clones}
    best2 = max(clones, key=lambda s: net2[s])
    stats.append((is_clone, top == greedy, top == best2, q[top], len(clones), len(origins), q[origins].sum() if origins else 0))
    # show interesting cases: Kata plays a clone that is NOT the greedy one, or prefers a jump origin while a good clone exists
    interesting = (is_clone and top != greedy and net2[top] > net2[greedy]) or ((not is_clone) and max(net2.values()) >= 2)
    if interesting and shown < 8:
        shown += 1
        print("=" * 60)
        print("case %d | Kata search top = %s (mass %.2f) | greedy clone = %s | clone squares %d, jump origins %d, mass on origins %.2f" % (
            shown, "CLONE" if is_clone else "JUMP-ORIGIN", q[top], "r%dc%d" % divmod(greedy, 7), len(clones), len(origins), q[origins].sum()))
        for r in range(7):
            line = ""
            for c in range(7):
                s = r * 7 + c
                if s == top:
                    ch = "T"
                elif s == greedy:
                    ch = "G"
                elif (us >> s) & 1:
                    ch = "M"
                elif (en >> s) & 1:
                    ch = "E"
                else:
                    ch = "."
                if s == top and s == greedy:
                    ch = "B"
                line += ch + " "
            print("   " + line + "    mass row: " + " ".join("%2d" % int(round(100 * q[r * 7 + c])) for c in range(7)))
        print("   T=Kata top, G=greedy(max conversions), B=both, M=mover stones, E=enemy stones")
        for nm, s in (("Kata top", top), ("greedy", greedy), ("best 2-ply clone", best2)):
            if s in info:
                print("   %-16s r%dc%d: converts %d, enemy best recapture %d, 2-ply net %+d, Kata mass %.2f" % (nm, s // 7, s % 7, info[s][0], info[s][1], net2[s], q[s]))
            else:
                print("   %-16s r%dc%d is a jump-origin pick (mass %.2f)" % (nm, s // 7, s % 7, q[s]))
S = np.array(stats, float)
print("\nOver %d stage-0 positions: Kata top is a clone %.0f%%; of those, equals greedy %.0f%%, equals best 2-ply %.0f%%" % (
    len(S), 100 * S[:, 0].mean(), 100 * S[S[:, 0] == 1, 1].mean(), 100 * S[S[:, 0] == 1, 2].mean()))
