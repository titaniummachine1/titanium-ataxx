import sys, glob, numpy as np
sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import probe_concepts as P

KDIR = "C:/gitProjects/titanium-ataxx/data/datasets/KataGo_Ataxx_202604"
FULL = int(P.FULL)
R1 = [int(x) for x in P.R1]
R2 = [int(x) for x in P.R2]
pcnt = lambda x: bin(x).count("1")

def bb(v):
    return int(sum(int(v[s]) << s for s in range(49)))

def recapture(us, en, s, origin=None):
    u = us & ~(1 << origin) if origin is not None else us
    c = en & R1[s]
    u2 = u | (1 << s) | c
    e2 = en & ~c
    emp2 = ~(u2 | e2) & FULL
    le, x = 0, e2
    while x:
        b = x & -x
        k = b.bit_length() - 1
        le |= R1[k] | R2[k]
        x ^= b
    le &= emp2
    best, y = 0, le
    while y:
        b = y & -y
        k = b.bit_length() - 1
        best = max(best, pcnt(u2 & R1[k]))
        y ^= b
    return pcnt(c), best

rng = np.random.default_rng(7)
fs = sorted(glob.glob(KDIR + "/*/tdata/*.npz"))
pairs = []
for f in [fs[i] for i in rng.choice(len(fs), 6, replace=False)]:
    z = np.load(f)
    pl = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :49]
    gi = z["globalInputNC"]
    q = z["policyTargetsNCMove"][:, 0, :].astype(np.float64)
    emp = 49 - pl[:, 1].sum(1) - pl[:, 2].sum(1)
    st0 = {}
    for i in np.nonzero((gi[:, 0] == 0) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0) & (emp >= 12) & (emp <= 32))[0]:
        st0[(bb(pl[i, 1]), bb(pl[i, 2]))] = i
    for j in np.nonzero((gi[:, 0] == 1) & (q.sum(1) > 0) & (pl[:, 3].sum(1) == 0))[0]:
        key = (bb(pl[j, 1]), bb(pl[j, 2]))
        if key in st0 and pl[j, 4].sum() == 1:
            i = st0[key]
            pairs.append((pl[i], q[i] / q[i].sum(), int(np.nonzero(pl[j, 4])[0][0]), q[j] / q[j].sum()))
print("paired stage-0/stage-1 positions:", len(pairs))

rows = []
for pl, q0, org, q1 in pairs:
    us, en = bb(pl[1]), bb(pl[2])
    empty = ~(us | en) & FULL
    clones = [s for s in range(49) if (empty >> s) & 1 and (R1[s] & us)]
    top = int(q0[:49].argmax())
    if top not in range(49) or not ((us >> top) & 1):
        continue                                   # only positions where Kata's top first step is a JUMP-ORIGIN pick
    dest = int(q1[:49].argmax())
    c_d, rec_d = recapture(us, en, dest, top)
    best_clone = max(((1 + recapture(us, en, s)[0] - recapture(us, en, s)[1]), s) for s in clones) if clones else (-9, -1)
    # was the origin stone attackable (enemy can land next to it)?
    le = 0
    x = en
    while x:
        b = x & -x
        k = b.bit_length() - 1
        le |= R1[k] | R2[k]
        x ^= b
    le &= empty
    origin_exposed = bool(R1[top] & le)
    dest_empty_nb = pcnt(R1[dest] & empty)
    dest_filled_nb = pcnt(R1[dest] & (us | en))
    org_empty_nb = pcnt(R1[top] & empty)
    # does the vacated origin become a hole next to enemy-reachable squares?
    rows.append(dict(conv=c_d, rec=rec_d, net=1 + c_d - rec_d - 1, bc_net=best_clone[0], origin_exposed=origin_exposed,
                     dest_filled=dest_filled_nb, org_empty=org_empty_nb, mass0=q0[top], mass1=q1[dest], pl=pl, top=top, dest=dest, bc=best_clone[1]))
print("positions where Kata's top first step is a jump:", len(rows))
conv = np.array([r["conv"] for r in rows]); rec = np.array([r["rec"] for r in rows])
net = conv - rec
bcn = np.array([r["bc_net"] for r in rows])
print("jump destination: converts mean %.2f (clone best-2ply net mean %.2f), enemy recapture mean %.2f, jump net(conv-rec) mean %.2f" % (conv.mean(), bcn.mean(), rec.mean(), net.mean()))
print("jump beats best clone's 2-ply net (jump 1+conv-rec vs clone): %.0f%% ; equal %.0f%% ; worse %.0f%%" % (
    100 * ((1 + net) > bcn).mean(), 100 * ((1 + net) == bcn).mean(), 100 * ((1 + net) < bcn).mean()))
print("destination converts 0 stones: %.0f%% ; 1-2: %.0f%% ; 3+: %.0f%%" % (100 * (conv == 0).mean(), 100 * ((conv >= 1) & (conv <= 2)).mean(), 100 * (conv >= 3).mean()))
print("origin stone was attackable (enemy could land next to it): %.0f%%  (baseline for a random own stone: see below)" % (100 * np.mean([r["origin_exposed"] for r in rows])))
df = np.array([r["dest_filled"] for r in rows])
print("filled neighbours around the destination (0-8): " + " ".join("%d:%.0f%%" % (k, 100 * (df == k).mean()) for k in range(9)))
# baseline: random own stone that has a jump target
base = []
for pl, q0, org, q1 in pairs:
    us, en = bb(pl[1]), bb(pl[2])
    empty = ~(us | en) & FULL
    le, x = 0, en
    while x:
        b = x & -x
        k = b.bit_length() - 1
        le |= R1[k] | R2[k]
        x ^= b
    le &= empty
    for s in range(49):
        if (us >> s) & 1 and (R2[s] & empty):
            base.append(bool(R1[s] & le))
print("baseline: share of own jump-capable stones that are attackable: %.0f%%" % (100 * np.mean(base)))
print()
for k, r in enumerate(rows[:8]):
    pl = r["pl"]; us, en = bb(pl[1]), bb(pl[2])
    print("-" * 52)
    print("jump r%dc%d -> r%dc%d (mass %.2f x %.2f) | converts %d, enemy recaptures %d | best clone net %+d (r%dc%d)" % (
        r["top"] // 7, r["top"] % 7, r["dest"] // 7, r["dest"] % 7, r["mass0"], r["mass1"], r["conv"], r["rec"], r["bc_net"], r["bc"] // 7, r["bc"] % 7))
    for rr in range(7):
        line = ""
        for c in range(7):
            s = rr * 7 + c
            ch = "M" if (us >> s) & 1 else ("E" if (en >> s) & 1 else ".")
            if s == r["top"]:
                ch = "F"          # from
            if s == r["dest"]:
                ch = "D"          # destination
            line += ch + " "
        print("   " + line)
    print("   F=jump origin (vacated)  D=destination  M mover  E enemy")
