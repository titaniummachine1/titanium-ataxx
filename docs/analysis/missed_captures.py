"""Missed big captures (user bug report D427): replay ti sprt games.jsonl and list moves where the mover could have
flipped >= MIN_FLIPS enemy stones but played a move that flipped far fewer, and whether the opponent then removed the chance
(the square is no longer available with that many flips on the mover's next turn).

    python -I docs/analysis/missed_captures.py <games.jsonl> [...] [--who baseline|candidate|any] [--min 6] [--gap 4]

Prints per engine: moves scanned, missed-capture count and rate, and the first examples (FEN, played move, best capture,
engine score). Classic Ataxx rules (7x7, clone = distance 1, jump = distance 2, flips = enemy stones adjacent to the target).
"""
import json
import sys

N = 7


def parse_fen(fen):
    rows, side = fen.split()[0], fen.split()[1]
    b = {}
    for r, row in enumerate(rows.split("/")):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                b[(r, c)] = ch  # x, o or '-' blocker
                c += 1
    return b, side


def sq(name):
    return (7 - int(name[1]), ord(name[0]) - 97)


def nb(p, d):
    r, c = p
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if max(abs(dr), abs(dc)) == d and 0 <= r + dr < N and 0 <= c + dc < N:
                yield (r + dr, c + dc)


def flips(b, to, side):
    enemy = "o" if side == "x" else "x"
    return sum(1 for q in nb(to, 1) if b.get(q) == enemy)


def moves(b, side):
    out = set()
    for p, v in b.items():
        if v != side:
            continue
        for d in (1, 2):
            for q in nb(p, d):
                if q not in b:
                    out.add((None if d == 1 else p, q))
    return out


def best_capture(b, side):
    best = (-1, None)
    for frm, to in moves(b, side):
        f = flips(b, to, side)
        if f > best[0]:
            best = (f, (frm, to))
    return best


def play(b, mv, side):
    b = dict(b)
    if mv == "0000":
        return b
    if len(mv) == 2:
        to = sq(mv)
    else:
        frm, to = sq(mv[:2]), sq(mv[2:])
        del b[frm]
    b[to] = side
    enemy = "o" if side == "x" else "x"
    for q in nb(to, 1):
        if b.get(q) == enemy:
            b[q] = side
    return b


def name(p):
    return chr(97 + p[1]) + str(7 - p[0])


def fen_of(b, side):
    rows = []
    for r in range(N):
        s, e = "", 0
        for c in range(N):
            v = b.get((r, c))
            if v is None:
                e += 1
            else:
                if e:
                    s += str(e)
                    e = 0
                s += v
        rows.append(s + (str(e) if e else ""))
    return "/".join(rows) + " " + side


args = [a for a in sys.argv[1:]]
who = "baseline"
mn, gap = 6, 4
FREE = 3  # 'for free': after taking it, the opponent's best recapture flips <= FREE
files = []
i = 0
while i < len(args):
    if args[i] == "--who":
        who = args[i + 1]; i += 2
    elif args[i] == "--min":
        mn = int(args[i + 1]); i += 2
    elif args[i] == "--gap":
        gap = int(args[i + 1]); i += 2
    else:
        files.append(args[i]); i += 1

stats = {}
examples = []
for fn in files:
    for line in open(fn, encoding="utf-8"):
        if not line.strip():
            continue
        g = json.loads(line)
        b, side = parse_fen(g["start"])
        mv = g["moves"]
        for k, m in enumerate(mv):
            mover = m["mover"]
            if who in (mover, "any"):
                st = stats.setdefault(mover, [0, 0, 0])
                st[0] += 1
                bf, bm = best_capture(b, side)
                if m["move"] != "0000":
                    to = sq(m["move"][-2:])
                    pf = flips(b, to, side)
                    enemy_side = "o" if side == "x" else "x"
                    taken = play(b, (name(bm[0]) + name(bm[1])) if bm[0] else name(bm[1]), side) if bm else b
                    recap = best_capture(taken, enemy_side)[0] if bm else 99
                    if bf >= mn and pf <= bf - gap and recap <= FREE:
                        nb_after = play(b, m["move"], side)
                        reply = mv[k + 1]["move"] if k + 1 < len(mv) else "0000"
                        nb2 = play(nb_after, reply, "o" if side == "x" else "x")
                        still = flips(nb2, bm[1], side) if bm[1] not in nb2 else 0
                        st[1] += 1
                        if still < bf - 2:
                            st[2] += 1
                        if len(examples) < 12:
                            examples.append((fn.replace("\\", "/").split("/")[-1], mover, fen_of(b, side), m["move"], pf, name(bm[1]), bf, m.get("score"), m.get("depth"), reply, still))
            b = play(b, m["move"], side)
            side = "o" if side == "x" else "x"

for mover, (n, miss, plugged) in stats.items():
    print(f"{mover}: moves {n} | missed FREE capture (best >= {mn}, recapture <= {FREE}, played <= best-{gap}): {miss} ({100.0 * miss / max(1, n):.2f}%) | opponent removed it next move: {plugged}")
print("\nexamples: file | mover | FEN | played (flips) | best capture square (flips) | score depth | reply | flips still available after reply")
for e in examples:
    print(" | ".join(map(str, e)))
