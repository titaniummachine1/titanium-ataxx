SIZE = 7
SQUARES = 49
FULL = (1 << SQUARES) - 1


def sq_of(name):
    f = ord(name[0].lower()) - ord('a')
    r = int(name[1])
    if not (0 <= f < 7 and 1 <= r <= 7):
        return None
    return (7 - r) * 7 + f


def name_of(sq):
    r, f = divmod(sq, 7)
    return '%s%d' % ('abcdefg'[f], 7 - r)


def ring1(sq):
    r, f = divmod(sq, 7)
    out = []
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            rr, ff = r + dr, f + df
            if 0 <= rr < 7 and 0 <= ff < 7:
                out.append(rr * 7 + ff)
    return out


RING = [ring1(s) for s in range(49)]


def parse_board(text):
    parts = text.strip().split()
    if not parts:
        return None
    side = 'b'
    if parts[-1] in ('b', 'w') and len(parts) > 1:
        side = parts[-1]
        parts = parts[:-1]
    board = ''.join(parts)
    if len(board) != 49:
        return None
    occ0 = occ1 = blk = 0
    for i, c in enumerate(board):
        if c == 'x':
            occ0 |= 1 << i
        elif c == 'o':
            occ1 |= 1 << i
        elif c == '#':
            blk |= 1 << i
    stm = 0 if side == 'b' else 1
    return occ0, occ1, blk, stm


def board_str(occ0, occ1, blk, stm):
    out = []
    for i in range(49):
        bit = 1 << i
        out.append('#' if blk & bit else 'x' if occ0 & bit else 'o' if occ1 & bit else '.')
    return ''.join(out) + (' b' if stm == 0 else ' w')


def apply_move(occ0, occ1, blk, stm, token):
    token = token.strip()
    if token in ('pass', '0000', ''):
        return occ0, occ1, blk, 1 - stm
    own, enm = (occ0, occ1) if stm == 0 else (occ1, occ0)
    if len(token) == 4:
        frm = sq_of(token[0:2])
        to = sq_of(token[2:4])
        if frm is None or to is None:
            return None
    elif len(token) == 2:
        to = sq_of(token)
        if to is None:
            return None
        cand = own & sum(1 << s for s in RING[to])
        if cand == 0:
            return None
        frm = (cand & -cand).bit_length() - 1
    else:
        return None
    if not (0 <= to < 49) or (blk >> to) & 1:
        return None
    if ((occ0 | occ1) >> to) & 1:
        return None
    fr, fc = divmod(frm, 7)
    tr, tc = divmod(to, 7)
    dist = max(abs(fr - tr), abs(fc - tc))
    new_own = own
    if dist == 1:
        new_own |= 1 << to
    elif dist == 2:
        if not ((own >> frm) & 1):
            return None
        new_own = (new_own & ~(1 << frm)) | (1 << to)
    else:
        return None
    conv = 0
    for s in RING[to]:
        if (enm >> s) & 1:
            conv |= 1 << s
    new_own |= conv
    new_enm = enm & ~conv
    if stm == 0:
        return new_own, new_enm, blk, 1
    return new_enm, new_own, blk, 0
