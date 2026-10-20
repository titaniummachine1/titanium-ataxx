import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ataxx_move as M

RE_GAME = re.compile(r'^-- game \d+:.*titanium=(x|o).*?: (titanium wins|opponent wins|draw)')


def key_of(o0, o1, bl, st):
    return (o0, o1, bl, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='logs/trees_games.txt')
    ap.add_argument('--exe', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--nodes', type=int, default=50000)
    ap.add_argument('--max-games', type=int, default=60)
    ap.add_argument('--out', default='data/nnue/trees.npz')
    ap.add_argument('--min-nodes-per-game', type=int, default=8)
    a = ap.parse_args()

    games = []
    cur = None
    with open(a.log, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = RE_GAME.match(line.strip())
            if m:
                cur = {'color': 0 if m.group(1) == 'x' else 1,
                       'result': m.group(2), 'moves': []}
                games.append(cur)
                continue
            if cur is not None and line.strip() and not line.startswith('===') and not line.startswith('total') and not line.startswith('Match:'):
                toks = line.strip().split()
                if toks and all(2 <= len(t) <= 4 for t in toks):
                    cur['moves'] = toks
                    cur = None
    games = [g for g in games if len(g['moves']) >= a.min_nodes_per_game][:a.max_games]
    print('games %d' % len(games), flush=True)

    nodes = []
    index = {}
    edges = []
    outs = []
    seen_edges = set()

    start = M.parse_board('......x ....... ....... ....... ....... ....... o...... b')

    def add(pos, out_val):
        k = key_of(*pos)
        i = index.get(k)
        if i is None:
            i = len(nodes)
            index[k] = i
            nodes.append(pos)
            outs.append(out_val)
        return i

    for g in games:
        pos = start
        prev = add(pos, 0.0)
        for mv in g['moves']:
            if mv == 'pass':
                pos = (pos[0], pos[1], pos[2], 1 - pos[3])
                continue
            nxt = M.apply_move(*pos, mv)
            if nxt is None:
                break
            stm_moved = pos[3]
            won = (g['result'] == 'titanium wins' and g['color'] == stm_moved) or \
                  (g['result'] == 'opponent wins' and g['color'] != stm_moved)
            draw = g['result'] == 'draw'
            o = 0.5 if draw else (1.0 if won else 0.0)
            outs[prev] = o
            ci = add(nxt, o)
            e = (prev, ci)
            if e not in seen_edges:
                seen_edges.add(e)
                edges.append(e)
            prev = ci
            pos = nxt

    print('nodes %d edges %d' % (len(nodes), len(edges)), flush=True)

    boards = [M.board_str(*p) for p in nodes]
    cmd = [a.exe, 'label', '--nodes', str(a.nodes), '--tuple', a.net, '--tuple-native']
    inp = ('\n'.join(boards) + '\n').encode()
    out = subprocess.run(cmd, input=inp, capture_output=True).stdout.decode('utf-8', 'replace')
    teach = []
    for line in out.splitlines():
        parts = line.split()
        if parts:
            try:
                teach.append(float(parts[0]))
            except ValueError:
                pass
    print('labels %d' % len(teach), flush=True)
    if len(teach) != len(nodes):
        raise SystemExit('label count mismatch %d vs %d' % (len(teach), len(nodes)))

    regret = np.zeros(len(nodes), np.float32)
    t = np.array(teach, np.float32)
    for p, c in edges:
        r = t[p] + t[c]
        if r > regret[p]:
            regret[p] = r
    regret = np.clip(regret, 0, 800)

    occ0 = np.array([p[0] for p in nodes], np.uint64)
    occ1 = np.array([p[1] for p in nodes], np.uint64)
    blk = np.array([p[2] for p in nodes], np.uint64)
    stm = np.array([p[3] for p in nodes], np.uint8)
    outv = np.array(outs, np.float32)
    ep = np.array([e[0] for e in edges], np.int64)
    ec = np.array([e[1] for e in edges], np.int64)
    wsum = np.ones(len(nodes), np.float32)
    np.savez_compressed(a.out, occ0=occ0, occ1=occ1, blk=blk, stm=stm,
                        teacher=t, out=outv, edge_p=ep, edge_c=ec,
                        wsum=wsum, regret=regret)
    print('wrote %s nodes %d edges %d teacher mean %+.1f regret mean %.1f'
          % (a.out, len(nodes), len(edges), float(t.mean()), float(regret.mean())), flush=True)


if __name__ == '__main__':
    main()
