import json

raw_log = json.load(open('c:/gitProjects/titanium-ataxx/logs/match_atx_6ae11253_log.json', encoding='utf-8'))

board = [[0]*7 for _ in range(7)]
board[0][0] = 1 # Black (atx)
board[6][6] = 1
board[6][0] = 2 # White (Titanium)
board[0][6] = 2

def apply_move(board, p, x0, y0, x1, y1):
    other = 2 if p == 1 else 1
    dist = max(abs(x0 - x1), abs(y0 - y1))
    if dist == 2:
        board[y0][x0] = 0
    board[y1][x1] = p
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = x1 + dx, y1 + dy
            if 0 <= nx < 7 and 0 <= ny < 7:
                if board[ny][nx] == other:
                    board[ny][nx] = p

plies = []

for e in raw_log:
    if '0' in e:
        info = e['0']
        resp = info.get('response', {})
        x0, y0, x1, y1 = resp.get('x0'), resp.get('y0'), resp.get('x1'), resp.get('y1')
        dbg = info.get('debug', '')
        tm = info.get('time', 0)
        apply_move(board, 1, x0, y0, x1, y1)
        b = sum(row.count(1) for row in board)
        w = sum(row.count(2) for row in board)
        plies.append({
            'ply': len(plies) + 1,
            'player': 'atx (Black)',
            'move': f"({x0},{y0})->({x1},{y1})",
            'kind': 'jump' if max(abs(x0-x1), abs(y0-y1)) == 2 else 'clone',
            'b': b, 'w': w,
            'time': tm,
            'debug': dbg,
            'resp': resp
        })
    elif '1' in e:
        info = e['1']
        resp = info.get('response', {})
        x0, y0, x1, y1 = resp.get('x0'), resp.get('y0'), resp.get('x1'), resp.get('y1')
        dbg = info.get('debug', '')
        tm = info.get('time', 0)
        apply_move(board, 2, x0, y0, x1, y1)
        b = sum(row.count(1) for row in board)
        w = sum(row.count(2) for row in board)
        plies.append({
            'ply': len(plies) + 1,
            'player': 'Titanium (White)',
            'move': f"({x0},{y0})->({x1},{y1})",
            'kind': 'jump' if max(abs(x0-x1), abs(y0-y1)) == 2 else 'clone',
            'b': b, 'w': w,
            'time': tm,
            'debug': dbg,
            'resp': resp
        })

print(f"Total plies: {len(plies)}")
b_final = sum(row.count(1) for row in board)
w_final = sum(row.count(2) for row in board)
print(f"Final Count: atx (Black)={b_final}, Titanium (White)={w_final}\n")

print("Turn | Player | Move | Type | Stones (B-W) | Time | Debug / Eval")
print("-" * 80)
for p in plies:
    pl = "ATX" if "atx" in p['player'] else "TIT"
    d = p['debug']
    if not d and 'data' in p['resp']:
        d = f"data={p['resp']['data'][:30]}"
    print(f"{p['ply']:4d} | {pl:3s} | {p['move']:11s} | {p['kind']:5s} | B:{p['b']:2d} W:{p['w']:2d} (net {p['b']-p['w']:+3d}) | {p['time']:5d}ms | {d}")
