import sys
import json
import re

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae230979eb7df09d05b67dd"
s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

# Extract players and log
p_idx = r.text.find("playerNames = ")
if p_idx != -1:
    p_sub = r.text[p_idx + len("playerNames = "):]
    p_end = p_sub.find(";")
    players = json.loads(p_sub[:p_end].strip())
    print("Players:", players)

idx = r.text.find("var _rawLogJSON = ")
sub = r.text[idx + len("var _rawLogJSON = "):]
end_idx = sub.find(";")
match_data = json.loads(json.loads(sub[:end_idx].strip()))

moves = []
for i, step in enumerate(match_data):
    for pid in ("0", "1"):
        if pid in step and isinstance(step[pid], dict) and "response" in step[pid]:
            resp = step[pid]["response"]
            dbg = step[pid].get("debug", "")
            t = step[pid].get("time", 0)
            moves.append((len(moves) + 1, pid, resp, dbg, t))

print(f"Total plies: {len(moves)}")
print("\n--- All moves with depth <= 1 or time >= 3000ms or score drop ---")
for ply, pid, resp, dbg, t in moves:
    if pid == "1" or pid == "0":
        # check if depth 0 or depth 1
        if "d0 " in dbg or "d1 " in dbg or t >= 3500 or "d4 " in dbg:
            print(f"Ply {ply:3d} | P{pid} | ({resp.get('x0')},{resp.get('y0')})->({resp.get('x1')},{resp.get('y1')}) | {dbg} | {t}ms")

print("\n--- Last 10 moves ---")
for ply, pid, resp, dbg, t in moves[-10:]:
    print(f"Ply {ply:3d} | P{pid} | ({resp.get('x0')},{resp.get('y0')})->({resp.get('x1')},{resp.get('y1')}) | {dbg} | {t}ms")

# Result
print("\nFinal outcome:")
for step in match_data[-3:]:
    print(step)
