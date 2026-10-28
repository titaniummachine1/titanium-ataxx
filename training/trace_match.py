import sys
import json
import re

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae22cc69eb7df09d05b6256"
s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

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
print("Ply | Player | Move | Eval/Debug | Time")
print("-" * 60)
for ply, pid, resp, dbg, t in moves:
    # Extract eval if present
    # P1 (Titanium): d14 n899200 t11957 s-71 bundled-main/22
    # P0 (PyJataxx): {"ccc":5695757,"expansions":133115,"h":0.0763426348567009,"maxLevel":16}
    print(f"{ply:3d} | P{pid} | ({resp.get('x0')},{resp.get('y0')})->({resp.get('x1')},{resp.get('y1')}) | {dbg[:40]} | {t}ms")
