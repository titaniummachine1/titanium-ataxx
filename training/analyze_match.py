import sys
import json
import re
from datetime import datetime

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae22cc69eb7df09d05b6256"
s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

m_time = re.search(r'data-livestamp="(\d+)"', r.text)
if m_time:
    ts = int(m_time.group(1))
    print("Match timestamp:", ts, datetime.fromtimestamp(ts))

idx = r.text.find("var _rawLogJSON = ")
sub = r.text[idx + len("var _rawLogJSON = "):]
end_idx = sub.find(";")
match_data = json.loads(json.loads(sub[:end_idx].strip()))

# Look at moves and debug logs
moves = []
for i, step in enumerate(match_data):
    for pid in ("0", "1"):
        if pid in step and isinstance(step[pid], dict) and "response" in step[pid]:
            resp = step[pid]["response"]
            dbg = step[pid].get("debug", "")
            t = step[pid].get("time", 0)
            moves.append((pid, resp, dbg, t))

print(f"Total moves: {len(moves)}")
print("First 10 moves:")
for m in moves[:10]:
    print(f"  P{m[0]}: {m[1]} | t={m[3]}ms | dbg={m[2]}")

print("\nLast 10 moves:")
for m in moves[-10:]:
    print(f"  P{m[0]}: {m[1]} | t={m[3]}ms | dbg={m[2]}")
