import sys
import re
import json
from datetime import datetime

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae22cc69eb7df09d05b6256"
s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

for m in re.finditer(r'([0-9a-f]{24})', r.text):
    pass

# search for dates or timestamps
dates = re.findall(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', r.text)
print("Dates found:", dates)

# search for "time" in json
m_obj = re.findall(r'"([a-zA-Z_]+time)":\s*"?([^",}]+)"?', r.text)
print("Time fields:", m_obj)

# Check user bots matches to see when it was played
r2 = s.get("https://www.botzone.org.cn/mybots/detail/6ad7e0c3bdad901f323f6516")
j2 = r2.json()
for m in j2["bot"].get("rank_matches", []):
    if m["_id"] == MATCH_ID:
        print("Found match in my bot details:", m)
