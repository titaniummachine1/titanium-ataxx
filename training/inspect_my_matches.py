import sys
import json
sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

s = get_session()
r = s.get("https://www.botzone.org.cn/mybots/detail/6ad7e0c3bdad901f323f6516")
j = r.json()
bot = j["bot"]
print(f"Bot name: {bot['name']}, elo: {bot.get('score')}")
for m in bot.get("rank_matches", [])[:10]:
    mid = m["_id"]
    scores = m.get("scores", [])
    players = [p.get("bot", {}).get("name", "anon") for p in m.get("players", [])]
    print(f"Match {mid}: {' vs '.join(players)} -> {scores}")
