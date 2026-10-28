import os
import sys
import re
import json
from bs4 import BeautifulSoup

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae22cc69eb7df09d05b6256"

s = get_session()

# Check possible json endpoints
for ep in [
    f"https://www.botzone.org.cn/match/{MATCH_ID}",
    f"https://www.botzone.org.cn/match/detail/{MATCH_ID}",
    f"https://www.botzone.org.cn/match/log/{MATCH_ID}",
    f"https://www.botzone.org.cn/api/match/{MATCH_ID}",
]:
    r = s.get(ep)
    print(f"{ep} -> status: {r.status_code}, type: {r.headers.get('content-type')}")
    if "json" in r.headers.get("content-type", ""):
        try:
            print("  JSON:", json.dumps(r.json(), indent=2)[:500])
        except Exception:
            pass

# From HTML page
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")
soup = BeautifulSoup(r.text, "html.parser")
print("\nPage title:", soup.title.text if soup.title else "None")

for script in soup.find_all("script"):
    text = script.string or ""
    if "match" in text.lower() or "player" in text.lower() or "init" in text.lower():
        print("\n--- Script block ---")
        print(text[:1500])
