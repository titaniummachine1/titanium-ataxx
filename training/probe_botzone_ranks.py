import requests
import re
import json
from bs4 import BeautifulSoup

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
s.post("https://www.botzone.org.cn/login", data={"email": "terminatort8004@gmail.com", "password": "beniamin2005"}, timeout=20)

ATAXX_ID = "5809c7647f65182b044b5e3b"

top_bots = []
for page in (1, 2):
    url = f"https://www.botzone.org.cn/game/ranklist/{ATAXX_ID}?page={page}"
    r = s.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.find_all("tr", attrs={"data-botid": True})
    for tr in rows:
        bot_id = tr["data-botid"]
        cells = [c.text.strip().replace("\n", " ") for c in tr.find_all("td")]
        rank = cells[0] if len(cells) > 0 else ""
        name = cells[1] if len(cells) > 1 else ""
        author = cells[2] if len(cells) > 2 else ""
        score = cells[3] if len(cells) > 3 else ""
        top_bots.append({
            "rank": rank,
            "bot_id": bot_id,
            "name": name,
            "author": author,
            "score": score
        })

with open(r"C:\gitProjects\titanium-ataxx\data\top_ataxx_bots.json", "w", encoding="utf-8") as f:
    json.dump(top_bots, f, indent=2, ensure_ascii=False)

print(f"Total top bots fetched: {len(top_bots)}")
for b in top_bots[:20]:
    print(f"Rank #{b['rank']:<3} | Score: {b['score']:<6} | {b['name']:<25} | ID: {b['bot_id']}")
