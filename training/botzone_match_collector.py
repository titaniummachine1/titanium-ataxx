"""Botzone Match Collector for Ataxx (Game ID: 5809c7647f65182b044b5e3b).
Downloads public match logs from top-rated Botzone bots to diversify training data.
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

GAME_ATAXX = "5809c7647f65182b044b5e3b"
OUT_DIR = "data/botzone_matches"

def get_session(email=None, password=None):
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    if email and password:
        r = s.post("https://www.botzone.org.cn/login", data={"email": email, "password": password}, timeout=30)
        if r.status_code == 200:
            print("Logged in to Botzone successfully.")
    return s

def fetch_match_log(s, match_id):
    url = f"https://www.botzone.org.cn/match/{match_id}"
    r = s.get(url, timeout=20)
    if r.status_code != 200:
        return None
    m = re.search(r'var _rawLogJSON\s*=\s*(".*?");\s*\n', r.text)
    if not m:
        return None
    try:
        raw_str = json.loads(m.group(1))
        return json.loads(raw_str)
    except Exception as e:
        return None

def collect_recent_matches(s, max_matches=10):
    os.makedirs(OUT_DIR, exist_ok=True)
    url = f"https://www.botzone.org.cn/matches?game={GAME_ATAXX}"
    r = s.get(url, timeout=20)
    if r.status_code != 200:
        print(f"Failed to fetch matches page: {r.status_code}")
        return []
    
    match_ids = list(dict.fromkeys(re.findall(r'/match/([a-f0-9]{24})', r.text)))
    print(f"Found {len(match_ids)} match IDs on page 1: {match_ids}")
    
    saved = []
    for mid in match_ids[:max_matches]:
        out_file = os.path.join(OUT_DIR, f"{mid}.json")
        if os.path.exists(out_file):
            print(f"Match {mid} already downloaded.")
            saved.append(out_file)
            continue
        log = fetch_match_log(s, mid)
        if log:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
            print(f"Saved match {mid} ({len(log)} steps) -> {out_file}")
            saved.append(out_file)
            time.sleep(0.5)
    return saved

if __name__ == "__main__":
    email = os.environ.get("BOTZONE_EMAIL", "terminatort8004@gmail.com")
    password = os.environ.get("BOTZONE_PASSWORD", "beniamin2005")
    s = get_session(email, password)
    saved = collect_recent_matches(s, max_matches=5)
    print(f"Downloaded {len(saved)} Botzone matches successfully.")
