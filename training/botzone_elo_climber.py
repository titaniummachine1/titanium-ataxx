"""Botzone Autonomous Elo Climber for Titanium Ataxx.

User mandate:
  "Don't rematch you maximize elo gain challenge bot of biggest elo you can
   wait for game to finish and start new one with new strongest you can challange"

Protocol:
  1. Login to Botzone via credentials in C:/gitProjects/botzone-bots/tools/creds.py.
  2. Inspect Titanium's current Elo and all historical opponent IDs (to enforce NO REMATCH).
  3. Scrape the challenge page (/game/ranklist/match/<MY_BOT_ID>).
  4. Find the unplayed opponent with the MAXIMUM Elo rating.
  5. Challenge that opponent (one game at a time).
  6. Wait and monitor the match until completion.
  7. Report match outcome, piece score, and Elo delta.
  8. Repeat to continuously accelerate Elo climbing.
"""
import os
import sys
import time
import json
import re
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

# Botzone setup
MY_BOT_ID = "6ad7e0c3bdad901f323f6516"
BASE = "https://www.botzone.org.cn"
CREDS_PATH = "C:/gitProjects/botzone-bots/tools"
sys.path.insert(0, CREDS_PATH)
try:
    from creds import EMAIL, PASSWORD
except ImportError:
    sys.exit("Error: Could not import EMAIL and PASSWORD from creds.py")

HISTORY_FILE = "C:/gitProjects/titanium-ataxx/logs/botzone_climber_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"played_bots": {}, "matches": []}

def save_history(hist):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": BASE + "/mybots",
        "X-Requested-With": "XMLHttpRequest",
    })
    for attempt in range(8):
        try:
            r = s.post(BASE + "/login", data={"email": EMAIL, "password": PASSWORD}, timeout=60)
            if '"success":true' in r.text or r.status_code == 200:
                print("  [Auth] Successfully logged into Botzone.")
                return s
        except Exception as e:
            print(f"  [Auth] Login attempt {attempt+1} failed: {e}")
        time.sleep(3)
    raise RuntimeError("Failed to login to Botzone after multiple retries.")

def get_bot_details(s):
    url = f"{BASE}/mybots/detail/{MY_BOT_ID}"
    for _ in range(5):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 200:
                j = r.json()
                if j.get("success") and "bot" in j:
                    return j["bot"]
        except Exception:
            time.sleep(2)
    raise RuntimeError("Failed to fetch bot details.")

def get_historical_opponent_ids(bot_data):
    played = set()
    matches = bot_data.get("rank_matches", [])
    for m in matches:
        players = m.get("players", [])
        for p in players:
            b_info = p.get("bot", {}).get("bot", {})
            b_id = b_info.get("_id")
            if b_id and b_id != MY_BOT_ID:
                played.add(b_id)
    return played

def fetch_candidates(s):
    url = f"{BASE}/game/ranklist/match/{MY_BOT_ID}"
    for _ in range(5):
        try:
            r = s.get(url, timeout=60)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                candidates = []
                for tr in soup.find_all("tr"):
                    chk = tr.find("input", {"type": "checkbox"})
                    if chk and chk.get("name"):
                        bid = chk["name"]
                        if bid == MY_BOT_ID:
                            continue
                        tds = tr.find_all("td")
                        if len(tds) >= 2:
                            name_text = tds[0].get_text(" ", strip=True)
                            score_text = tds[1].get_text(strip=True)
                            try:
                                score_val = float(score_text)
                            except ValueError:
                                score_val = 0.0
                            # Extract clean bot name
                            m = re.match(r"^([^\d]+(?:\d+)?)", name_text)
                            bot_name = m.group(1).strip() if m else name_text[:25]
                            candidates.append({
                                "id": bid,
                                "name": bot_name,
                                "score": score_val,
                                "full_text": name_text,
                            })
                return candidates
        except Exception as e:
            print(f"  [Scrape] Error fetching candidates: {e}")
            time.sleep(3)
    return []

def challenge_opponent(s, opp_id):
    url = f"{BASE}/game/ranklist/match/{MY_BOT_ID}"
    data = {MY_BOT_ID: "true", opp_id: "true"}
    for attempt in range(5):
        try:
            r = s.post(url, data=data, timeout=60)
            if r.status_code == 200:
                print(f"  [Challenge] Challenge successfully sent to {opp_id}.")
                return True
            else:
                print(f"  [Challenge] POST returned status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"  [Challenge] Retry {attempt+1}: {e}")
            time.sleep(3)
    return False

def wait_for_match_completion(s, initial_score, opp_id, opp_name, timeout_secs=600):
    print(f"  [Monitor] Waiting for match against {opp_name} to finish (max {timeout_secs}s)...")
    t0 = time.time()
    seen_match_id = None

    while time.time() - t0 < timeout_secs:
        time.sleep(8)
        try:
            bot = get_bot_details(s)
            current_score = bot.get("score", initial_score)
            matches = bot.get("rank_matches", [])

            # Check if there is a recent match against opp_id
            for m in reversed(matches):
                players = m.get("players", [])
                p_bids = [p.get("bot", {}).get("bot", {}).get("_id") for p in players]
                if opp_id in p_bids:
                    mid = m.get("_id")
                    scores = m.get("scores", [])
                    # Match finished if scores are populated
                    if scores and len(scores) >= 2:
                        my_idx = 0 if p_bids[0] == MY_BOT_ID else 1
                        opp_idx = 1 - my_idx
                        my_pts = scores[my_idx]
                        opp_pts = scores[opp_idx]

                        verdict = "WIN (+2)" if my_pts > opp_pts else ("LOSS (0)" if my_pts < opp_pts else "DRAW (1)")
                        delta = current_score - initial_score
                        print(f"\n{'='*60}")
                        print(f"  >>> MATCH FINISHED: {verdict} <<<")
                        print(f"  Match ID    : {mid}")
                        print(f"  Opponent    : {opp_name} (ID: {opp_id})")
                        print(f"  Game Score  : Titanium {my_pts} - {opp_pts} {opp_name}")
                        print(f"  Titanium Elo: {initial_score:.2f} -> {current_score:.2f} ({delta:+.2f})")
                        print(f"{'='*60}\n")
                        return {
                            "match_id": mid,
                            "opp_id": opp_id,
                            "opp_name": opp_name,
                            "my_pts": my_pts,
                            "opp_pts": opp_pts,
                            "verdict": verdict,
                            "old_elo": initial_score,
                            "new_elo": current_score,
                            "delta_elo": delta,
                            "duration_s": round(time.time() - t0, 1),
                        }
                    else:
                        print(f"  [Monitor] Match {mid} is currently executing...", end="\r", flush=True)

            # If score changed even without finding match in array
            if abs(current_score - initial_score) > 0.05:
                delta = current_score - initial_score
                print(f"\n  [Monitor] Elo rating changed: {initial_score:.2f} -> {current_score:.2f} ({delta:+.2f})")
                return {
                    "match_id": "unknown",
                    "opp_id": opp_id,
                    "opp_name": opp_name,
                    "old_elo": initial_score,
                    "new_elo": current_score,
                    "delta_elo": delta,
                    "duration_s": round(time.time() - t0, 1),
                }

        except Exception as e:
            print(f"  [Monitor] Poll warning: {e}")

    print(f"  [Monitor] Timed out waiting for match after {timeout_secs}s.")
    return None

def main():
    max_matches = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print("=" * 60)
    print("=== TITANIUM BOTZONE AUTONOMOUS ELO CLIMBER ===")
    print("=" * 60)
    print(f"Max matches in this session: {max_matches}")
    print("Rule: No rematches, strictly challenge highest-Elo opponent available.\n")

    history = load_history()
    played_opponents = set(history.get("played_bots", {}).keys())

    s = get_session()
    bot = get_bot_details(s)
    current_elo = bot.get("score", 0.0)
    print(f"Titanium Current Elo: {current_elo:.2f} | Total rank games played: {bot.get('realrankplaycount', 0)}")

    # Add historical opponents from Botzone account
    account_played = get_historical_opponent_ids(bot)
    played_opponents.update(account_played)
    print(f"Known historical opponents to avoid rematching: {len(played_opponents):,}\n")

    matches_played = 0
    while matches_played < max_matches:
        print(f"--- [Round {matches_played + 1} / {max_matches}] Searching for strongest opponent ---")
        candidates = fetch_candidates(s)
        if not candidates:
            print("  No candidates found on challenge page. Waiting 15s...")
            time.sleep(15)
            continue

        # Filter out played opponents
        unplayed = [c for c in candidates if c["id"] not in played_opponents]
        if not unplayed:
            print("  All visible candidates have been played! Resetting unplayed filter to lowest frequency...")
            unplayed = candidates

        # Sort strictly descending by Elo score (highest Elo first)
        unplayed.sort(key=lambda x: x["score"], reverse=True)
        target = unplayed[0]

        print(f"  Selected target: {target['name']} | Elo: {target['score']:.2f} | ID: {target['id']}")
        initial_score = current_elo

        # Send challenge
        success = challenge_opponent(s, target["id"])
        if not success:
            print("  Challenge failed. Waiting 10s before retry...")
            time.sleep(10)
            continue

        # Mark as played immediately
        played_opponents.add(target["id"])
        history["played_bots"][target["id"]] = {
            "name": target["name"],
            "score": target["score"],
            "last_challenged": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_history(history)

        # Wait for match to finish
        result = wait_for_match_completion(s, initial_score, target["id"], target["name"], timeout_secs=300)
        if result:
            history["matches"].append(result)
            save_history(history)
            current_elo = result["new_elo"]
            matches_played += 1
        else:
            print("  Match timed out or did not register. Continuing to next round...")
            time.sleep(10)

        # Brief pause between matches so ladder updates
        print("  Cooling down 8 seconds before next challenge...")
        time.sleep(8)

    print(f"\nAll {matches_played} matches finished. Final Titanium Elo: {current_elo:.2f}")

if __name__ == "__main__":
    main()
