import sys
import re
import json

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae22cc69eb7df09d05b6256"
s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

# Extract _rawLogJSON using regex on the raw text
idx = r.text.find("var _rawLogJSON = ")
if idx != -1:
    sub = r.text[idx + len("var _rawLogJSON = "):]
    end_idx = sub.find(";\n")
    if end_idx == -1:
        end_idx = sub.find(";")
    js_str = sub[:end_idx].strip()
    # It's a JSON string literal
    inner_json = json.loads(js_str)
    # And inside it's the actual JSON
    match_data = json.loads(inner_json)
    print(f"Total steps: {len(match_data)}")
    
    # Also find players and score
    p_idx = r.text.find("playerNames = ")
    if p_idx != -1:
        p_sub = r.text[p_idx + len("playerNames = "):]
        p_end = p_sub.find(";")
        players = json.loads(p_sub[:p_end].strip())
        print("Players:", players)

    # Find scores
    s_idx = r.text.find("scores = ")
    if s_idx != -1:
        s_sub = r.text[s_idx + len("scores = "):]
        s_end = s_sub.find(";")
        print("Scores:", s_sub[:s_end].strip())

    # Print the last few turns
    print("\n--- Last 5 entries in match log ---")
    for entry in match_data[-5:]:
        print(entry)
