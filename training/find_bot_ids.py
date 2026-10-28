import sys
import re
import json

sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

MATCH_ID = "6ae22cc69eb7df09d05b6256"
s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

# Find bot IDs in the HTML
bot_ids = re.findall(r'/bot/([0-9a-f]{24})', r.text)
print("Bot IDs found on match page:", set(bot_ids))

user_ids = re.findall(r'/user/([0-9a-f]{24})', r.text)
print("User IDs found on match page:", set(user_ids))
