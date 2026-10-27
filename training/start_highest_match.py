import requests

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.botzone.org.cn/game/ranklist/match/6ad7e0c3bdad901f323f6516",
    "X-Requested-With": "XMLHttpRequest",
})
print("Logging in...")
r_login = s.post(
    "https://www.botzone.org.cn/login",
    data={"email": "terminatort8004@gmail.com", "password": "beniamin2005"},
    timeout=15,
)
r_login.raise_for_status()

# Opponent: c01 (634938b64172273132e03407), rating 1554.05 (#1 in available rankmatch window)
opp_id = "634938b64172273132e03407"
ti_id = "6ad7e0c3bdad901f323f6516"

post_url = f"https://www.botzone.org.cn/game/ranklist/match/{ti_id}"
data = {
    ti_id: "true",
    opp_id: "true",
}

print(f"Initiating match between Titanium and c01 (rating 1554.05)...")
r = s.post(post_url, data=data, timeout=30)
print("Status:", r.status_code)
print("Response:", r.text)

try:
    res = r.json()
    if res.get("success"):
        mid = res.get("matchid")
        print("\n" + "="*50)
        print(f"MATCH STARTED SUCCESSFULLY!")
        print(f"Match ID: {mid}")
        print(f"Watch URL: https://en.botzone.org.cn/match/{mid}")
        print("="*50)
except Exception as e:
    print("Error parsing json:", e)
