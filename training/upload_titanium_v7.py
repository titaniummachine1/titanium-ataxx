"""Upload updated titanium_v2.so and create Titanium v7 on Botzone."""

import os
import requests

SO_PATH = r"c:\gitProjects\main-ataxx\botzone\upload\data\titanium_v2.so"
PY_PATH = r"c:\gitProjects\main-ataxx\botzone\upload\titanium_botzone.py"

assert os.path.exists(SO_PATH), f"Missing {SO_PATH}"
assert os.path.exists(PY_PATH), f"Missing {PY_PATH}"

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})
print("Logging in to Botzone...")
r_login = s.post(
    "https://www.botzone.org.cn/login",
    data={"email": "terminatort8004@gmail.com", "password": "beniamin2005"},
    timeout=15,
)
r_login.raise_for_status()
print("Login successful.")

# Step 1: Replace /titanium_v2.so in user storage
print("Deleting existing titanium_v2.so in storage...")
r_del = s.post("https://www.botzone.org.cn/userfile/delete/titanium_v2.so", timeout=15)
print("Delete response:", r_del.text)

print(f"Uploading new titanium_v2.so ({os.path.getsize(SO_PATH)} bytes)...")
with open(SO_PATH, "rb") as f_so:
    files = {"file": ("titanium_v2.so", f_so, "application/octet-stream")}
    data = {"path": "", "filename": "titanium_v2.so"}
    r_up = s.post("https://www.botzone.org.cn/userfile/uploadfile", data=data, files=files, timeout=60)
    print("Upload .so response:", r_up.status_code, r_up.text)

# Step 2: Add Titanium version 7
print("Adding Titanium version 7...")
with open(PY_PATH, "rb") as f_py:
    files = {"source": ("titanium_botzone.py", f_py, "text/x-python")}
    data = {
        "id": "6ad7e0c3bdad901f323f6516",
        "name": "titanium_ataxx_PY6",
        "game": "5809c7647f65182b044b5e3b",
        "extension": "py36",
        "simpleio": "true",
        "enable_keep_running": "false",
        "opensource": "false",
        "description": "v7: depth 255, in-tree circular repetition pruning, bz_over check",
    }
    r_bot = s.post("https://www.botzone.org.cn/mybots/create", data=data, files=files, timeout=30)
    print("Add bot version response:", r_bot.status_code, r_bot.text)
