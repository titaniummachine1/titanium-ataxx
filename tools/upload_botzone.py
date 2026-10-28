"""Upload Botzone package via cloud container (high international bandwidth, reliable connection).

Usage:
    $env:BOTZONE_EMAIL = "your_email"
    $env:BOTZONE_PASSWORD = "your_password"
    modal run tools/upload_botzone.py
"""

import os
import modal

app = modal.App("botzone-uploader")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")

SO_PATH = "C:/gitProjects/main-ataxx/botzone/upload/data/titanium_v2.so"
PY_PATH = "C:/gitProjects/main-ataxx/botzone/upload/titanium_botzone.py"
BOT_ID = "6ad7e0c3bdad901f323f6516"  # titanium_ataxx_PY6
GAME_ID = "5809c7647f65182b044b5e3b"  # Ataxx

@app.function(image=image, timeout=300)
def upload_to_botzone(email: str, password: str, so_bytes: bytes, py_bytes: bytes):
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    print("Logging in to Botzone from cloud container...")
    r_login = s.post(
        "https://www.botzone.org.cn/login",
        data={"email": email, "password": password},
        timeout=30,
    )
    r_login.raise_for_status()
    print("Login status:", r_login.status_code)

    # Step 1: Upload titanium_v2.so to user storage (if provided)
    if so_bytes:
        print(f"Uploading titanium_v2.so ({len(so_bytes)} bytes)...")
        files = {"file": ("titanium_v2.so", so_bytes, "application/octet-stream")}
        data = {"path": "", "filename": "titanium_v2.so"}
        r_up = s.post("https://www.botzone.org.cn/userfile/uploadfile", data=data, files=files, timeout=180)
        print("Upload .so response:", r_up.status_code, r_up.text[:300])

    # Step 2: Modify bot settings to ensure persistent session mode (enable_keep_running=true)
    print("Modifying bot settings (enable_keep_running=true)...")
    r_mod = s.post("https://www.botzone.org.cn/mybots/modify", data={
        "botid": BOT_ID,
        "description": "Titanium Ataxx: Python driver + Rust engine (persistent)",
        "enable_keep_running": "true",
        "simpleio": "true",
        "opensource": "false",
    }, timeout=30)
    print("Modify response:", r_mod.status_code, r_mod.text[:300])

    # Step 3: Create Botzone version (Note: description length must be <= 100 chars!)
    print("Creating Botzone version...")
    files_bot = {"source": ("titanium_botzone.py", py_bytes, "text/x-python")}
    data_bot = {
        "id": BOT_ID,
        "name": "titanium_ataxx_PY6",
        "game": GAME_ID,
        "extension": "py36",
        "simpleio": "true",
        "enable_keep_running": "true",
        "opensource": "false",
        "description": "v8: persistent session, warm ID search, 4s/turn",
    }
    r_bot = s.post("https://www.botzone.org.cn/mybots/create", data=data_bot, files=files_bot, timeout=30)
    print("Add bot version response:", r_bot.status_code, r_bot.text[:300])
    return r_bot.text

@app.local_entrypoint()
def main():
    email = os.environ.get("BOTZONE_EMAIL", "")
    password = os.environ.get("BOTZONE_PASSWORD", "")
    if not email or not password:
        raise ValueError("BOTZONE_EMAIL and BOTZONE_PASSWORD environment variables required")
    so_bytes = b""
    if os.path.exists(SO_PATH):
        with open(SO_PATH, "rb") as f:
            so_bytes = f.read()
    with open(PY_PATH, "rb") as f:
        py_bytes = f.read()
    res = upload_to_botzone.remote(email, password, so_bytes, py_bytes)
    print("Result:", res)
