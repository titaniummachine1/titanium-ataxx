"""Upload Botzone package via cloud container (high international bandwidth, reliable connection).

Usage:
    $env:BOTZONE_EMAIL = "your_email"
    $env:BOTZONE_PASSWORD = "your_password"
    modal run tools/upload_botzone.py

If the .so upload stalls (requests reports 'The write operation timed out' after ~60 s,
seen 2026-10-29 from both Modal and the local PC), use curl instead - it succeeded in 640 s:
    curl -c cookies -d "email=...&password=..." https://www.botzone.org.cn/login
    curl -b cookies -F "file=@botzone/upload/data/titanium_v2.so;type=application/octet-stream"
         -F "path=" -F "filename=titanium_v2.so" https://www.botzone.org.cn/userfile/uploadfile
Then verify by downloading /userfiles/titanium_v2.so and comparing sha256.
"""

import os
import modal

app = modal.App("botzone-uploader")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO_PATH = os.path.join(ROOT, "botzone", "upload", "data", "titanium_v2.so")
PY_PATH = os.path.join(ROOT, "botzone", "upload", "titanium_botzone.py")
BOT_ID = "6ad7e0c3bdad901f323f6516"  # titanium_ataxx_PY6
GAME_ID = "5809c7647f65182b044b5e3b"  # Ataxx

@app.function(image=image, timeout=1800)
def upload_to_botzone(email: str, password: str, so_bytes: bytes, py_bytes: bytes):
    import time
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Connection": "close"})
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
        r_up = None
        for attempt in range(1, 6):
            try:
                r_up = s.post("https://www.botzone.org.cn/userfile/uploadfile", data=data, files=files,
                              timeout=(30, 900))
                print("Upload .so response:", r_up.status_code, r_up.text[:300])
                if r_up.status_code == 200 and '"success":true' in r_up.text.replace(" ", ""):
                    break
            except Exception as exc:
                print(f"Upload attempt {attempt} failed: {type(exc).__name__}: {exc}")
            time.sleep(5 * attempt)

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
        "description": "v12: W256 ep7 net 7AE0122F (best AVX2+AVX512), TNN3 head, SmallNet",
    }
    r_bot = s.post("https://www.botzone.org.cn/mybots/create", data=data_bot, files=files_bot, timeout=30)
    print("Add bot version response:", r_bot.status_code, r_bot.text[:300])
    return r_bot.text

@app.local_entrypoint()
def main():
    email = os.environ.get("BOTZONE_EMAIL", "")
    password = os.environ.get("BOTZONE_PASSWORD", "")
    if not email or not password:
        try:
            import sys
            sys.path.insert(0, "C:/gitProjects/botzone-bots/tools")
            import creds
            email = creds.EMAIL
            password = creds.PASSWORD
        except Exception:
            pass
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
