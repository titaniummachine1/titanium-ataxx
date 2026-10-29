"""Upload downscaled W=256 checkpoint to Modal volume ti-train."""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\gitProjects\titanium-ataxx"
LOCAL_PATH = os.path.join(ROOT, "data", "nnue", "w256_downscaled_ep7.pt")
REMOTE_PATH = "/up/w256_downscaled_ep7.pt"

def main():
    if not os.path.exists(LOCAL_PATH):
        sys.exit(f"ERROR: Local file not found: {LOCAL_PATH}")
    size_mb = os.path.getsize(LOCAL_PATH) / 1_048_576.0
    print(f"Uploading {LOCAL_PATH} ({size_mb:.1f} MB) -> {REMOTE_PATH}...", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    cmd = ["python", "-m", "modal", "volume", "put", "--force", "ti-train", LOCAL_PATH, REMOTE_PATH]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"FAILED to upload:\n{r.stderr}", flush=True)
        sys.exit(1)
    print("Uploaded successfully!", flush=True)

if __name__ == "__main__":
    main()
