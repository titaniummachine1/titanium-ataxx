"""Download saturated W=256 checkpoints from Modal volume ti-train."""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\gitProjects\titanium-ataxx"
LOCAL_DIR = os.path.join(ROOT, "data", "nnue", "modal_ckpt")

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    cmd = ["python", "-m", "modal", "volume", "get", "ti-train", "/ckpt/w256_d4_v7_sat", LOCAL_DIR]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode != 0:
        print("ERROR:\n", r.stderr.strip())
        sys.exit(1)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
