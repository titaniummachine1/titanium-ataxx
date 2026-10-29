"""Upload v7 dataset and W=512 upscaled checkpoint to Modal volume ti-train."""
import os
import subprocess
import sys

ROOT = r"C:\gitProjects\titanium-ataxx"
MODAL_UP = os.path.join(ROOT, "data", "nnue", "modal_up")

FILES_TO_UPLOAD = [
    ("champion_w512_d4_net2net.pt", "/up/champion_w512_d4_net2net.pt"),
    ("final_train_v7.npz", "/up/final_train_v7.npz"),
]

def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for local_name, remote_path in FILES_TO_UPLOAD:
        local_path = os.path.join(MODAL_UP, local_name)
        if not os.path.exists(local_path):
            sys.exit(f"ERROR: Local file not found: {local_path}")
        size_mb = os.path.getsize(local_path) / 1_048_576.0
        print(f"Uploading {local_name} ({size_mb:.1f} MB) -> {remote_path}...", flush=True)
        cmd = ["python", "-m", "modal", "volume", "put", "--force", "ti-train", local_path, remote_path]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0 and "already exists" not in r.stderr:
            print(f"FAILED to upload {local_name}:\n{r.stderr}", flush=True)
            sys.exit(1)
        print(f"Uploaded {local_name} successfully.", flush=True)

if __name__ == "__main__":
    main()
