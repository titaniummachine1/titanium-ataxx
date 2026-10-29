"""Launch Modal W=512 D4 saturation training with UTF-8 encoding wrapper."""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = r"C:\gitProjects\titanium-ataxx"

def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [
        sys.executable,
        "-m", "modal", "run", "--detach",
        os.path.join(ROOT, "training", "modal_train.py"),
        "--tag", "w512_d4_v7",
        "--data-tag", "v7",
        "--args", "--stage base --init /vol/up/champion_w512_d4_net2net.pt --width 512 --epochs 8 --bs 16384 --lr 3e-4 --diff-nudge 0.0 --opt ranger --ema 0.9995",
    ]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print("Process exited with code:", p.returncode)

if __name__ == "__main__":
    main()
