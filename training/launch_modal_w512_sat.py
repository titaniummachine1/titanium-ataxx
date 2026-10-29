"""Launch Modal W=512 D4 saturation continuation training with UTF-8 encoding wrapper."""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\gitProjects\titanium-ataxx"

def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [
        sys.executable,
        "-m", "modal", "run", "--detach",
        os.path.join(ROOT, "training", "modal_train.py"),
        "--tag", "w512_d4_v7_sat",
        "--data-tag", "v7",
        "--args", "--stage base --init /vol/ckpt/w512_d4_v7/net.pt --width 512 --epochs 12 --bs 16384 --lr 1.5e-4 --diff-nudge 0.0 --opt ranger --ema 0.9995",
    ]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print("Process exited with code:", p.returncode)

if __name__ == "__main__":
    main()
