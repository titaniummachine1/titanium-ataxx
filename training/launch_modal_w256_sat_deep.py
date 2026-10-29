"""Launch Modal W=256 D4 deep saturation training (12 epochs) continuing from Epoch 6."""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\gitProjects\titanium-ataxx"

def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    train_args = (
        "--stage base "
        "--init /vol/ckpt/w256_d4_v7_sat/net.pt "
        "--width 256 "
        "--epochs 12 "
        "--bs 16384 "
        "--lr 1e-4 "
        "--lr-min 5e-6 "
        "--opt ranger "
        "--diff-nudge 0.0 "
        "--ema 0.9995"
    )
    cmd = [
        sys.executable,
        "-m", "modal", "run", "--detach",
        os.path.join(ROOT, "training", "modal_train.py"),
        "--tag", "w256_d4_v7_sat_deep",
        "--data-tag", "v7",
        "--args", train_args,
    ]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print("Process exited with code:", p.returncode)

if __name__ == "__main__":
    main()
