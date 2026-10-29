"""Launch Modal W=256 D4 saturation / noise-cleaning training on v7 dataset."""
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
        "--init /vol/up/w256_downscaled_ep7.pt "
        "--out /vol/ckpt/w256_d4_v7_sat "
        "--width 256 "
        "--epochs 6 "
        "--bs 16384 "
        "--lr 2e-4 "
        "--lr-min 1e-5 "
        "--opt ranger "
        "--diff-nudge 0.0 "
        "--ema 0.9995"
    )
    cmd = [
        sys.executable,
        "-m", "modal", "run", "--detach",
        os.path.join(ROOT, "training", "modal_train.py"),
        "--tag", "w256_d4_v7_sat",
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
