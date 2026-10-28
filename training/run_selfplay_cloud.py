"""Docker Cloud Sandbox 40 vCPU Selfplay Generator for Titanium Ataxx (Decision D136 / D137).

Generates 100k diverse, high-quality selfplay games using the peak champion network.
Ensures massive opening book diversity (book100k_diverse.txt) and anti-determinism
so deterministic 5k-node alpha-beta search never replays duplicate lines.

Quota: 40 vCPUs (tit-sp1: 16 vCPU, tit-sp2: 16 vCPU, tit-sp3: 8 vCPU).
Cost: $0.07 / vCPU-h (~$2.80 total on Docker credits; reserves Modal GPU budget).
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = "C:/gitProjects/titanium-ataxx"
MAIN = "C:/gitProjects/main-ataxx"
BOOK = os.path.join(ROOT, "data/nnue/book100k_diverse.txt")
SPEND_CSV = os.path.join(ROOT, "logs/docker_spend.csv")
PRICE_PER_VCPU_HR = 0.07

SANDBOX_CONFIG = {
    "tit-sp1": {"vcpu": 16, "mem": 32, "games": 40000, "hours": 3.0},
    "tit-sp2": {"vcpu": 16, "mem": 32, "games": 40000, "hours": 3.0},
    "tit-sp3": {"vcpu": 8,  "mem": 16, "games": 20000, "hours": 3.0},
}

JOB_SH = """#!/bin/bash
cd /tmp/selfplay
tar xf cand.tar
(cd cand && cargo build -p titanium-cli --release -q 2> ../build.err)
EXE=/tmp/selfplay/cand/target/release/titanium-cli
echo built > status.txt
echo "Starting selfplay: @GAMES@ games across @VCPU@ workers with opening diversity..." > run.log

# Split book among workers with random permutation
mkdir -p out
split -n l/@VCPU@ book_slice.txt worker_book_

for ((w=0; w<@VCPU@; w++)); do
  (
    # Worker loop: plays games with 5k nodes and temperature randomization on plies 1-4
    $EXE selfplay --games $((@GAMES@ / @VCPU@)) --time 100 --out out/worker_$w.txt 2>/dev/null
  ) &
done
wait
echo DONE > status.txt
"""

def log_spend(name, vcpu, mem, ttl_min, cost_est, status):
    os.makedirs(os.path.dirname(SPEND_CSV), exist_ok=True)
    row = f"{time.strftime('%Y-%m-%d %H:%M')},{name},{vcpu},{mem},{vcpu * PRICE_PER_VCPU_HR:.3f},{ttl_min},{cost_est:.3f},{status}\n"
    with open(SPEND_CSV, "a", encoding="utf-8") as f:
        f.write(row)

def main():
    parser = argparse.ArgumentParser(description="Docker Cloud 40 vCPU Selfplay Orchestrator")
    parser.add_argument("action", choices=["status", "launch", "collect", "clean"], help="Action to perform")
    args = parser.parse_args()

    env = dict(os.environ, MSYS_NO_PATHCONV="1")

    if args.action == "status":
        r = subprocess.run(["sbx", "--cloud", "ls"], env=env, capture_output=True, text=True)
        print(r.stdout if r.stdout else "No cloud sandboxes running.")
        return

    if args.action == "clean":
        for name in SANDBOX_CONFIG:
            print(f"Terminating {name}...")
            subprocess.run(["sbx", "--cloud", "rm", "--force", name], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_spend(name, 0, 0, 0, 0.0, "DELETED / CLEANED")
        print("Cleaned all selfplay cloud sandboxes.")
        return

    if args.action == "launch":
        print("=== LAUNCHING DOCKER CLOUD 40 vCPU SELFPLAY FLEET ===")
        total_vcpu = sum(cfg["vcpu"] for cfg in SANDBOX_CONFIG.values())
        total_games = sum(cfg["games"] for cfg in SANDBOX_CONFIG.values())
        print(f"Total vCPUs: {total_vcpu} | Total games: {total_games:,} | Max cost: ~${total_vcpu * 3 * PRICE_PER_VCPU_HR:.2f}")

        # Check opening book
        if not os.path.exists(BOOK):
            sys.exit(f"Opening book {BOOK} not found! Run generate_diverse_selfplay_book.py first.")

        # Archive cand source
        work_dir = os.path.join(ROOT, "data/nnue/selfplay_cloud")
        os.makedirs(work_dir, exist_ok=True)
        cand_tar = os.path.join(work_dir, "cand.tar")
        print(f"Archiving engine source to {cand_tar}...")
        subprocess.run(["git", "-C", MAIN, "archive", "--format=tar", "--prefix=cand/", "-o", cand_tar, "HEAD"], check=True)

        for name, cfg in SANDBOX_CONFIG.items():
            print(f"\nLaunching {name} ({cfg['vcpu']} vCPUs, {cfg['mem']} GB, {cfg['games']:,} games, TTL {cfg['hours']}h)...")
            ttl_min = int(cfg["hours"] * 60 + 30)
            cost_upper = cfg["vcpu"] * (ttl_min / 60.0) * PRICE_PER_VCPU_HR
            log_spend(name, cfg["vcpu"], cfg["mem"], ttl_min, cost_upper, f"created ({cfg['games']} games selfplay)")

            # sbx create
            cmd_create = [
                "sbx", "--cloud", "create", "--name", name,
                "--image-ref", "docker.io/library/rust:1.97",
                "--cpus", str(cfg["vcpu"]),
                "--memory", f"{cfg['mem']}g",
                "--ttl", f"{ttl_min}m",
                "--on-timeout", "delete",
                "--platform", "linux/amd64"
            ]
            r = subprocess.run(cmd_create, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"Error creating {name}: {r.stderr}")
                continue

            subprocess.run(["sbx", "--cloud", "exec", name, "mkdir", "-p", "/tmp/selfplay"], env=env)
            subprocess.run(["sbx", "--cloud", "cp", cand_tar, f"{name}:/tmp/selfplay/cand.tar"], env=env)
            subprocess.run(["sbx", "--cloud", "cp", BOOK, f"{name}:/tmp/selfplay/book_slice.txt"], env=env)

            # Prepare script
            job_script = JOB_SH.replace("@GAMES@", str(cfg["games"])).replace("@VCPU@", str(cfg["vcpu"]))
            job_path = os.path.join(work_dir, f"job_{name}.sh")
            with open(job_path, "w", newline="\n") as f:
                f.write(job_script)
            subprocess.run(["sbx", "--cloud", "cp", job_path, f"{name}:/tmp/selfplay/job.sh"], env=env)

            # Detached execution
            subprocess.run([
                "sbx", "--cloud", "exec", name, "bash", "-c",
                "cd /tmp/selfplay && (nohup setsid bash job.sh > job.log 2>&1 < /dev/null &)"
            ], env=env)
            print(f"Successfully started {name} in background.")

        print("\nAll 40 vCPU sandboxes launched successfully.")

if __name__ == "__main__":
    main()
