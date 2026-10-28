"""Docker Cloud SPSA 16-vCPU Tuner for Titanium Ataxx Search Selectivity (TweakSels).

Deploys a dedicated 16-vCPU sandbox (tit-spsa1), compiles the candidate titanium-cli
engine, and executes a multi-thousand iteration SPSA search parameter optimization
with 24 parallel workers against book100k_diverse.txt.

Cost: $0.07 / vCPU-h = $1.118 / h (charged against $250 Docker credit allowance).
Logs all spend to logs/docker_spend.csv. Automatically self-destructs on completion or ttl.
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
SANDBOX_NAME = "tit-spsa1"
VCPUS = 16
MEM_GB = 32
TTL_MIN = 120


def log_spend(status, cost_est):
    os.makedirs(os.path.dirname(SPEND_CSV), exist_ok=True)
    row = f"{time.strftime('%Y-%m-%d %H:%M')},{SANDBOX_NAME},{VCPUS},{MEM_GB},{VCPUS * PRICE_PER_VCPU_HR:.3f},{TTL_MIN},{cost_est:.3f},{status}\n"
    with open(SPEND_CSV, "a", encoding="utf-8") as f:
        f.write(row)


def run_cmd(cmd, check=True):
    print(f"+ {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and res.returncode != 0:
        print(f"FAILED: {res.stderr}")
        sys.exit(res.returncode)
    return res.stdout.strip()


def launch(iters=3000, workers=24, nodes=1500):
    print("=== LAUNCHING DOCKER CLOUD SPSA 16-vCPU TUNER ===")
    worst_cost = (TTL_MIN / 60.0) * VCPUS * PRICE_PER_VCPU_HR
    print(f"Sandbox: {SANDBOX_NAME} (16 vCPU, 32GB RAM, TTL {TTL_MIN}m, max cost: ${worst_cost:.3f})")
    log_spend("LAUNCHED", worst_cost)

    # 1. Create source tarball
    tar_path = "c:/gitProjects/main_src.tar.gz"
    if not os.path.exists(tar_path):
        run_cmd(f"git -C {MAIN} archive --format=tar.gz -o {tar_path} HEAD")

    # 2. Create sandbox
    run_cmd(
        f"sbx --cloud create --name {SANDBOX_NAME} --image-ref docker.io/library/rust:1.97 "
        f"--cpus {VCPUS} --memory {MEM_GB}g --ttl {TTL_MIN}m --on-timeout delete --platform linux/amd64"
    )

    # 3. Setup directory & upload
    run_cmd(f"sbx --cloud exec {SANDBOX_NAME} mkdir -p /tmp/spsa")
    print("Uploading source archive and book...")
    run_cmd(f"sbx --cloud cp {tar_path} {SANDBOX_NAME}:/tmp/spsa/main_src.tar.gz")
    run_cmd(f"sbx --cloud cp {BOOK} {SANDBOX_NAME}:/tmp/spsa/book100k_diverse.txt")
    run_cmd(f"sbx --cloud cp {ROOT}/training/spsa.py {SANDBOX_NAME}:/tmp/spsa/spsa.py")

    # 4. Compile inside container
    print("Compiling titanium-cli on 16 vCPUs...")
    build_script = (
        "cd /tmp/spsa && "
        "mkdir -p src && "
        "tar xf main_src.tar.gz -C src && "
        "cd src && "
        "cargo build --release -p titanium-cli"
    )
    run_cmd(f'sbx --cloud exec {SANDBOX_NAME} bash -c "{build_script}"')

    # 5. Launch detached SPSA process
    print(f"Starting SPSA tuning ({iters} iters, {workers} workers, {nodes} nodes)...")
    run_script = (
        f"cd /tmp/spsa && "
        f"nohup python3 spsa.py --bin src/target/release/titanium-cli "
        f"--book book100k_diverse.txt --iters {iters} --workers {workers} "
        f"--nodes {nodes} --out spsa_cloud_trace.csv > spsa.log 2>&1 < /dev/null &"
    )
    run_cmd(f'sbx --cloud exec {SANDBOX_NAME} bash -c "{run_script}"')
    print("=== DOCKER CLOUD SPSA TUNER SUCCESSFULLY DETACHED ===")


def status():
    out = run_cmd(f'sbx --cloud exec {SANDBOX_NAME} bash -c "tail -n 10 /tmp/spsa/spsa_cloud_trace.csv 2>/dev/null"', check=False)
    print("--- SPSA TRACE (LAST 10 ROWS) ---")
    print(out)
    log_tail = run_cmd(f'sbx --cloud exec {SANDBOX_NAME} bash -c "tail -n 5 /tmp/spsa/spsa.log 2>/dev/null"', check=False)
    print("--- SPSA LOG (LAST 5 ROWS) ---")
    print(log_tail)


def collect_and_delete(start_time_minutes=30):
    print("Collecting trace from sandbox...")
    run_cmd(f"sbx --cloud cp {SANDBOX_NAME}:/tmp/spsa/spsa_cloud_trace.csv {ROOT}/logs/spsa_cloud_trace.csv", check=False)
    run_cmd(f"sbx --cloud cp {SANDBOX_NAME}:/tmp/spsa/spsa.log {ROOT}/logs/spsa_cloud.log", check=False)
    actual_cost = (start_time_minutes / 60.0) * VCPUS * PRICE_PER_VCPU_HR
    log_spend(f"DELETED after ~{start_time_minutes} min (actual ~${actual_cost:.2f})", actual_cost)
    run_cmd(f"sbx --cloud rm --force {SANDBOX_NAME}")
    print(f"Sandbox {SANDBOX_NAME} deleted. Results collected in {ROOT}/logs/.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["launch", "status", "collect", "clean"])
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--nodes", type=int, default=1500)
    p.add_argument("--mins", type=int, default=30)
    args = p.parse_args()

    if args.action == "launch":
        launch(iters=args.iters, workers=args.workers, nodes=args.nodes)
    elif args.action == "status":
        status()
    elif args.action == "collect" or args.action == "clean":
        collect_and_delete(args.mins)
