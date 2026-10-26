"""NNUE training on a Modal GPU (D80): the same training/train_nnue.py, TI_DEVICE=cuda, data rebuilt inside the container.

Only small files are uploaded (volume `ti-train`, /up): the slim label npz (data/nnue/modal_up/final_{train,val}_v4.npz = the
columns train_nnue.Split reads) and the init / resume checkpoints. The 12.3 GB feature index is NOT uploaded: each container
writes the raw occ0/occ1/stm files from the npz and runs the ENGINE's own `titanium-cli nndump` (main-ataxx source built in
the image), then checks sha256 against the local data/nnue/nnfeat files and refuses to train on a mismatch.
Checkpoints, the resume file and the log live on the volume (/ckpt), committed every few minutes.

  modal volume put ti-train data/nnue/modal_up/final_train_v4.npz /up/final_train_v4.npz   (once; also val + init .pt)
  python -m modal run training/modal_train.py --tag smoke --args "<train_nnue args>"           (attached)
  python -m modal run --detach training/modal_train.py --tag base_v4s --args "<train_nnue args>" (long run)
  modal volume get ti-train /ckpt/<tag> data/nnue/modal_ckpt/
Cost: L4 $0.80/h + 4 cores $0.19/h + 24 GiB $0.19/h = ~$1.2/h (Modal pricing page 2026-10-25); hard timeout 4 h per call (relaunch to continue; the trainer resumes).
"""
import os

import modal

ROOT = "C:/gitProjects/titanium-ataxx"
ENGINE_SRC = os.environ.get("ENGINE_SRC", "C:/gitProjects/main-ataxx")
IGNORE = ["target", "data", "logs", "web", "training", "docs", "downlaods", "botzone", "**/*.pt", "**/*.npz", "node_modules", ".git"]
CARGO = "/root/.cargo/bin/cargo"
# sha256 of the LOCAL feature files (data/nnue/nnfeat, 2026-10-25): the container's nndump output must match exactly
SHA = {
    "train.cnt": "ed9cac3fb156c3cae30b204d241cbd735cbbfd260d6f8e4c3eb2baf8a7d3e99b",
    "train.idx": "80452e0b9452081004045c1285b20963525c32bf2f06a77d6ba42b8bc95eea3e",
    "val.cnt": "b621f279a867a78569b0475ed67607eceaff40be77f589d6177161c5ceb36a35",
    "val.idx": "b6b680e06c4433aa464908ff244ddb65ae65396dab03d25bddbb5bec165c97c3",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .add_local_dir(ENGINE_SRC, "/eng", copy=True, ignore=IGNORE)
    .run_commands("cd /eng && %s build -p titanium-cli --release" % CARGO, "sha256sum /eng/target/release/titanium-cli")
    .pip_install("torch==2.4.1", "numpy==2.2.6", "optuna==5.0.0")
    .add_local_file(ROOT + "/training/train_nnue.py", "/app/train_nnue.py")
    .add_local_file(ROOT + "/training/optuna_ft.py", "/app/optuna_ft.py")
)
vol = modal.Volume.from_name("ti-train", create_if_missing=True)
app = modal.App("titanium-train", image=image)
DATA = "/data/ti"


def build_features():
    """npz -> raw occ files -> engine nndump -> sha256 check (exactly the local feature files)."""
    import hashlib
    import shutil
    import subprocess
    import time
    import zipfile

    import numpy as np
    import numpy.lib.format as F
    fin, feat = DATA + "/data/nnue/final", DATA + "/data/nnue/nnfeat"
    os.makedirs(fin, exist_ok=True)
    os.makedirs(feat, exist_ok=True)
    for sp in ("train", "val"):
        t0 = time.time()
        shutil.copy("/vol/up/final_%s_v4.npz" % sp, "%s/final_%s_v4.npz" % (fin, sp))
        with zipfile.ZipFile("%s/final_%s_v4.npz" % (fin, sp)) as z:
            for c in ("occ0", "occ1", "stm"):
                with z.open(c + ".npy") as f:
                    F.read_array(f).tofile("%s/%s.%s" % (feat, sp, c))
        r = subprocess.run(["/eng/target/release/titanium-cli", "nndump", "--in", "%s/%s" % (feat, sp), "--out", "%s/%s" % (feat, sp)],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("nndump %s failed: %s" % (sp, r.stderr[-1000:]))
        for ext in ("cnt", "idx"):
            h = hashlib.sha256()
            with open("%s/%s.%s" % (feat, sp, ext), "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            ok = h.hexdigest() == SHA["%s.%s" % (sp, ext)]
            print("features %s.%s sha256 %s %s" % (sp, ext, h.hexdigest()[:16], "MATCHES local" if ok else "MISMATCH"), flush=True)
            if not ok:
                raise RuntimeError("feature file %s.%s differs from the local one - refusing to train" % (sp, ext))
        print("features %s ready in %.0fs" % (sp, time.time() - t0), flush=True)


@app.function(gpu="L4", cpu=4.0, memory=24576, timeout=4 * 3600, volumes={"/vol": vol})
def train(tag: str, args: str, script: str = "train_nnue.py"):
    import subprocess
    import threading
    import time
    build_features()
    out = "/vol/ckpt/" + tag
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, TI_ROOT=DATA, TI_DEVICE="cuda", TI_DATA_TAG="v4")
    stop = threading.Event()

    def committer():
        while not stop.wait(60):  # every minute, so a stopped call loses at most ~1 min
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    # train_nnue.py writes its checkpoints under /ckpt/<tag>/net*; optuna_ft.py gets --out from args (also /vol/ckpt/...)
    cmd = ["python", "-u", "/app/" + script] + args.split() + (["--out", out + "/net"] if script == "train_nnue.py" else [])
    print("+", " ".join(cmd), flush=True)
    with open(out + "/log.txt", "a") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd="/app")
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
        rc = p.wait()
    stop.set()
    vol.commit()
    print("trainer exit code", rc, flush=True)
    return rc


@app.local_entrypoint()
def main(tag: str, args: str, script: str = "train_nnue.py"):
    print("exit code", train.remote(tag, args, script))
