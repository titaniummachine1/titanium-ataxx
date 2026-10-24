import json
import os
import re
import subprocess
import time

ROOT = r"C:\gitProjects\titanium-ataxx"
PY = "python"
TRAIN = os.path.join(ROOT, "training", "train_stream.py")
CACHE = os.path.join(ROOT, "data", "nnue", "stage1_base_train.npz")
VAL = os.path.join(ROOT, "data", "nnue", "stage1_base_val.npz")
INIT = os.path.join(ROOT, "data", "nnue", "tupS55tree.tup")
OUTDIR = os.path.join(ROOT, "data", "nnue", "autotune")
LOG = os.path.join(ROOT, "logs", "autotune_stage1.log")
os.makedirs(OUTDIR, exist_ok=True)


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_trial(tag, rows, epochs, cfg, init=INIT):
    out = os.path.join(OUTDIR, tag + ".tup")
    cmd = [
        PY, TRAIN,
        "--cache", CACHE,
        "--rows", str(rows),
        "--epochs", str(epochs),
        "--max-epochs", str(epochs),
        "--plateau", "0",
        "--init", init,
        "--out", out,
        "--val-probe", VAL,
        "--tcol", "teacher",
        "--outcome-weight", str(cfg.get("outcome", 0.5)),
        "--wdl-loss",
        "--wdl-scale", str(cfg.get("wdl_scale", 400.0)),
        "--opt", cfg["opt"],
        "--opt-lr", str(cfg["lr"]),
        "--conv-lr", str(cfg.get("conv_lr", cfg["lr"])),
        "--wd", str(cfg.get("wd", 1e-6)),
        "--threads", "4",
    ]
    if cfg.get("lookahead"):
        cmd += ["--lookahead", "--lookahead-k", str(cfg.get("lookahead_k", 5)),
                "--lookahead-alpha", str(cfg.get("lookahead_alpha", 0.5))]
    if cfg.get("cyclic_t0"):
        cmd += ["--cyclic-T0", str(cfg["cyclic_t0"]), "--cyclic-mult", "2",
                "--cyclic-lr-min", str(cfg.get("cyclic_min", 1e-5))]
    log("START %s rows=%d epochs=%d cfg=%s" % (tag, rows, epochs, json.dumps(cfg)))
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=7200)
    text = p.stdout or ""
    with open(os.path.join(OUTDIR, tag + ".log"), "w", encoding="utf-8") as f:
        f.write(text)
    vals = re.findall(r"val_mse\s+([0-9.]+)", text)
    if p.returncode != 0 or not vals:
        log("FAIL %s rc=%d vals=%s tail=%s" % (tag, p.returncode, vals, text[-500:].replace("\n", " | ")))
        return {"tag": tag, "cfg": cfg, "rows": rows, "epochs": epochs,
                "val": None, "ok": False, "out": out, "elapsed_s": time.time() - t0}
    val = float(vals[-1])
    log("DONE %s val=%.4f elapsed=%.0fs out=%s" % (tag, val, time.time() - t0, out))
    return {"tag": tag, "cfg": cfg, "rows": rows, "epochs": epochs,
            "val": val, "ok": True, "out": out, "elapsed_s": time.time() - t0}


def main():
    configs = [
        {"opt": "adamw", "lr": 1e-4},
        {"opt": "adamw", "lr": 2e-4},
        {"opt": "adamw", "lr": 4e-4},
        {"opt": "nadam", "lr": 2e-4},
        {"opt": "radam", "lr": 2e-4},
        {"opt": "adamw", "lr": 2e-4, "lookahead": True},
        {"opt": "nadam", "lr": 2e-4, "lookahead": True},
        {"opt": "adamw", "lr": 2e-4, "cyclic_t0": 2},
    ]
    results = []
    log("PHASE 1 fixed subset 1M, 1 epoch")
    for i, cfg in enumerate(configs):
        results.append(run_trial("p1_%02d" % i, 1000000, 1, cfg))
    good = sorted([r for r in results if r["ok"]], key=lambda r: r["val"])
    for r in good:
        log("RANK %s val=%.4f %s" % (r["tag"], r["val"], json.dumps(r["cfg"])))
    if not good:
        log("NO GOOD TRIALS")
        return
    top = good[:3]
    log("PHASE 2 top3 3M rows, 2 epochs")
    phase2 = []
    for i, r in enumerate(top):
        phase2.append(run_trial("p2_%02d" % i, 3000000, 2, r["cfg"], init=INIT))
    good2 = sorted([r for r in phase2 if r["ok"]], key=lambda r: r["val"])
    if not good2:
        log("NO GOOD PHASE2")
        return
    best = good2[0]
    log("BEST %s val=%.4f cfg=%s out=%s" % (
        best["tag"], best["val"], json.dumps(best["cfg"]), best["out"]))

    log("PBT exploitation/exploration from best checkpoint")
    pbt_cfgs = [
        dict(best["cfg"], lr=best["cfg"]["lr"] * 0.8),
        dict(best["cfg"], lr=best["cfg"]["lr"] * 1.2),
    ]
    pbt = []
    for i, cfg in enumerate(pbt_cfgs):
        pbt.append(run_trial("pbt_%02d" % i, 3000000, 1, cfg, init=best["out"]))
    good3 = sorted([r for r in pbt if r["ok"]], key=lambda r: r["val"])
    if good3 and good3[0]["val"] < best["val"]:
        best = good3[0]
        log("PBT WIN %s val=%.4f cfg=%s" % (best["tag"], best["val"], json.dumps(best["cfg"])))

    with open(os.path.join(OUTDIR, "autotune_results.json"), "w", encoding="utf-8") as f:
        json.dump({"phase1": results, "phase2": phase2, "pbt": pbt,
                   "best": best, "created_at": time.time()}, f, indent=2)

    log("FINAL full stage1 from best checkpoint, 6 epochs")
    final = run_trial("final_stage1", 0, 6, best["cfg"], init=best["out"])
    log("FINAL RESULT %s" % json.dumps(final))
    log("autotune complete")


if __name__ == "__main__":
    main()
