"""TSTG model registry (D416): hash every .ttst, copy each distinct content ONCE to data/nnue/tstg_store/tstg_<SHA8>.ttst
(read-only), and write docs/TSTG_MODELS.tsv (name -> sha8, size, mtime, store path, status). Re-run after any new model.

    python -I docs/analysis/tstg_model_registry.py

Gates must reference the store path (immutable, content-addressed), never a mutable name like tstg_best.ttst.
Status strings come from STATUS below (verified in docs/research/hce-patterns-speed/TSTG_VERIFICATION_2026_11_06.md); unknown files are UNVERIFIED.
"""
import datetime
import glob
import hashlib
import os
import shutil
import stat

ROOT = "C:/gitProjects/titanium-ataxx"
STORE = ROOT + "/data/nnue/tstg_store"
STATUS = {
    "tstg_distil_best.ttst": "D440 sibling distillation (train_v9 + 18.5M NNUE-labelled children, 14-15 epochs from tstg_fix_best)",
    "tstg_fix_disneutral.ttst": "D438 = tstg_fix_best with the dis head set to the NNUE per-phase mean dis (w_dis 0): isolation test of the dis scale (TSTG mean dis 299 vs NNUE 99 cp)",
    "tstg_fix_best.ttst": "D434 colour-fix retrain of v10_best (20 epochs, 11 min Modal CPU): val corr 0.579 (NNUE 0.666), MAE 284; policy head = hand defaults (untrained)",
    "tstg_v10_best.ttst": "VALUE v9-trainer (14.5% rows colour-swapped, D414); policy head = hand defaults (untrained); val corr 0.511 (NNUE 0.666); pure-TSTG 0/14 games",
    "tstg_v10_pol_dis_fit.ttst": "value = v10_best (val identical) + fitted 24 policy + dis floats (5e5e445); NEVER GATED",
    "tstg_v10_polfit.ttst": "value = v10_best + fitted 24 policy floats (2e279c2); NEVER GATED",
    "tstg_v23_saturated_master.ttst": "value = v10_best (val identical); pure-TSTG 0/18 at 5k",
    "tstg_v24_champion_master.ttst": "CORRUPT: hand-overwritten by build_champion_tstg (MAE vs NNUE 531 cp); pure-TSTG 0/40",
    "tstg_best.ttst": "CORRUPT: overwritten by build_champion_tstg (= v24)",
    "tstg_v17_champion.ttst": "val corr 0.312; pure-TSTG 0/10",
    "tstg_v22_hybrid_master.ttst": "value = v17 (val identical)",
    "tstg_v11_saturated.ttst": "val corr 0.304 (data/ copy); backbone used by build_champion_tstg",
    "tstg_v11_final.ttst": "val corr 0.269; as ORDERING head over NNUE 39% (198 g, 11-04)",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8].upper()


os.makedirs(STORE, exist_ok=True)
rows = []
for p in sorted(glob.glob(ROOT + "/data/*.ttst") + glob.glob(ROOT + "/data/nnue/*.ttst")):
    s = sha(p)
    dst = f"{STORE}/tstg_{s}.ttst"
    if not os.path.exists(dst):
        shutil.copy2(p, dst)
        os.chmod(dst, stat.S_IREAD)
    p = p.replace("\\", "/")
    rel = p[len(ROOT) + 1:]
    name = os.path.basename(p)
    st = STATUS.get(name, "UNVERIFIED")
    if rel == "data/nnue/tstg_v11_saturated.ttst":
        st = "UNVERIFIED (different content from data/tstg_v11_saturated.ttst)"
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
    rows.append((rel, s, os.path.getsize(p), mt, dst[len(ROOT) + 1:], st))

with open(ROOT + "/docs/TSTG_MODELS.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("# TSTG model registry (D416). Regenerate: python -I docs/analysis/tstg_model_registry.py. Gate ONLY store paths.\n")
    f.write("path\tsha8\tbytes\tmtime\tstore\tstatus\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "files,", len({r[1] for r in rows}), "distinct contents ->", STORE)
