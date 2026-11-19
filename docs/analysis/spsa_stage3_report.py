"""SPSA stage 3 progress report (docs/analysis). Parses logs/spsa_stage3pol.log and prints
per-knob default -> current theta_next, the delta, and the phase-3 search effect.

Run: python docs/analysis/spsa_stage3_report.py [log] [knobs]
"""
import re
import sys
import statistics as st

LOG = sys.argv[1] if len(sys.argv) > 1 else r"C:\gitProjects\logs\spsa_stage3pol.log"
KNOBS = sys.argv[2] if len(sys.argv) > 2 else r"C:\gitProjects\titanium-ataxx\tune\knobs_stage3_policy.txt"

DEF = dict(
    rfp1=257, rfp2=347, rfp3=478, rfp4=774, rfp_s1=32, rfp_s2=32, rfp_s3=32, rfp_s4=32, rfp_hinge=0,
    dis_knee=300, rfp_dismax=100000, ph_rfp0=64, ph_rfp1=64, ph_rfp2=64, ph_rfp3=64,
    ph_lmr0=64, ph_lmr1=64, ph_lmr2=64, ph_lmr3=64, ph_fut0=64, ph_fut1=64, ph_fut2=64, ph_fut3=64,
    lmr_half=47, lmr_min=6, fut_div=57, dis_fut=24, fut_hinge=0, fut_ext=1,
    pc_d=6, pc_r=3, pc_t=140, pc_k=32, pc_hinge=0, nmp_tgt=11, nmp_fill=26, nmp_r=3, nmp_dismax=100000,
    iir_d=4, corr_apply=64, corr_wmax=128, ord_clone=500, ord_jpen=500, ord_jconn=250,
    lmr_dis=0, lmr_hinge=0, fut_ddis=0, fut_dhinge=0, pol_w=100, pg_t=40, pg_d=10, pg_max=2,
)

bounds = {}
for line in open(KNOBS, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    p = line.split()
    bounds[p[0]] = (int(p[1]), int(p[2]))

rows = []
for line in open(LOG, encoding="utf-8", errors="replace"):
    if not line.startswith("k="):
        continue
    d = dict(re.findall(r"(\w+)=(-?[\d.]+)", line.split("theta_next=")[0]))
    if "theta_next" in line:
        tn = dict(re.findall(r"(\w+)=(-?[\d.]+)", line.split("theta_next=")[1]))
    else:
        tn = {}
    rows.append((int(d["k"]), float(d["score"]), int(d["nodes"]), tn))

if not rows:
    print("no iterations found")
    raise SystemExit(1)

print(f"iterations logged: {len(rows)}  (k {rows[0][0]}..{rows[-1][0]}, nodes={rows[-1][2]})")
last = rows[-1]
win = rows[-50:]
print(f"mean score last 50 iters: {st.mean(r[1] for r in win):.4f}  "
      f"(last iter {last[1]:.4f}, {last[1]*100:.1f}%)   <- noise floor is ~0.5 +-0.05 at 100 games")
print(f"theta_next at k={last[0]}\n")
hdr = f"{'knob':10s} {'def':>6s} {'theta':>6s} {'delta':>7s} {'%':>7s}  bounds"
print(hdr)
for k in bounds:
    if k not in last[3]:
        continue
    v = int(float(last[3][k]))
    d = DEF.get(k)
    lo, hi = bounds[k]
    pct = f"{100.0*(v-d)/d:+.0f}%" if d else ("-" if v == 0 else "inf")
    at = " AT-BOUND" if v in (lo, hi) else ""
    print(f"{k:10s} {d:6d} {v:6d} {v-d:+7d} {pct:>7s}  [{lo},{hi}]{at}")
