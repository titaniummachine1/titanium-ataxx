"""Regenerates the README graphs in assets/ from the numbers recorded in docs (HANDOFFS / DECISIONS D-numbers in the dev repo).

    python tools/make_readme_graphs.py        # needs matplotlib; writes assets/kata_arena.png and assets/elo_steps.png

Numbers (all paired, seat-balanced games from the Titanium hub; sources in the dev repo's docs/KATA_ARENA.md and docs/DECISIONS.md):
  Kata arena, measured 2026-10-31 with the older net 7AE0122F, Botzone end rules, 1837 games, Kata on a Modal L4 GPU:
    Titanium nodes 1000/2500/5000/10000/20000 = ~7/20/40/83/166 ms -> Elo 0/+137/+315/+422/~+470
    Kata visits 1/2/4/8/16 = ~13/17/22/37/~78 ms -> Elo -30/+233/+283/+320/~+500
  Strength steps, each vs the previous main at 5k nodes unless noted: net 54A6B02B +23.1, net 46FCB5A4 +28.6 +-17.9 (1,230 g),
    policy-head ordering +71.0 +-30.2 (486 g); 100 ms on the PC +125.7 +-43.8 (274 g, vs 46FCB5A4 without the head).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
os.makedirs(OUT, exist_ok=True)
INK, GRID, TI, KA = "#1f2937", "#d1d5db", "#2563eb", "#dc2626"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=INK)


def kata_arena():
    ti_ms, ti_elo = [7, 20, 40, 83, 166], [0, 137, 315, 422, 470]
    ka_ms, ka_elo = [13, 17, 22, 37, 78], [-30, 233, 283, 320, 500]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
    style(ax)
    ax.plot(ti_ms, ti_elo, "o-", color=TI, lw=2, label="Titanium (1k, 2.5k, 5k, 10k, 20k nodes)")
    ax.plot(ka_ms, ka_elo, "s--", color=KA, lw=2, label="KataAtaxx on a GPU (1, 2, 4, 8, 16 visits)")
    ax.set_xscale("log")
    ax.set_xticks([7, 10, 20, 40, 80, 160])
    ax.set_xticklabels(["7", "10", "20", "40", "80", "160"])
    ax.set_xlabel("time per move (ms, log scale)", color=INK)
    ax.set_ylabel("Elo (Titanium 1k nodes = 0)", color=INK)
    ax.set_title("Titanium vs KataAtaxx: Elo per time spent (measured 2026-10-31, older net)", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kata_arena.png"))
    plt.close(fig)


def elo_steps():
    labels = ["Result weight 0.55\n(net 54A6B02B)", "Result weight by phase\n(net 46FCB5A4)", "Policy-head ordering\n(5k nodes)", "Policy-head ordering\n(100 ms, on the PC)"]
    elo = [23.1, 28.6, 71.0, 125.7]
    err = [None, 17.9, 30.2, 43.8]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    style(ax)
    ax.grid(False, axis="y")
    ys = list(range(len(labels)))[::-1]
    for y, e, d, c in zip(ys, elo, err, [TI, TI, TI, "#059669"]):
        ax.barh(y, e, color=c, height=0.55, xerr=d, error_kw=dict(ecolor=INK, capsize=3, lw=1))
        ax.text(e + (d or 0) + 3, y, "+%.1f" % e + (" ±%.1f" % d if d else ""), va="center", color=INK, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlim(0, 190)
    ax.set_xlabel("Elo gain over the previous main (paired games; bars = 95% interval where recorded)", color=INK, fontsize=8)
    ax.set_title("Strength steps of 2 November 2026", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "elo_steps.png"))
    plt.close(fig)


if __name__ == "__main__":
    kata_arena()
    elo_steps()
    print("wrote", OUT)
