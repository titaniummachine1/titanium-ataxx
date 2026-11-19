# ace apex: technical overview (as of 26 Sep 2026)

Quoridor engine: AlphaZero-style MCTS guided by a nanochat-style transformer, trained by self-play plus deep-search "golden" data.
Current champion: **apex512_gdose1** (sha cd56279c).

---

## 1. The network

| Item | Value |
|---|---|
| Architecture | nanochat GPT (karpathy/nanochat, commit 92d63d4), adapted to the board |
| Layers x width x heads | **8 x 512 x 8** |
| Parameters | **25,619,857** (weight matrices 25,619,840) |
| Input tokens | **145** = 81 square tokens (14 feature planes each) + 64 wall-intersection tokens (4 wall values + the 4 neighbouring squares' planes = 60 features) |
| Token embedding | one Linear per token type (no vocabulary) |
| Position encoding | 2-D rotary (rows on half the head dims, columns on the other half; intersections at +0.5,+0.5), base 100 |
| Attention | bidirectional (no causal mask), QK-norm with 1.2 sharpening |
| Block | pre-RMSNorm (no affine), no biases, relu^2 MLP x4, value embeddings on alternating layers with a sigmoid gate, per-layer residual/x0 lambdas, mid-layer residual backout |
| Policy head | per token: each square -> pawn-move logit, each intersection -> horizontal + vertical wall logits = **209 actions** (81 + 64 + 64); logit softcap 15 |
| Value head | mean-pooled tokens -> norm -> relu^2 MLP -> tanh, one value in [-1, +1] for the side to move |
| Symmetry | the board is encoded from the side to move's point of view; left-right mirror used as augmentation and to merge duplicate positions |

Why 8 x 512 (derived, see ledger SIZE-DERIVATION-2): depth from path-finding (attention doubles reach per layer; paths < 81 steps -> 7 layers + 1 to decide = 8);
width from nanochat's aspect rule (64 x depth = 512); 12 x L x d^2 = 25.17M matrix parameters, and the data we can generate (~101M distinct rows x 3 passes ~ 303M)
matches nanochat's 12x-parameters training horizon. The net is far from full (~50-90M bits capacity vs a small fraction used).

## 2. The search

- **Rust PUCT MCTS** (`qmcts-core`), batched: the search collects leaves, the GPU evaluates them in one batch, results are fed back. Virtual loss with 4-8 leaves in flight.
- **Exact race solver**: once both players' paths can't be blocked any more, positions are solved exactly (no net needed).
- **Proven win/loss backups** and **resignation** when proven or hopeless.
- **No tree reuse** in the gate and deep-search tools (each move searched fresh).
- Throughput on the laptop (RTX 4060): ~4,700 sims/s in batched self-play (torch.compile, 1.6x); a single deep search ~1,300 sims/s.
- Deep searches: **1.3M sims per move ~ 17-20 min**; 100k ~ 80 s.

## 3. Training loop

**Self-play generation** (`python/selfplay.py`): 50 sims per move, Dirichlet noise 0.3 re-applied every move ("renoise"), temperature for the first 8 plies,
fp32, starts taken from real positions (deep-search trees, both sides to move). ~86,000 games / ~1.4M rows in ~4 h on the laptop.

**Training** (`python/train.py`, and `research/train_hooked.py` = the same + per-row sampling weights):
- Continues from the champion (never from scratch); optimizer MuonAdamW (Muon for matrices, AdamW for the rest).
- Learning rate **0.05 x nanochat's base** (matrices 0.001, embeddings 0.015, head 0.0004, scalars 0.025): warmup 40 steps, flat, then linear warmdown over the last 65% to 5%.
  (Full nanochat LR on a warm start was tried and failed: NANO-LR-1.)
- Batch 1,024 (micro-batch 32), 1.25 passes over the window, ~2,000 steps, ~50 min.
- Window: newest self-play rows + the previous loop's rows (~27% replay) + golden rows.
- Value target = 0.5 x game result + 0.5 x search value; policy target = search visit counts.
- Rows are **deduplicated by position (mirror-folded)**: visits are summed, values averaged.

**Golden rows** (the method that produced the last two champions):
- Deep searches at **1.3M sims per move** along the main opening line and at the face-off; every position they searched with **>= 10,000 visits within 6 plies** becomes a golden row (1,547 positions).
- Target = that deep search's **own visit counts** (not first-guess x Q: that "action policy" target keeps blind-spot moves starved).
- Each golden row written **20 times** (so its value isn't outvoted when merged with a few self-play rows) and oversampled to **3-10% of training draws**.
- Result: the face-off move **a3h** went from 0.03% of the net's first guess to 99.9%, and the face-off value from +0.20 to +0.56.
- Lesson learned: lightly searched deep positions (500+ visits) hurt as training targets, whether dosed or not; only well-searched positions help.

**Sweeps** (finding blind spots): every legal move at a position gets its own 100k search (so the net can't hide a move from itself), then every move within 0.1
of the best wall gets 1.3M. Running now for P1 at plies 2, 4 and 6 of the march.

## 4. Evaluation (how a net is promoted)

- **Gate**: new net vs champion, **640 games** = 320 opening positions x both seats, 400 sims per move, paired bootstrap confidence interval (`paired_net_match.py`).
- Opening bank: bare board, 2-ply starts (`openings_bare2_v1`).
- **Promotion rule (owner)**: score > 50% vs the champion promotes. Stop early only if promotion is practically impossible (< 1%), or non-promotion mathematically impossible.
- Fixed yardstick: pinned **b3** (older champion); latest partial reading: champion 54.1% vs b3 (196 games).
- Diagnostics (never promote): held-out loss, the a3h face-off check, P0/P1 split.

## 5. Current results (26 Sep)

| Net | vs | Score | Note |
|---|---|---|---|
| golden2 | d2apol | 50.23% | first net playing a3h; promoted |
| **gdose1** (champion) | golden2 | **50.70%** | same recipe at 10% golden dose; promoted |
| golden3 / set1 / mix1 | champion's line | 48.4 / 47.5 / 47.5% | whole-tree data: not promoted |
| gdose1 | b3 (anchor) | 54.1% (partial) | stopped at 196 games by the owner |

P0/P1: between top nets P0 wins ~77-95% of games; P1 is the frontier (sweeps + P1 play-out test running).

## 6. Hardware and running costs

**Laptop** (owned): AORUS 15, RTX 4060 laptop GPU, 8 GB RAM (single stick; a second 8 GB stick ~ GBP 20-30 would double RAM and enable dual channel).
- Power under load ~150 W -> ~0.15 kWh/h -> at ~GBP 0.25/kWh **~GBP 0.04 per hour, ~GBP 1 per day** running 24/7.
- One full loop (generate ~4 h + train ~50 min + gate ~2.3 h) ~ 7 h ~ **GBP 0.30** of electricity.
- A 24-ply deep line at 1.3M/move ~ 8 h ~ GBP 0.30. A full every-move sweep of one position ~ 1.5-2.5 h.

**Rentals** (vast.ai, owner-approved only):
| Box | Price (scan 26 Sep) | Use |
|---|---|---|
| 1x RTX 4090, 252 GB RAM (owner's pick) | ~$0.53/h | sweeps, 13M-50M deep lines (big trees need RAM) |
| 4x RTX 5090, 504 GB RAM | ~$1.89/h | parallel sweeps + deep lines |
| (past) 4x RTX 5090 box | $1.55/h | the 8x512 campaign, 7-14 Sep |

Planned rental: nested wall-pair sweeps of plies 0-24 (P1 first) ~25-28 h ~ **$14-15** of the owner's $20.
Future "final" job: a 1-billion-sim first move for both sides (~1-2 days on a 4090, needs a memory-light tree first).

**Playing apex**: the Board page runs the net in the browser with WebGPU (free; the weights are downloaded to the browser, so anyone with the link can copy them).
A hosted server (weights private) would need a rented GPU: roughly $0.40-0.80/h on demand.

## 7. Where things live

- Code: `C:\Users\alex\qmcts` (`python/` net, training, self-play; `rust/qmcts-core` search; `tools/` deep search, sweeps, golden builder; `research/` weighted trainer).
- Nets: `data/nets/` (champion `apex512_gdose1.pt`, anchor `apex512_b3.pt`).
- Record of every experiment, result and retraction: `LEDGER.md`. Restart instructions: `docs/RESUME_20260926.md`.
