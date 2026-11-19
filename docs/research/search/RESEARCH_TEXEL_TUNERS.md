# Texel tuners from the chess-engine Discord (user paste 2026-11-02): what applies to Titanium (D315)

Source: a Discord thread the user pasted (Gedas texel-tuner, lynx-chess/texel-tuner PR #11, jw1912/hce-tuner, Ethereal tuner.c,
lichess-big3-resolved / zurichess / 4ku datasets, byte-knight +12.87 Elo). Read: the PR #11 page (WebFetch summary) and Titanium source.
NOT read: the PR diff, Ethereal tuner.c, hce-tuner source, the Reddit endgame-scale thread, the datasets.

## What Titanium's eval actually is (source, worktree mainpromo-ataxx = main c86733b)
- Leaf eval = the W256 NNUE (F5FA8557). `search.rs:751-757`: only TNN2 ("v2") nets take the hand eval as an S57 skip input;
  the current TNN3 net calls `eval_with_dis(b, 0)`, so the hand eval is NOT computed at leaves.
- The hand eval (`search.rs:222`, material + PST + tempo + contact penalty + multicapture penalty) survives in exactly one place:
  the futility SmallNet prefilter (`search.rs:1101`, `hand` is an input to `sn.bound_acc`). Frozen with SmallNet tau 0.02 (LAWS B11).
- The NNUE is already trained with a Texel-style loss in the hub (bullet, by jw, the same jw as hce-tuner):
  MSE between sigmoid(eval/400) and 0.75*sigmoid(cp5k/400) + 0.25*result (titaniumtest src/targets.rs).

## The thread's ideas, mapped
| Idea | In the thread | Titanium status | Worth doing? |
|---|---|---|---|
| Blend of WDL and sigmoided eval as target (lynx PR #11; byte-knight +12.87 Elo) | chess HCE | already our target (0.25 result); D311: LESS result = -32 Elo, 0.4 = +3.7 (3,200 g) | no new work; it confirms our D311 direction |
| Fit the sigmoid scale K; error is NOT convex in K (jw); Ethereal samples K on a grid and shrinks the range | chess HCE | `ti train export --calibrate` fits per-bucket scale+shift by Gauss-Newton from (1,0) and stops at the first non-improving step (titaniumtest src/export.rs:121-173): can stop in a local minimum | YES, cheap: grid-then-refine, and measure |
| Calibration side effect | - | calibrate RAISED val both times it was measured: rwphase1 0.011060 -> 0.011296 (+2.1%), grow256a 0.017797 -> 0.017936 (+0.8%) (D242/D310). I found no gate that isolates calibration (calibrated vs uncalibrated, same trunk). | YES: 5k gate sat16a/best+dis-refit (no calibrate) vs F5FA8557 |
| "Resolved" (quiet) datasets | chess HCE needs quiet positions | Ataxx has no quiet positions (BEHAVIOR VOLATILITY); our labels are 5k-search scores = already resolved | analog only: drop rows whose 5k score is unstable (nnue-pytorch WDL-skip idea, REVIEW_NOTES) |
| Adam bias correction (Gedas) | tuner detail | bullet Ranger optimiser handles it | no |
| Classic Texel of hand-eval constants | the main use of these tuners | hand eval only feeds the futility prefilter bound | low: possible later as a hub `ti texel` over the prefilter, frozen until gated |

## Plan (hub only, Rust; no Python)
1. Gate calibration itself: export sat16a/best.tnn with dis refit but WITHOUT --calibrate, 5k gate vs F5FA8557 (+ ti insight/analyze). If the
   uncalibrated net is equal or better, calibration is not a gain and every export should skip it.
2. If calibration matters: replace the local Gauss-Newton fit by an Ethereal-style grid over the scale (e.g. 0.5..1.5) and shift, then refine;
   log per-bucket before/after on a HELD-OUT slice (val), not only on the fit rows.
3. Not planned: porting a chess Texel tuner. The tuners tune linear HCE weights; our eval is an NNUE already trained on the same loss.
