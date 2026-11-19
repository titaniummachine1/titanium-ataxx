# Strength plan, evening 2026-11-01 (D310). Ranked; each step says what done looks like. Executable by any model.

Main: exe c86733b, net F5FA8557 (sat16a_cal, W256). Botzone 4 s/turn. Why the eval is the lever: losses to sotax / PyJataxx come
from late-middlegame eval lag of 6-10 turns, not depth (D278, D210). Speed: profile already exists (D241: eval = 74% of search
time, 1.40-1.48 evals per node); samply needs admin on Windows and is NOT needed. Botzone time margin is tuned (user, D309): leave it.
Budget: Modal $29.39 left for November (user screenshot 20:17), Docker ~$3.6 of $200. Rules: docs/LAWS.md (hub only, no Python
training/tuning; gate rule P >= 95% promote else extend).

## 1. Lambda-by-phase net (RUNNING, D310)
- Idea: the 5k-label/result blend uses result weight 0.25 everywhere; vs 200k truth the best weight is 0.35 at <=6 empties,
  0.04 at 7-12, 0.00 at >=13 (docs/research/strategy/RESEARCH_2026_11_01.md sec. 2). Too much result noise in the middlegame = the eval lag.
- Built: hub titaniumtest 75bf96e `ti train run --result-weight-phase a,b,c` (training targets + val on the run's own target) and
  (uncommitted) `ti train export --result-weight-phase` (provenance re-measure + Texel calibration on the same target).
- Trained: Modal tag rwphase1 (init /vol/runs/sat16a/best.tnn, sat16a recipe + weights 0.35,0.04,0.0), early stop sb 29,
  own-target val 0.011200 (sb 0) -> 0.011060. Exported data/nnue/proxy/rwphase1_cal.tnn sha 6867B14A (+ .prov), calibrated,
  dis head refit, engine loads it. Local copy of the run: runs/rwphase1/rwphase1.
- Gate (sandbox tit-rw1, files C:\gitProjects\logs\sbx_rw1, results copied to its res/): same exe c86733b both sides,
  candidate net 6867B14A vs F5FA8557; legs 5k nodes 800 g (book0 64000), 100 ms 400 g (66000), 1000 ms 200 g (68000).
  Watch: `sbx --cloud exec tit-rw1 bash -c "cat /tmp/g/status.txt /tmp/g/gates/status.txt; tail -2 /tmp/g/gates/rw_*.out" < /dev/null`.
  Copy out res (logs + .games.jsonl via `cat`/base64 if `sbx cp` download fails), delete the sandbox, ledger row.
- Same architecture as main, so speed is identical: the 5k leg can decide alone (LAWS B9 / law 2); time legs confirm.
- If ahead but undecided: extend. If clearly ahead at P >= 95%: report, the user decides promotion.
- RESULT (D311): 0.35/0.04/0.00 = -32.2 +-23.6 at 5k (800 g, LOS 0.4%): less result in the middlegame HURTS (the 200k 'truth' shares the horizon blind spot; the result sees the future). Opposite direction 0.4/0.4/0.4 (rwphase2_cal 8C7A2519) = +3.5 +-23.6 (800 g, LOS 61%), extending +2400 g in tit-rw1 (gates/rw2b_5k.out). The run early-stopped at sb 3 (small change); a bigger move needs a higher weight (0.5-0.6) or a longer run without --plateau.
- Sandbox tit-rw1 ttl ends ~21:40 local (created 20:40); delete it when the extension is done and copy gates/rw2b_5k.* to C:\gitProjects\logs\sbx_rw1\res first (big files in 300 KB base64 chunks: the exec stream drops large outputs).

## 2. Hard-position ("regret") mining (NEXT, design)
User question 2026-11-01: use Titanium / Zataxx to beat Titanium, counterfactual or regret training? Answer: yes, as HARD-POSITION
MINING with Titanium as the label source:
- Source positions: the plies just before Titanium's own eval swung against it in LOST games: Botzone losses (data/botzone/),
  gate games vs Moonbird / Zataxx (they beat Titanium at 100 ms, so their games hold Titanium's real misjudgements) and
  self-play gate logs (logs/gates/*.games.jsonl, 67 files, 256 MB). `ti analyze` already finds turning points.
- Labels: Titanium deep search (50k-200k nodes, `ti relabel --nodes N --net F5FA8557`), never the opponent's score (user rule:
  opponents are teachers of positions, not of short-search labels). Zataxx is better per second but worse per node (+81 for
  Titanium at 5k, D166), so a deep Titanium label beats a Zataxx label.
- Train: fine-tune with these rows mixed into v9 (weight via wsum), same recipe, gate as in step 1. Needs: a `ti mine` (or
  `ti analyze --dump-positions`) subcommand that writes positions.bin of the k plies before each turning point. Hub, Rust.
- Counterfactual: for each mined position also add the position after the move the deep search prefers (the line Titanium missed).

## 3. SPSA stage 1 (RUNNING on the PC since ~19:00)
C:\gitProjects\logs\spsa_stage1.log, 160 iterations of 100 games (5000/5000/5000/50000 nodes). Resume: command in
C:\gitProjects\logs\spsa_stage1.cmd.txt + `--resume`. At the end: gate tuned theta vs untouched main at 100 ms + 1000 ms (Docker).
If the net changes (step 1/2 promoted), rerun on the new net.

## 4. Speed (CPU, PC after SPSA or Docker)
D241: 1.40-1.48 evals per node. Find why a node evaluates more than once (static eval + futility/ProbCut child evals?) and cache /
reuse; target ~1.0-1.1 evals/node = up to ~20-25% faster. Bench node counts must stay identical, then a 100 ms + 1000 ms gate.

## Not doing now
Net width changes (W128 loses at 1000 ms, W512 too slow: D255, D151); UBFM (D276); more LMR screens (flat, D269/D271).

## Gate reporting rule (user D312)
Every gate report includes 	i insight + 	i analyze output on the games.jsonl (free, hub only): score vs eval-expected, value lost per move, blunder %, knps/depth, tilt, sign agreement, turning points. Gate scripts must run both at the end; sandbox hub.tar must be rebuilt from hub HEAD (it lacked the end-of-gate report). rwphase2 pooled at 1,058 g = +6.6 +-20.9 (LOS 73%); extension continues to 2,400 g.

## Result of step 1 (D313)
rwphase1 (less result weight) = worse (-32). rwphase2 (0.4 everywhere) pooled 3,200 g = +3.7 Elo, no measurable difference (ti insight z +1.18 value lost, tilt z -0.37). 3,000-game rule reached: user decides. Sandbox deleted. Next lever: result weight 0.5-0.6 in the middlegame (same command, both train and export flags) or hard-position mining.

## Training-method sweep (D320, user: are we optimising the training methods?)
Done: result weight (0 mid -32, 0.4 +4, 0.55 +22), calibration off (+15). Running: 0.7 (rwphase5), 0.85 (rwphase6). Queue, one variable at a time vs the best recipe so far, each = Modal run (~$0.2) + 5k gate with ti insight/analyze (~$0.3): lr/lr-min/warmup; steps + early stop; proof-blend; wsum clip; from scratch vs fine-tune; data scaling 25/50/100%; EMA; per-phase result weight. Export never uses --calibrate again unless a gate says so.
