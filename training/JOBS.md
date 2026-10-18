# JOBS — live background work tracker (update on launch/finish/kill)

## RULES
- Max ONE training + label workers at a time (16GB box swap-thrashes beyond that — S28 lesson).
- Long jobs go via SCHTASKS (survives shell death — Start-Process children get reaped on abort).
- Every job: append-only log + DONE marker + LEDGER row on verdict. No foreground waits.
- EVERY game goes in the DB: `python training/results.py backfill` after every gate/match run (parses match + book logs, deduped by path). No exceptions.

## LIVE (2026-10-05 ~15:15)
| job | what | started | log | out | state |
|---|---|---|---|---|---|
| s49b x8 | 12M graft labels @5k (lab12m 24 chunks) | 12:45 | logs/s49_label12m.txt | data/nnue/lab12m/scores_*.txt | KILLED 2026-10-17 (8 workers @100% CPU, no recorded approval; rows cached, resumable via --skip) |
| s52 | fused-trainer validation, 400k graft, lr .05 | schtasks titan_s52 | logs/s52_train.log | data/nnue/tupS52fuse.tup | RUNNING (ep40, val 8768) |
| s46b-seed42 | seed sweep (8,9 done: 8649/8456) | schtasks?/s46b.cmd | logs/s46_s42.log | data/nnue/tupS46s42.tup | RUNNING (seed21 killed stuck) |
| r0conf5k | CONFIDENCE GATE (user-authorized 2026-10-17): R0 vs S37fix @5k nodes, 200 book games (100 pos x both seats), 6 workers, same release exe both sides | schtasks titan-r0conf5k | logs/s53r0_conf5k_book.txt | tally in log | DONE: chunk1 110-90, chunk2 (fresh book100b, 0 overlap) 118-82; pooled @5k 335-264-1/600 = 0.559, 95% CI [0.519,0.599] CONFIRMED stronger per-node. LEDGER S53r0conf. |
## QUEUED NEXT (in order, one at a time after s52)
1. Gate s52fuse vs S37fix @5k 100g (fused validation verdict).
2. Seeds 21 (redo) + gate all seeds vs S37fix; keep best.
3. Full PBT program (4 members x 3 gens) on 400k.
4. NMP-R2 3rd leg (pooled 108-84).
5. Merge winner to main (time legs per protocol).

## DEAD / PARKED
- s46-seed21 (killed stuck, no ckpt — redo).
- s51/s51b (superseded by s52 fused validation).
- ow10/ow20 postfix: gated (45-55, 44-53) REJECTED.
| regretloop | AUTONOMOUS REGRET LOOP (user-authorized 2026-10-17, ~2h): champ vs Moonbird 40g/4w -> mine losses -> graft-label @500k x4 -> warm retrain (regret upweight + 10k base replay) -> gate 100g, adopt iff better | schtasks titan-regretloop | logs/regret_loop_status.txt | RUNNING round 1 (infra patched: book_gate --moves-out, miner ## startpos override) |
| night | OVERNIGHT (user-authorized, unattended): P1 corr-build+gate (auto-merge to main ONLY on SPRT pass, LEDGER row), P2 regret v2 2M-labels books C/D to 06:30, P3 backfill+closeout | schtasks titan-overnight 22:45 | logs/overnight_status.txt | QUEUED |
