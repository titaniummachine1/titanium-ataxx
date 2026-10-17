# JOBS — live background work tracker (update on launch/finish/kill)

## RULES
- Max ONE training + label workers at a time (16GB box swap-thrashes beyond that — S28 lesson).
- Long jobs go via SCHTASKS (survives shell death — Start-Process children get reaped on abort).
- Every job: append-only log + DONE marker + LEDGER row on verdict. No foreground waits.

## LIVE (2026-10-05 ~15:15)
| job | what | started | log | out | state |
|---|---|---|---|---|---|
| s49b x8 | 12M graft labels @5k (lab12m 24 chunks) | 12:45 | logs/s49_label12m.txt | data/nnue/lab12m/scores_*.txt | RUNNING (~1M/12.4M) |
| s52 | fused-trainer validation, 400k graft, lr .05 | schtasks titan_s52 | logs/s52_train.log | data/nnue/tupS52fuse.tup | RUNNING (ep40, val 8768) |
| s46b-seed42 | seed sweep (8,9 done: 8649/8456) | schtasks?/s46b.cmd | logs/s46_s42.log | data/nnue/tupS46s42.tup | RUNNING (seed21 killed stuck) |

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
