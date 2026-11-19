# SEE as an eval term, measured (2026-11-14, branch `exp/max-nps`)

Prototype: `crates/titanium/src/see_brute.rs` (`see-brute` CLI command).
Question answered: does SEE improve an eval that otherwise only counts pieces,
and what does it cost in the max-speed brute kernels?

## Definition used (bit-identity is the contract)

    V = max over my moves of [ min over the opponent's replies of M ]
    M = my stones - opponent stones
    terminal = Botzone rule (either side stuck -> mobile side absorbs every empty)

`swing_2ply` is the fast implementation (engine movegen, one child board per
move); `swing_naive` is a plain recursive 2-ply negamax written independently.
The unit test `exact_2ply_matches_naive_recursive_search` requires them to be
identical on 400 random positions, terminals included. That is the meaning of
"SEE equals a 2-ply deep search with material as eval".

## Quality (200 real mid-game positions, judge = Titanium depth 8)

| eval | agreement with the judge |
|---|---|
| material only (kernels) | 73/200 = 36.5 % |
| material + SEE, w=100 | 74/200 = 37.0 % |
| material + SEE, w=200 | 37/200 = 18.5 % |
| material + SEE, w=400 | 25/200 = 12.5 % |

At its only neutral weight the gain is +0.5 points, i.e. noise. Over-weighting
is actively harmful: 200 cp per stone swamps material.

## Cost (4 positions, depth 4, one thread)

| eval | nodes | wall | rate |
|---|---|---|---|
| material only | 625 329 | 0.031 s | **20.2 Mnps** |
| material + exact SEE | 736 905 | 66.0 s | **11.2 knps** |

**~1800x slower per node.** Cause: every leaf evaluation performs its own
2-ply search, i.e. ~40 of my moves x ~60 opponent replies of real movegen.

Depth reachable in 100 ms: material ~4.6 plies; exact SEE < 1 ply (hundreds of
nodes only). The earlier "0.5 ply" figure came from an approximate proxy.

## The bit-plane shortcut is NOT exact (rejected, with the number)

A no-movegen version using `victim_planes` + `planes_max` agreed with the exact
value on only **84.8 %** of 400 random positions. Two structural reasons:

1. A jump vacates its origin, so the post-move stone SET depends on the origin;
   the opponent's reply value depends on that set, not just on the stone count.
   Every origin is a distinct case.
2. After the opponent's reply the game can END (Botzone absorption), which is not
   a material value at all and needs a mobility check per reply.

`plane_max` can give the best conversion count cheaply, but neither of the above
is recoverable from it, so it cannot be bit-identical by construction.

## What this means

- SEE as a **leaf eval** is not viable at any weight: 1800x cost for no measured
  quality gain.
- The exact value is still useful where it is paid once instead of per node:
  root ordering, pruning decisions, or offline labelling (the endgame solver in
  `exp/endgame-solver` uses the same 2-ply/terminal semantics).
- If a cheap SEE is wanted, it must be a **table** indexed by (phase, my
  conversions, opponent's best reply) like the TPOL5 head, trained once — not a
  per-leaf computation.
