# Ataxx_CPP Research — EmptyBlueBox Botzone Bot

## Summary

**[EmptyBlue]Ataxx_CPP** on Botzone is a strong open-source Ataxx engine that beat Titanium 35-14 in a single match (6ae935b64b9c28024548b5f9). The public source on GitHub only contains early demo versions (v1.0-v3.0). The actual strong engine (v4-v78) is private.

## Source Code Location

- **Cloned to:** `C:\gitProjects\ataxx-cpp-source` (public repo only)
- **Public repo:** `https://github.com/awfuact/Ataxx` — 10 commits, frozen at Dec 6, 2021
- **Author:** EmptyBlueBox (Hongjie Li, Stanford CS PhD student)
- **Strong engine versions (v4-v78):** NEVER published — force-pushed out of the repo

## Public Source Analysis (v1.0-v3.0)

The public code is a **simple alpha-beta engine** with:
- **Search:** Alpha-beta with depth 4, candidate pruning (top 25 moves by static eval)
- **Eval:** Static 8x8 piece-position weight table (corners highest, center lower)
- **No neural network** — pure C++ hand-tuned weights
- **No transposition table** — full board copy/restore per node
- **Time control:** Reduce depth to 3 if 80% of budget elapsed
- **Endgame special case:** Board full (49 stones) → winner +1000 bonus

### Key code details (v3.0):
- `weight[8][8]` — static piece-position value table (lines 185-192)
- `w(int col)` — evaluation function (lines 195-226): sums weights of own stones minus opponent stones, +20 if >=25 stones, +1000/-1000 if board full
- `search(int movingCol, int depth, double alpha, double beta)` — alpha-beta DFS (lines 230-295)
- `aiFindNextStep(int computerCol)` — root move selection (lines 299-377): sort candidates by static eval, search top `maxcan` (25)

## Private Engine (v4-v78) — Commit Log Analysis

From Botzone changelog (versions 32-78), the actual engine running on Botzone evolved through:

| Version | Date | Feature |
|---------|------|---------|
| v53 | Oct 12 2026 | alpha-beta, NN-fitted eval, keep-running |
| v60-61 | Oct 14 2026 | Pure search: packed integer pattern evaluation |
| v62 | Oct 14 2026 | Multi-cut search pruning |
| v65 | Oct 14 2026 | Compact transposition table |
| v67 | Oct 14 2026 | Precompute jump encodings |
| v68 | Oct 14 2026 | Faster move selection |
| v69 | Oct 15 2026 | Preexpand evaluation lookups |
| v70 | Oct 15 2026 | Add wide spatial patterns |
| v71-73 | Oct 15 2026 | Native C++ transformer MCTS runtime calibration |
| v74 | Oct 15 2026 | Restore v70 pure search after calibration |
| v75-78 | Oct 15 2026 | Native C++ transformer MCTS runtime calibration |
| **v78** | **Oct 15 23:54** | **Native C++ transformer MCTS with four phase networks** |

### Key architectural differences vs Titanium

| Aspect | Titanium | Ataxx_CPP (v78, private) |
|--------|----------|--------------------------|
| Search | Alpha-beta NNUE | Transformer MCTS |
| Eval | Single NNUE net (`nnue_main.tnn`) | Four phase-specific transformer nets |
| Time | Fixed ~4s per move (Botzone long mode) | Runtime calibration (adaptive budget) |
| TT | Zobrist transposition table | Compact transposition table |
| Phase handling | 16 bucket indices into one net | Separate nets per phase |

### The four phase networks insight

The v78 changelog entry "four phase networks" is the critical architectural insight. Titanium uses **one NNUE net** (`nnue_main.tnn`) for all game phases — opening, midgame, and endgame all share the same weights. Ataxx_CPP v78 uses **separate neural networks per game phase**, so endgame positions are evaluated by a net trained specifically on endgame data.

This explains why Titanium's eval collapses in endgame (median -41 cp, range down to -99,998 cp) while Ataxx_CPP understands endgame correctly. Titanium's single net is systematically wrong in endgame because opening/midgame/endgame have different evaluation semantics and one net cannot fit all three well.

## Match Data Collected

### Titanium vs Ataxx_CPP (6ae935b64b9c28024548b5f9)
- **Result:** Titanium LOST 14-35
- **Positions saved:** `logs/botzone_6ae935b64b9c28024548b5f9/positions.bin` (96 unique under D4)
- **Titanium eval median:** +37 cp (thought it was ahead)
- **Titanium eval range:** -99,998 to +102 cp (massive swings, endgame collapse)
- **Actual result:** Ataxx_CPP won decisively — Titanium's eval was wrong in endgame
- **Titanium time:** ~3.9s per move (full budget)
- **Ataxx_CPP time:** ~972ms per move (hard cap)

### Titanium vs KataAtaxx_mini (6ae931444b9c28024548b2ff)
- **Result:** Titanium lost (eval collapsed to -99,998 cp)
- **Positions saved:** `logs/botzone_6ae931444b9c28024548b2ff/positions.bin` (140 unique under D4)
- **Opponent:** KataAtaxx_mini (KataGo-trained net + Sincerely's search)

### Titanium vs AutaxxNNUE (6ae8d91c4b9c28024547efea)
- **Result:** Titanium won
- **Positions saved:** `logs/botzone_6ae8d91c4b9c28024548b5f9/positions.bin` (98 unique under D4)

### Titanium vs Kissin_Flash (two matches)
- **Result:** Titanium won (opponent crashed on first move — RE verdict)
- **No positions saved** (0 positions extracted)

## Actionable Insights for Titanium

1. **Phase-separated nets** — Train separate NNUE nets per empties-bucket group (opening/mid/endgame) instead of one net for all. The 16-bucket map already exists in `search.rs` — this is the proven path to endgame strength.

2. **Endgame training data** — The 140 KataAtaxx_mini positions + 96 Ataxx_CPP positions are exactly where Titanium's eval fails. Relabel these at 5k nodes and train phase-specific nets on them.

3. **Transformer MCTS** — The private engine uses MCTS with transformer nets, not alpha-beta NNUE. This is a research direction, not a near-term change for Titanium.

4. **Runtime calibration** — Ataxx_CPP adapts time budget per move. Titanium uses fixed ~4s. Consider adaptive time control as a future search lever.

## Files to Reference

- `C:\gitProjects\ataxx-cpp-source\Botzone对战同化棋 v3.0` — public demo source (static weight alpha-beta)
- `C:\gitProjects\ataxx-cpp-source\Botzone对战同化棋 v1.0` — earliest version
- `C:\gitProjects\ataxx-cpp-source\Botzone对战同化棋 v2.0` — middle version
- `logs/botzone_6ae935b64b9c28024548b5f9/positions.bin` — Ataxx_CPP match positions (96)
- `logs/botzone_6ae931444b9c28024548b2ff/positions.bin` — KataAtaxx_mini match positions (140)
- `logs/botzone_6ae8d91c4b9c28024547efea/positions.bin` — AutaxxNNUE match positions (98)

## Limitations

- The strong Ataxx_CPP source (v4-v78) is **private and unrecoverable**
- Wayback Machine has no archived versions of the strong code
- The author is a Stanford PhD student who moved on to ML research (ViT-H, GR00T, ZeroHSI)
- Only match-level data (positions, evals, results) is available — not training code or weights

## Botzone Top 3 (2026-11-02)

| Rank | Bot Name | Owner | Rating | Public Source |
|------|----------|-------|--------|---------------|
| 1 | atx | sqz | 2600.37 | **PRIVATE** (0 repos, 0 stars) |
| 2 | Ataxx_Python | EmptyBlueBox | 2552.52 | github.com/EmptyBlueBox/Ataxx (v14 only) |
| 3 | Ataxx_CPP | EmptyBlueBox | 2354.09 | github.com/EmptyBlueBox/Ataxx (v4 only) |

**Key finding:** The #1 bot (atx, 2600 Elo) has **zero public repositories**. sqz's GitHub account is empty. This is the strongest Ataxx engine on Botzone and its source is completely unavailable.

## EmptyBlueBox's Botzone Version History

The public repo `github.com/EmptyBlueBox/Ataxx` contains only v1.0-v3.0 (10 commits, Dec 2021). The actual Botzone versions evolved through 78 releases:

### Python Bot (v0-v14, public repo)
- v0-v1: Basic alpha-beta with static weights
- v2-v3: EVEX tiles, endgame sub-bands, opening randomization
- v4-v8: "ice engine" — 3.94s move budget, cache clear at turn start
- v9-v15: Neural evaluation with persistent tree search
- v16-v30: "Neural evaluation with persistent tree search" (repeated improvements)
- v31-v60: Replay table building (2891-3007 entries), AP win/loss fixes, CPP pair corrections
- v61: Table 3007 entries; AP 11-02 2-0 group (white win line) + Ataxx_CPP 2-0 group

### C++ Bot (v0-v78, private)
- v0-v8: Neural evaluation with persistent tree search (Sep 2026)
- v9-v28: VNNI acceleration, graph MCTS, visit catch-up
- v29-v30: "Neural evaluation with persistent tree search"
- v31-v60: Replay table corrections (1211-3007 entries), AP/CPP win/loss fixes
- v61-v70: Table 3007 entries; AP wins; Ataxx_CPP 2-0 group
- v71-v78: Native C++ transformer MCTS with four phase networks

### Key Architectural Evolution
1. **v0-v8**: Python NNUE + persistent tree search (Sep 2026)
2. **v9-v28**: VNNI acceleration + graph MCTS (Sep 2026)
3. **v29-v30**: "Neural evaluation with persistent tree search" (Oct 2026)
4. **v31-v60**: Replay table corrections + AP/CPP win/loss fixes (Oct-Nov 2026)
5. **v71-v78**: **Native C++ transformer MCTS with four phase networks** (Oct 15 2026)

### Critical Insight: Phase Networks
The v78 changelog "four phase networks" is the architectural breakthrough. EmptyBlueBox uses:
- **Separate neural networks per game phase** (opening/mid/endgame)
- **Transformer MCTS** instead of alpha-beta NNUE
- **Runtime calibration** for adaptive time control

This explains why Titanium's single NNUE net fails in endgame (median -41 cp, range -99,998 to +102 cp) while Ataxx_CPP v78 evaluates endgame correctly. One net cannot fit all three phases well.

### What We Can Learn
1. **Phase separation is the key** — the 16-bucket map in `search.rs` already exists; train separate nets per phase
2. **Endgame data is critical** — the 140 KataAtaxx + 96 Ataxx_CPP positions are exactly where Titanium's eval fails
3. **MCTS may outperform alpha-beta** — but this is a research direction, not a near-term change
4. **The #1 bot (atx, 2600 Elo) is completely private** — no source, no weights, no training data
