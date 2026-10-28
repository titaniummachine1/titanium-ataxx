# ⚡ Persistent vs. One-Shot on Botzone — Verified Plan (Track A)

Reviewed against the code on 2026-10-28. This replaces the earlier draft, which had wrong and overstated claims (listed in §1).

---

## 1. Corrections to the earlier draft

| Earlier claim | What the code says | Verdict |
| :--- | :--- | :--- |
| "Persistent mode loses TT / ecache between moves" | In `TI_MODE=long` the driver keeps the `TiEngine` alive; `ti_search` reuses `e.searcher`, so **TT, ecache, killers and history already carry over**. `ecache` is only cleared on new game / eval change (`search.rs:673`). | ❌ Wrong. Persistence already exists. What's missing is **TT hygiene**. |
| "Our SPRT gates measure one-shot" | `titaniumtest` sends `uainewgame` **once per game** (`main.rs:412`) and FEN + `go` each move, so the `Searcher` stays warm inside a game. | ❌ The reverse is true: **every gate so far measured persistent play**. Botzone one-shot (cold every move) has **never been gated**. |
| "History `/2` per turn wipes 94% of ordering" | Fresh cutoffs refill history in every search (`hist_add`, `search.rs:1124`). | ⚠️ Overstated. It's a tuning knob, not a bug. |
| "Long mode = 4 s per turn" | That's the driver's assumption (`titanium_botzone.py:36`). Botzone's docs only say the per-turn limit still applies; the process is **`SIGSTOP`ped between turns**. | ⚠️ **Not verified.** If long mode really gets 6 s, persistent wins outright and there's nothing to trade off. |
| "Pondering / thinking on opponent time" (implied by "persistent tree") | `SIGSTOP` between turns. | ❌ Impossible on Botzone. Don't build it. |
| TT move `0` could clash with a real move | `to_u32 = from<<8 | to`; `(0,0)` isn't a legal move. | ✅ No clash. |
| `TtEntry` has 6 bytes of padding | 8+4+4+1+1 = 18, aligned to 24. | ✅ Correct. Adding `age: u8` costs no RAM. |

---

## 2. Verified items, in order

### A0. Measure before building (cheap, do first)
1. **Find the real long-mode time limit.** Upload the long-mode driver as a *test version* and play one unranked or bot-vs-bot game. The debug line already logs the elapsed `t…`. The whole 4 s vs 6 s question depends on this number.
2. **Add `--reset-every-move` to `titaniumtest sprt`** (send `uainewgame` before each `search_move` for one side). That's the only honest way to simulate Botzone one-shot locally. About 10 lines of code.
3. **Baseline gate, no engine changes:** `warm @ 3800 ms` vs `cold-every-move @ 5700 ms`. This tells us how much persistence is worth *today*, before any fixes.

### A1. TT generation aging (real issue, but the rule must protect reusable entries)
- **Problem:** depth-preferred replacement (`search.rs:452`) never evicts old deep entries. Each clone adds a stone and stone count never goes down, so earlier-turn positions become unreachable, yet they keep blocking their slots for the rest of the game.
- **Fix:** put `age: u8` in the padding; `tt.age += 1` once per `search_gen`.
- ⚠️ **Correction:** *don't* use "replace anything with an old age". That would also evict last turn's **still-reachable** predicted-reply subtree, which is the whole point of persistence. Use Stockfish-style scoring instead: replace when `new_depth >= old_depth - 4 * age_diff` (or the slot is empty / same key). Also refresh `age` on a TT-cutoff hit so live subtree entries stay young.

### A2. Keep the TT move on fail-low (verified, standard Stockfish behavior)
- `negamax` updates `best_mv` whenever `score > best`, including when every move fails low (`search.rs:1093`). On a fail-low, those scores are just upper bounds, so the "best move" is noise, and `Tt::store` writes it over a real refutation move (same-key path).
- **Fix:** in `finish_node`, when `best <= alpha_orig`, pass "no move" and have `store` keep the existing `e.mv` for the same key.
- This also helps one-shot mode (it applies within a single search).

### A3. History decay `/2` → `×3/4` per search: **tuning, low priority**
Only worth trying as a separate SPRT after A1 and A2.

### A4. Killer 2-ply shift: **dropped**
The killers at ply 2 of the old tree came from many different sibling subtrees, so they say little about the actual new root. The engine also can't tell how many plies have passed through UAI FEN. Low value, adds complexity.

### A5. Countermove table: **optional, later**
Standard technique, but ordering in Ataxx is already dominated by captures and the TT move. Gate it only after A1 and A2 land.

---

## 3. Gating protocol
1. A1+A2 vs main at equal time (100 ms, warm, `book100`): must not lose (`elo0=0, elo1=5`).
2. Re-run the A0.3 asymmetric gate with A1+A2: `warm @ 3800` vs `cold-every-move @ 5700`.
3. Switch Botzone to `TI_MODE=long` only if (a) the real limit from A0.1 is known, (b) step 2 passes, and (c) there are zero timeouts in the Docker harness (`tools/build_botzone.py`).

> [!NOTE]
> Local gates use standard rules + `book100`; Botzone uses its own judge rules (4-corner start, no passes, repetition ban). Persistence effects shouldn't depend on the rules, but the final Botzone sign-off stays a real-match check.
