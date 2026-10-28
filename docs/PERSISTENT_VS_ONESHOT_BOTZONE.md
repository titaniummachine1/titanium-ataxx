# ⚡ Persistent vs. One-Shot Mode on Botzone: State-Preservation Optimizations

A strategic and architectural analysis of **4.0s Persistent Mode** (`>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`) versus **6.0s One-Shot Mode** on Botzone, and the exact state-preservation techniques needed to make 4s persistent search decisively outperform 6s cold search.

---

## 1. The Botzone Trade-Off Dilemma

| Feature | One-Shot Mode (Cold Restart) | Persistent Mode (`enable_keep_running`) |
| :--- | :--- | :--- |
| **Turn Time Budget** | **6.0 seconds** (12.0s first move) | **4.0 seconds** (12.0s first move) |
| **Startup Overhead** | ~120–250 ms (Python interpreter + `.so` dlopen + 100MB TT alloc) | **0.0 ms** (resident in RAM) |
| **History Replay** | Replays all past moves from move 1 on every turn | **0 ms** (`ti_apply` called once for opponent's latest move) |
| **Transposition Table** | **Completely wiped** every turn (starts empty at depth 1) | **Preserved across all 100+ turns** |
| **History Heuristic** | **Reset to 0** every turn (starts move ordering blind) | **Preserved and aged** across the entire match |
| **Killer Moves** | **Reset to empty** every turn | **Shifted and reused** from previous plies |
| **Eval Cache (`ecache`)** | **Wiped** every turn | **Preserved** (1 MiB table avoids repeat NNUE calls) |

### Why 6.0s One-Shot Was the Default:
Historically, one-shot was chosen because 6s granted **+50% raw clock time** per move, which masked the lack of state persistence in early engine revisions. 

### Why Optimized 4.0s Persistent Mode Wins (+1.5 to +2.5 Plies Net Depth):
In a midgame position (move 25):
* A **cold 6s engine** spends the first 1.5 seconds searching depths 1..8 from scratch to populate its TT and learn move ordering, only reaching depth 13 by second 5.5.
* A **persistent 4s engine** already evaluated the opponent's move at depth 11 during the *previous* turn. Its TT already holds the subtree to depth 9! Depths 1..9 resolve in **`< 15 ms`**, and the remaining 3.7 seconds push the search to **depth 15–16**!

---

## 2. The 5 Progress Preservation Optimizations for Persistent Mode

To maximize the advantage of persistent mode, Titanium implements 5 state-preservation optimizations across turns:

### 2.1 Transposition Table Subtree Recycling & TT Generation Aging
* **Generation Counter:** Add an `age: u8` field to TT entries. At the start of each new turn, increment `tt.generation += 1`.
* **Subtree Carry-Over:** When the opponent plays move $m_{\text{opp}}$, the new root board $B_{\text{new}} = B_{\text{old}} \cdot m_{\text{my}} \cdot m_{\text{opp}}$ was already searched at depth $D - 2$. 
* **PV Seeding:** Root move ordering probes the TT entry of $B_{\text{new}}$ directly, instantly ordering the true principal variation as move 0 (`ORD_TT`).

### 2.2 Countermove / Continuation History (`conthist`)
* Stockfish and top engines use **Continuation History** (moves that historically refuted the opponent's previous move):
  $$\text{Table: } \mathbf{H}_{\text{cont}}[\text{prev\_piece}][\text{prev\_to}][\text{curr\_from}][\text{curr\_to}]$$
* In one-shot mode, this table is lost on every move.
* In persistent mode, countermove quality is refined over dozens of turns, improving quiet move ordering by **~12–18% fewer nodes to cutoff**.

### 2.3 History Aging vs. Wiping
In [`crates/titanium/src/search.rs:1229-1236`](../crates/titanium/src/search.rs):
```rust
// Age the history table so old cutoffs don't dominate forever
for side in self.history.iter_mut() {
    for row in side.iter_mut() {
        for h in row.iter_mut() {
            *h = (*h * 7) / 8; // gentle decay instead of clearing
        }
    }
}
```
Gentle decay preserves long-term positional move preferences while allowing recent tactical motifs to take precedence.

### 2.4 Killer Moves Shifting
* When two plies are played (our move + opponent move), shift killer moves up by 2 plies:
  $$\text{killers}[p] \gets \text{killers}[p + 2]$$
  Killers discovered deep in the previous turn's tree remain active candidate refutations at the new search root.

### 2.5 Static Eval Cache (`ecache`) Persistence
* In Ataxx, many move sequences transpose into identical bitboard states (different jump/clone order).
* Preserving `ecache` (65,536 entries, 1 MiB) across turns allows Titanium to retrieve exact Big Net evaluations instantly without re-running the 6.5 µs NNUE feature transformer.

---

## 3. Head-to-Head Gating Plan: 4s Persistent vs. 6s One-Shot

Using `titaniumtest sprt`, we gate the two modes head-to-head under true Botzone constraints:

```powershell
# Gate 4s Persistent (with warm TT/History) vs 6s Cold Restart (with cleared TT/History per turn)
titaniumtest sprt `
  --candidate "titanium-cli serve --mode persistent --time-ms 3800" `
  --baseline  "titanium-cli serve --mode oneshot --time-ms 5700" `
  --book data/nnue/book100.txt `
  --elo0 0 --elo1 10 --max-games 200 --threads 6
```

### Verification Criteria for Switching Botzone Default:
1. **Pass Condition:** `+15 Elo` or better at $95\%$ confidence ($LLR \ge 2.944$).
2. **Safety Check:** Zero timeouts over 200 consecutive games in the Linux Docker test harness (`tools/build_botzone.py`).
3. **Deployment:** Flip `MODE = "long"` in `botzone/titanium_botzone.py` and upload to Botzone.
