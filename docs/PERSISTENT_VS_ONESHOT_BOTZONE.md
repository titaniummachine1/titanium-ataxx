# ⚡ Persistent vs. One-Shot Mode on Botzone: State-Preservation Optimizations

A code-audited specification for **Track A (Immediate Execution while Relabel Runs)**: fixing the 4 persistence bottlenecks in [`crates/titanium/src/search.rs`](../crates/titanium/src/search.rs) and gating **4.0s Persistent Mode** (`>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`) against **6.0s Cold One-Shot Mode**.

---

## 1. Competitive Evidence: Why Botzone #1 (`atx`) Uses Persistent Mode

From our Botzone ranklist scrape (`ranklist_ataxx.html`), #1 ranked bot **`atx`** (`2562.93` Elo by `sqz`) states as its literal description:
> **`"Neural evaluation with persistent tree search."`**

`atx` deliberately gives up the extra 2 seconds of One-Shot mode (`6.0s` $\to$ `4.0s`) because **a warm Transposition Table, warm Move-Ordering History, warm Killers, and warm Eval Cache across moves save 1.5–2.5 seconds of shallow re-search on every turn** and provide deeper subtree accuracy than a cold 6.0s restart.

---

## 2. Four Concrete Persistence Bottlenecks Found in `search.rs`

Auditing [`crates/titanium/src/search.rs`](../crates/titanium/src/search.rs) revealed why persistent mode was previously handicapped:

### Bottleneck A1: TT "Zombie Clogging" Across Moves (`search.rs:326` & `452`)
* **Current Code ([`search.rs:326-332`](../crates/titanium/src/search.rs) & [`448-461`](../crates/titanium/src/search.rs)):**
  ```rust
  struct TtEntry {
      key: u64,   // 8B
      mv: u32,    // 4B
      score: i32, // 4B
      depth: u8,  // 1B
      flag: u8,   // 1B (+ 6 bytes of unused struct padding = 24 bytes!)
  }
  // In Tt::store:
  if e.flag == FLAG_EMPTY || e.depth <= depth || e.key == key { *e = ... }
  ```
* **Why This Hurts Persistent Games:**
  In Ataxx, every clone move permanently increases total stone count on the board. Therefore, positions from Turn 1–5 are **mathematically unreachable** by Turn 10. However, because `Tt::store` has **no `age` (`generation`) check**, a `depth = 11` entry stored on Turn 1 **permanently blocks** Turn 25 from storing any `depth = 1..10` entry in that hash slot! Over a 60-turn game, the TT becomes progressively clogged with unreachable early-game zombie entries.
* **Zero-RAM Fix:**
  Add `age: u8` inside the existing 6-byte struct padding of `TtEntry` (`size_of::<TtEntry>()` remains exactly `24` bytes). Increment `self.tt.step_age()` at the start of `search_gen`. Replace whenever `e.flag == FLAG_EMPTY || e.key == key || e.age != self.age || e.depth <= depth`.

### Bottleneck A2: Fail-Low (`FLAG_UPPER`) Overwrites Existing TT Best Move (`search.rs:452` & `1175`)
* **Current Code ([`search.rs:960`](../crates/titanium/src/search.rs), [`1095`](../crates/titanium/src/search.rs), [`1175`](../crates/titanium/src/search.rs)):**
  When a PVS zero-window scout fails low (`best <= alpha_orig`, `FLAG_UPPER`), `best_mv` is simply the first move searched that failed to beat `alpha` (or `0xFFFF`). Yet `Tt::store` unconditionally overwrites `e.mv` when `e.key == key`!
* **Fix:**
  In `Tt::store`, when `e.key == key && (flag == FLAG_UPPER || mv == 0xFFFF || mv == 0) && e.mv != 0 && e.mv != 0xFFFF`, preserve the existing `e.mv` refutation move.

### Bottleneck A3: Unshifted `killers` & Over-Aggressive History Decay (`search.rs:1229-1236`)
* **Current Code ([`search.rs:1229-1236`](../crates/titanium/src/search.rs)):**
  - `self.killers` is indexed by search `ply` (`0..MAX_PLY`), and is **never shifted** when the game advances by 2 plies (our move + opponent reply). Thus `killers[0]` at Turn $T+1$ holds stale root killers from Turn $T$ instead of `killers[2]`!
  - `self.history` is divided by 2 (`*h /= 2`) every turn—erasing $1 - (1/2)^4 = 93.75\%$ of learned move-ordering signal after just 4 turns!
* **Fix:**
  - Shift killers by 2 plies at the start of each consecutive turn (`self.killers.copy_within(2..MAX_PLY, 0); self.killers[MAX_PLY - 2..] = [(0, 0); 2];`).
  - Soften per-turn history decay from `*h /= 2` ($0.50\times$) to `*h = (*h * 3) / 4` ($0.75\times$).

### Bottleneck A4: 1-Ply Countermove Table (`countermoves[stm][prev_from][prev_to]`)
* Store the quiet/jump move `m` that most recently caused a beta cutoff in response to the opponent's move `(prev_from, prev_to)` in `stack`, giving a countermove ordering bonus in `score_into`. Across a persistent game, recurring opponent jump maneuvers are refuted on the very first quiet move tried.

---

## 3. Execution & Gating Order (Immediate Track A)

1. **Step A1–A4 Implementation in `crates/titanium/src/search.rs`:**
   - Implement `Tt::age` + fail-low `e.mv` preservation + 2-ply killer shift + $0.75\times$ history decay.
   - Gate at equal time (`100ms/move`, persistent `Searcher` across moves of each game) via `titaniumtest sprt`.
2. **Step A5 Asymmetric Botzone Gate (`3.8s Persistent` vs `5.7s Cold One-Shot`):**
   - Run 200-game paired opening match where Candidate keeps `Searcher` warm at `3800 ms/move` and Baseline calls `reset_for_new_game()` before each move at `5700 ms/move`.
   - Upon passing, set `MODE = "long"` (`>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`) in [`botzone/titanium_botzone.py`](../botzone/titanium_botzone.py).
