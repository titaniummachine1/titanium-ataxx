# exp/nnue worktree (Titanium Ataxx NNUE engine branch)

This worktree only holds the ENGINE code of the NNUE work (branch `exp/nnue`, from main 298b7ae). Its own `AGENTS.md`
is main's older copy. The live project memory is in the main folder — read these first:

@C:/gitProjects/titanium-ataxx/AGENTS.md

- Current state + what is owed: `C:\gitProjects\titanium-ataxx\HANDOFFS.md` (TOP entry).
- NNUE design ground truth: `C:\gitProjects\titanium-ataxx\docs\NET_ANALYSIS.md` (parts 1 + 2, section 11 = v2 decisions).
- Training scripts: `C:\gitProjects\titanium-ataxx\training\` (train_nnue.py, optuna_nnue.py, pbt_nnue.py,
  export_nnue.py, gate_nnue.py). Data: `C:\gitProjects\titanium-ataxx\data\nnue\nnfeat\` (features dumped by
  `titanium-cli nndump`; regenerate after any change to `crates/titanium/src/nnfeat.rs`).

Engine files of this branch: `crates/titanium/src/nnfeat.rs` (feature groups, brute-force tests),
`crates/titanium/src/nnue.rs` (inference, TNN1 format), `search.rs` (`set_nnue` hook; old path must stay
bit-identical: `bench --depth 8` = 10915 nodes, run from `C:\gitProjects\main-ataxx`), `titanium-cli` commands
`nndump`, `nneval` (Rust vs torch parity), `nnbench`, `match --nnue/--opp-nnue` (use `--tuple none` on the NNUE side).
Commit before any gate (law 8: gates run between git refs only).
