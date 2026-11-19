# pat3_screen (analysis only, NOT the hub, NOT a gate)

Scratch Rust screen: does a 25-window 3x3 net (Jataxx-style: 3^9 states per window, 128->32 hidden here, 3 piece-count output buckets)
add game-outcome information on top of the champion-labelled 5k search score (cp)?
Data: `data/nnue/packed/train_v9.tdat` (28.99M rows) with `val_v9.tdat` (602k rows); outcome = game result, side-to-move relative.
Source copy of C:/gitProjects/titanium-rl3x3 (`crates/titanium-cli/src/rl.rs`, `crates/titanium/src/net3.rs`), GNU build, target-cpu x86-64-v3.
Run: `titanium-cli rltrain --tdat <train_v9.tdat> --val-tdat <val_v9.tdat> --y outcome [--base none] --hidden 32 --epochs 3 --lr 0.003 --out <net.n3>`
Model: P(win) = sigmoid((cp + 100*net)/S), S calibrated on cp alone; Adagrad; no weight decay.
Logs: `logs/pat3_screen/R1..R3.log`. Results: `docs/research/hce-patterns-speed/RESEARCH_JATAXX.md` addendum. The proper home is a hub port (LAWS A7).
