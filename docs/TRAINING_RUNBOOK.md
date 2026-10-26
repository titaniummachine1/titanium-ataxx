# Training runbook (2026-10-26) - how e11827d1 was made, and how to repeat it

Data: data/nnue/final/final_{train,val,test}_v4.npz (titanium_cp = 2026-10-25 relabel, see docs/RELABEL_RUNBOOK.md) + features
data/nnue/nnfeat/{split}.cnt/.idx (engine `titanium-cli nndump`, rows in npz order). Trainer: training/train_nnue.py (reads v4 by default).

## GPU (Modal) - the fast path
One-time upload (Git Bash: `export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8`, the volume must exist: `python -m modal volume create ti-train`):

    python -m modal volume put --force ti-train data/nnue/modal_up/final_train_v4.npz /up/final_train_v4.npz
    python -m modal volume put --force ti-train data/nnue/modal_up/final_val_v4.npz   /up/final_val_v4.npz
    python -m modal volume put --force ti-train <seed .pt>                              /up/<seed>.pt

(the slim npz = only the columns Split reads; rebuild with the snippet in DECISIONS D80 if the labels change). Each container rebuilds the
12 GB feature index with the engine's nndump and checks sha256 against the local files (SHA dict in training/modal_train.py: UPDATE it if
the features change, or the run refuses to train).

Stage 1 to saturation (detached; ~2.7 min/epoch on an L4, ~$1.2/h, each call capped at 4 h - re-run the same command to resume):

    python -m modal run --detach training/modal_train.py --tag <tag> --args "--stage base --init /vol/up/<seed>.pt --epochs 60 --auto-lr --auto-restarts 2 --min-rel 0.002 --metric mse_target --lr 1.1e-3 --lr-min 2e-5 --hypergrad --bs 4096 --width 256 --opt ranger --wd 1.4e-6 --la-k 10 --la-alpha 0.32 --ema 0.9995 --block 1024 --seed 7 --threads 4 --min-free-gb 1.5"

(e11827d1 = first `--lr 4e-3 --la-k 8 --la-alpha 0.56 --wd 3e-7` from c4829350 to saturation, then the settings above from the best search
trial; for a fresh round start straight with the settings above.) Continuation hyperparameter search from a saturated net:

    python -m modal run --detach training/modal_train.py --tag hpo_x --script optuna_ft.py --args "--init /vol/ckpt/<tag>/net.pt --out /vol/ckpt/hpo_x --trials 16 --epochs 2 --threads 4"

Results: `python -m modal volume get --force ti-train /ckpt/<tag>/net.pt data/nnue/modal_ckpt/<tag>/net.pt` (net.pt = best of live / EMA /
SWA / plane point; the log is /ckpt/<tag>/log.txt). Stop a run: `python -m modal app stop -y <app id>`.

## Judge a net (games only - val loss is a guide, D92 showed it tracks Elo here)
    python training/gate_net_vs_main.py --pt data/nnue/modal_ckpt/<tag>/net.pt --tag <gate tag>                  # vs main's net
    python training/gate_net_vs_main.py --pt <a.pt> --tag <t> --champ <other.tnn>                                 # vs any net
    python training/gate_net_vs_main.py --tag <t> --cand-exe <exe> --tnn <net.tnn>                                # engine change, same net
416 games at 5k nodes, pure node limit (--time 0), paired book both seats, tripwire = shard 0 replayed bit-identically (else VOID).
ONE gate at a time. Never touch main-ataxx while a gate runs (the champion side reads its exe and nnue_main.tnn live).

## Do NOT
- fine-tune on the Kata teacher columns (teacher_p3 / kata_td): -137 Elo (D98); they are TD values with the game's own result in them (D101);
- use Ataxx_CPP or KataAtaxx as teachers: Titanium beats both on equal terms (D105);
- trust a gate whose tripwire says MISMATCH, or run gates while the box is overloaded (D102).

## Promote a net to main (user approval naming the net)
Copy the exported .tnn to data/nnue/nnue_main.tnn in a SEPARATE worktree from main (git add -f), commit, check `match` default games equal an
explicit `--nnue` run, fast-forward main when no job reads main-ataxx, rebuild, `python tools/build_botzone.py` for the Botzone upload.

## Futility small net (D109/D110) - RETRAIN IT WITH EVERY NEW BIG NET
The engine compiles in data/nnue/fut_small.tsn (52,844 B): a tiny net trained as a LOWER BOUND of the big net's eval of
futility children (tau 2%). It is fitted to ONE big net (e11827d1); a new nnue_main.tnn without a new small net = the
filter prunes on the wrong bound. Steps (CPU only: PC or a Docker sandbox, ~1 h total):

    # 1. record child positions + the big net's evals (probe branch exp/fut-probe, worktree C:/gitProjects/futprobe-ataxx;
    #    rebuild it with the NEW net in data/nnue/nnue_main.tnn); roots = 49-char board lines; ~5 min per 4 x 1000 roots
    FUT_PROBE=data/nnue/smallnet/rec_trK.bin titanium-cli label --nodes 5000 --tuple none --nnue <net> < roots_trK.txt
    FUT_PROBE=data/nnue/smallnet/rec_ev.bin  titanium-cli label --nodes 5000 --tuple none --nnue <net> < roots_ev.txt   # disjoint roots
    # 2. pack (dedupe train, keep every eval call)
    python training/smallnet_pack.py
    # 3. eval-mode net, then bound heads warm-started from it (both saturate by val-driven LR halving)
    python training/smallnet_train.py --out data/nnue/smallnet/sn_w32.pt --width 32
    python training/smallnet_train.py --out data/nnue/smallnet/sn_q32.pt --width 32 --quantiles 0.002,0.005,0.01,0.02,0.05 --init data/nnue/smallnet/sn_w32.pt --lr 1e-3
    # 4. score as a filter on the held-out calls (skipped big-net evals vs wrong prunes; baseline = hand filter)
    python training/smallnet_eval.py data/nnue/smallnet/sn_q32.pt
    # 5. export one head + parity reference, copy into the engine, rebuild, check parity and speed
    python training/smallnet_export.py --pt data/nnue/smallnet/sn_q32.pt --tau 0.02 --out data/nnue/fut_small.tsn
    titanium-cli snparity --ref data/nnue/fut_small.tsn.ref.txt      # max |rust - torch| <= 0.5 cp, ~180 ns/eval
    # 6. gate 100 ms + 50 ms vs main (training/time_gate.py); git add -f data/nnue/fut_small.tsn (repo_hygiene EXEMPT)

`--no-fut-sn` on any titanium-cli command (serve included) switches the filter off = exact full-strength search (D112).
Docker: python:3.12 image, `pip install numpy torch --index-url https://download.pytorch.org/whl/cpu`; uploading the 1.5 GB
sn_train.npz took ~14 min, so generating the data inside a rust:1.97 sandbox may be faster next time.
