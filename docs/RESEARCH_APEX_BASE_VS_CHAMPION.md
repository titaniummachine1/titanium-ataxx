# Apex (Quoridor) finding: retrain from a fixed base, not from the champion. Does it carry over to Ataxx? (2026-11-02)

Source: text the user pasted in chat ("Why apex trains from a base model, not from the current champion"; a Quoridor bot, related files salvaged in docs/external/ace_apex/).
Claim (Quoridor, self-play): continuing the champion made apex ~50 Elo weaker; every champion since was trained from the SAME fixed base on all data at once.
Reasons given: (1) a strong net is over-confident, teaching a new move loosens it everywhere; (2) copies of copies carry mistakes forward; (3) ~1/3 of neurons go permanently silent and gentle fine-tuning never wakes them; (4) the knowledge is in the data, not the weights. Random init was also worse.

## Assessment against Titanium's own record (what is checked, what is not)
- Agrees: never random init (LAWS C14, Mistake 39: a fresh downscale fine-tuned at lr 2e-4 went 0-413-3). Gains this week came from changing the TARGET on the same net and data (+23, +28 Elo), i.e. the knowledge is in the data/target.
- Already how we train: the Modal recipe (docs/RUNBOOK_OVERNIGHT_2026_11_02.md line ~26) starts every blend run from /vol/runs/sat16a/best.tnn (F5FA8557 family), NOT from the latest winner. Believed true for all rwphase nets; only the runbook recipe was checked, not each run's provenance.
- Does not carry over directly: Apex is self-play (its data depends on the net); ours is supervised on 5k-node search labels plus game results, independent of the net being trained, so the copies-of-copies mechanism is much weaker. An earlier continuation (D94, 2026-10-26) WON +132 Elo, but that net was undertrained.
- NOT measured: the dead-neuron fraction of nnue_main.tnn (46FCB5A4); the width ablations (W128/256/512 reach the same loss floor, D232-D234) hint that capacity is not the bottleneck but are not a dead-neuron count.

## Proposed tests (cheap, not started)
T1 retrain with the identical rwphase9 recipe and data but --init = the current champion 46FCB5A4 instead of sat16a: Modal ~$0.40, gate 5k in Docker vs main. If it loses by tens of Elo the Quoridor finding holds for us: write "always restart from sat16a" into the runbook. If it wins, chain improvements.
T2 count inactive L1/L2 neurons of 46FCB5A4 over a position sample (hub command, Rust, LAWS A7; no Python).
