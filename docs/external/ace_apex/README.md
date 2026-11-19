# Salvaged from Downloads (2026-11-02, before the user wiped it; SSD was full)
Source: the user's other project "ace apex" (Quoridor, AlphaZero-style MCTS + nanochat transformer, `C:\Users\alex\qmcts`).
- ACE_APEX_ARCHITECTURE.md: net, search, training loop, gating, costs. Ideas relevant to Titanium's policy/value work: "golden rows"
  (deep-search positions with >= 10k visits, written 20x and oversampled 3-10% of draws; lightly searched deep rows HURT as targets),
  warm-start from the champion at 0.05x base LR (full LR failed), value target = 0.5 result + 0.5 search value, policy target = visit counts,
  legal-move policy per token, dedup by position.
- APEX_ROWS_README.txt: format of the search-labelled row exports (per-move sweep values, node visit/value rows).
- QUORIDOR_LEDGER_2026-08-21.md.gz: an older Rust-engine experiment ledger from that regime (gzip; `gzip -dc`). Not Ataxx data.
NOT salvaged (large, left in Downloads which the user said can be wiped): apex_rows*.zip, nets (.pt/.onnx), html demos, mge_csv.
