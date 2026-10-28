"""Extract Botzone 2200+ Elo Ground-Truth WDL Anchors & Top Engine Scores.

User Rule:
1. Top 10 engine scores of positions extracted (valuable reference, though weaker than Titanium).
2. Rest down to 2200 Elo strictly used for WDL (W, L, D, N empirical winrate).
3. Anything below 2200 Elo is worthless and excluded.
"""

import os
import re
import sys
import json
import struct

sys.path.insert(0, r"C:\gitProjects\titanium-ataxx")
from training.scrape_atx_matches import replay_match_positions
from training.stockfish_wdl_updater import canonical_key

# Verified Botzone Leaderboard >= 2200 Elo
ELITE_2200_BOTS = {
    "6aa07f142418e3081fb60ede": ("atx", 2561.13),
    "6ad3afdef80b7d76046b6c37": ("Ataxx_Python", 2559.68),
    "6ad29fd3f80b7d76046a9501": ("Ataxx_CPP", 2367.69),
    "65bc052713e80a09c4e39327": ("cheetax", 2245.45),
    "6a4132631ba515095a0bbd65": ("KataAtaxx_mini", 2224.08),
    "68ed1a4cdcee294b521dda28": ("Kissin_Flash", 2215.70),
}
ELITE_2200_NAMES = {name for name, _ in ELITE_2200_BOTS.values()}

def get_match_outcome(rec):
    bots = rec.get("bots", [])
    if len(bots) >= 2 and len(bots[0]) >= 2 and len(bots[1]) >= 2:
        try:
            s0 = int(bots[0][1])
            s1 = int(bots[1][1])
            if s0 > s1:
                return 1.0
            elif s0 < s1:
                return 0.0
            else:
                return 0.5
        except (ValueError, TypeError):
            pass
    return None

def parse_engine_score(bot_name: str, debug_str: str):
    """Extract search evaluation score / piece count from top engines' debug logs."""
    if not debug_str:
        return None
    try:
        # atx: {"v": 40, ...} (expected piece count out of 49)
        if "atx" in bot_name and '"v":' in debug_str:
            m = re.search(r'"v":\s*(-?\d+(\.\d+)?)', debug_str)
            if m:
                v = float(m.group(1))
                # convert 0..49 pieces to cp: (v - 24.5) * 50
                return (v - 24.5) * 50.0

        # cheetax: 'depth 20 ... eval score -56'
        if "cheetax" in bot_name and "eval score" in debug_str:
            m = re.search(r'eval score\s+(-?\d+)', debug_str)
            if m:
                return float(m.group(1))

        # Ataxx_CPP: 'e7369 s15253 q-0.147' -> q is Q-value (-1.0 to +1.0)
        if "CPP" in bot_name and "q" in debug_str:
            m = re.search(r'q(-?\d+\.\d+)', debug_str)
            if m:
                return float(m.group(1)) * 400.0

        # Standard UCI / xboard score: 'score 57' or 's-69'
        m = re.search(r'(?:score|s)\s*(-?\d+)', debug_str)
        if m:
            val = float(m.group(1))
            if abs(val) < 20000:
                return val
    except Exception:
        pass
    return None

def extract_anchors(raw_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "botzone_2200_anchors.jsonl")
    out_bin = os.path.join(out_dir, "botzone_2200_anchors.bin")

    print(f"Loading Botzone raw matches from {raw_path}...")
    print(f"Applying strict >= 2200 Elo filter: {sorted(ELITE_2200_NAMES)}")

    anchors = {}
    included_matches = 0
    total_positions_replayed = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            outcome_black = get_match_outcome(rec)
            if outcome_black is None:
                continue

            # Strict 2200 Elo filter: at least one bot MUST be in ELITE_2200_NAMES
            bots = rec.get("bots", [])
            b_names = [b[0].strip() for b in bots if len(b) > 0]
            if not any(any(name == b or name in b for name in ELITE_2200_NAMES) for b in b_names):
                continue

            included_matches += 1
            logs = rec.get("logs", [])
            positions = replay_match_positions(logs, [1 if outcome_black==1.0 else 0, 1 if outcome_black==0.0 else 0])

            # Extract engine debug evaluations per ply where available
            ply_scores = {}
            ply_idx = 0
            for entry in logs:
                for side_key in ("0", "1"):
                    if side_key in entry:
                        ply_idx += 1
                        side_data = entry[side_key]
                        dbg = side_data.get("debug")
                        bot_name = b_names[int(side_key)] if int(side_key) < len(b_names) else ""
                        score = parse_engine_score(bot_name, dbg)
                        if score is not None:
                            ply_scores[ply_idx] = (score, bot_name)

            for ply, occ0, occ1, blockers, turn, _ in positions:
                if ply <= 2:
                    continue
                total_positions_replayed += 1
                stm_win = (outcome_black == 1.0) if turn == 0 else (outcome_black == 0.0)
                stm_loss = (outcome_black == 0.0) if turn == 0 else (outcome_black == 1.0)

                k = canonical_key(occ0, occ1, blockers, turn)
                eng_score, eng_bot = ply_scores.get(ply, (None, None))

                if k not in anchors:
                    anchors[k] = {
                        "occ0": occ0,
                        "occ1": occ1,
                        "blockers": blockers,
                        "turn": turn,
                        "w": 1 if stm_win else 0,
                        "l": 1 if stm_loss else 0,
                        "d": 0 if (stm_win or stm_loss) else 1,
                        "n": 1,
                        "outcome_p": 1.0 if stm_win else (0.0 if stm_loss else 0.5),
                        "eng_scores": [eng_score] if eng_score is not None else [],
                    }
                else:
                    entry = anchors[k]
                    if stm_win:
                        entry["w"] += 1
                    elif stm_loss:
                        entry["l"] += 1
                    else:
                        entry["d"] += 1
                    entry["n"] += 1
                    entry["outcome_p"] = (entry["w"] + 0.5 * entry["d"]) / entry["n"]
                    if eng_score is not None:
                        entry["eng_scores"].append(eng_score)

    duplicates = sum(1 for v in anchors.values() if v["n"] > 1)
    scored_positions = sum(1 for v in anchors.values() if len(v["eng_scores"]) > 0)

    print(f"\nAnchor Extraction Results (>= 2200 Elo):")
    print(f"  Matches Included: {included_matches:,}")
    print(f"  Positions Replayed: {total_positions_replayed:,}")
    print(f"  Unique D4 Canonical Anchors: {len(anchors):,}")
    print(f"  Duplicate Positions Calibrated (2+ visits): {duplicates:,} ({duplicates/len(anchors)*100:.2f}%)")
    print(f"  Positions with Engine Evaluations: {scored_positions:,}")

    # Write output files
    with open(out_jsonl, "w", encoding="utf-8") as f_json, open(out_bin, "wb") as f_bin:
        for k, v in anchors.items():
            mean_eng = sum(v["eng_scores"]) / len(v["eng_scores"]) if v["eng_scores"] else None
            record = {
                "occ0": v["occ0"],
                "occ1": v["occ1"],
                "blockers": v["blockers"],
                "turn": v["turn"],
                "w": v["w"],
                "l": v["l"],
                "d": v["d"],
                "n": v["n"],
                "outcome_p": round(v["outcome_p"], 4),
                "engine_cp": round(mean_eng, 1) if mean_eng is not None else None,
            }
            f_json.write(json.dumps(record) + "\n")
            f_bin.write(struct.pack("<QQQB", v["occ0"], v["occ1"], v["blockers"], v["turn"]))

    print(f"Wrote {len(anchors):,} ground-truth anchors -> {out_jsonl}")
    print(f"Wrote binary index -> {out_bin} ({os.path.getsize(out_bin):,} bytes)")

if __name__ == "__main__":
    raw = r"C:\gitProjects\titanium-ataxx\data\nnue\atx_scraped\atx_matches_raw.jsonl"
    out = r"C:\gitProjects\titanium-ataxx\data\nnue\botzone_anchors"
    extract_anchors(raw, out)
