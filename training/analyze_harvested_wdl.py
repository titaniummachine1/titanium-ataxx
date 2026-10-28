import sys
import os
import json

sys.path.insert(0, r"C:\gitProjects\titanium-ataxx")
from training.scrape_atx_matches import replay_match_positions
from training.stockfish_wdl_updater import canonical_key

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

def analyze_harvest():
    raw_path = r"C:\gitProjects\titanium-ataxx\data\nnue\atx_scraped\atx_matches_raw.jsonl"
    if not os.path.exists(raw_path):
        print("No matches file found at", raw_path)
        return

    print("Replaying harvested Botzone matches under D4 dihedral symmetry with real scores...")
    stats = {}
    total_pos = 0
    valid_matches = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            outcome_black = get_match_outcome(rec)
            if outcome_black is None:
                continue
            valid_matches += 1
            positions = replay_match_positions(rec.get("logs", []), [1 if outcome_black==1.0 else 0, 1 if outcome_black==0.0 else 0])
            for ply, occ0, occ1, blockers, turn, _ in positions:
                if ply <= 2:
                    continue
                total_pos += 1
                # turn 0 = black (stm = 0), turn 1 = white (stm = 1)
                stm_win = (outcome_black == 1.0) if turn == 0 else (outcome_black == 0.0)
                stm_loss = (outcome_black == 0.0) if turn == 0 else (outcome_black == 1.0)

                k = canonical_key(occ0, occ1, blockers, turn)
                if k not in stats:
                    stats[k] = {"w": 0, "l": 0, "d": 0, "n": 0}
                if stm_win:
                    stats[k]["w"] += 1
                elif stm_loss:
                    stats[k]["l"] += 1
                else:
                    stats[k]["d"] += 1
                stats[k]["n"] += 1

    duplicates = sum(1 for v in stats.values() if v["n"] > 1)
    contested = sum(1 for v in stats.values() if v["n"] > 1 and v["w"] > 0 and v["l"] > 0)
    print(f"\nHarvest Replay Summary:")
    print(f"  Total positions replayed: {total_pos:,}")
    print(f"  Unique D4 canonical positions: {len(stats):,}")
    print(f"  Positions visited 2+ times (Duplicates): {duplicates:,} ({duplicates/len(stats)*100:.2f}%)")
    print(f"  Positions with contested outcomes (W and L): {contested:,}")

    sorted_anchors = sorted(stats.items(), key=lambda x: x[1]["n"], reverse=True)
    print("\nTop 10 Most Frequently Visited Positions (Ground Truth Anchors):")
    for i, (k, v) in enumerate(sorted_anchors[:10], 1):
        wdl_p = (v["w"] + 0.5 * v["d"]) / v["n"]
        print(f"  #{i:<2} Visited {v['n']:<3} times | W={v['w']:<3} L={v['l']:<3} D={v['d']:<2} -> Calibrated WDL Winrate: {wdl_p:.4f}")

if __name__ == "__main__":
    analyze_harvest()
