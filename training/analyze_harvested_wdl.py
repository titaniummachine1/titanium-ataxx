import sys
import os
import json

sys.path.insert(0, r"C:\gitProjects\titanium-ataxx")
from training.scrape_atx_matches import replay_match_positions
from training.stockfish_wdl_updater import canonical_key

ELITE_2200_NAMES = {"atx", "Ataxx_Python", "Ataxx_CPP", "cheetax", "KataAtaxx_mini", "Kissin_Flash"}
ELITE_2100_NAMES = set(ELITE_2200_NAMES) | {"PyJataxx"}

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

def analyze_harvest(filter_mode="elite_both_2100"):
    raw_path = r"C:\gitProjects\titanium-ataxx\data\nnue\atx_scraped\atx_matches_raw.jsonl"
    if not os.path.exists(raw_path):
        print("No matches file found at", raw_path)
        return

    print(f"\n=======================================================")
    print(f"Analyzing harvest under filter: {filter_mode}")
    print(f"=======================================================")
    stats = {}
    total_pos = 0
    matches_used = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            outcome_black = get_match_outcome(rec)
            if outcome_black is None:
                continue

            bots = rec.get("bots", [])
            b_names = [b[0].strip() for b in bots if len(b) > 0]
            
            if filter_mode == "elite_both_2200":
                if not (len(b_names) >= 2 and all(any(name == b or name in b for name in ELITE_2200_NAMES) for b in b_names[:2])):
                    continue
            elif filter_mode == "elite_both_2100":
                if not (len(b_names) >= 2 and all(any(name == b or name in b for name in ELITE_2100_NAMES) for b in b_names[:2])):
                    continue
            elif filter_mode == "elite_any_2100":
                if not any(any(name == b or name in b for name in ELITE_2100_NAMES) for b in b_names):
                    continue

            matches_used += 1
            positions = replay_match_positions(rec.get("logs", []), [1 if outcome_black==1.0 else 0, 1 if outcome_black==0.0 else 0])
            for ply, occ0, occ1, blockers, turn, _ in positions:
                if ply <= 2:
                    continue
                total_pos += 1
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
    print(f"Matches included: {matches_used:,}")
    print(f"Total positions replayed: {total_pos:,}")
    print(f"Unique D4 canonical positions: {len(stats):,}")
    print(f"Duplicate positions (2+ visits): {duplicates:,} ({duplicates/max(1, len(stats))*100:.2f}%)")
    print(f"Positions with contested outcomes (W and L): {contested:,}")

    sorted_anchors = sorted(stats.items(), key=lambda x: x[1]["n"], reverse=True)
    print("Top 5 Most Visited Elite Anchors:")
    for i, (k, v) in enumerate(sorted_anchors[:5], 1):
        wdl_p = (v["w"] + 0.5 * v["d"]) / v["n"]
        print(f"  #{i} Visited {v['n']:<3} times | W={v['w']:<3} L={v['l']:<3} D={v['d']:<2} -> WDL Winrate: {wdl_p:.4f}")

if __name__ == "__main__":
    analyze_harvest("elite_both_2200")
    analyze_harvest("elite_both_2100")
    analyze_harvest("elite_any_2100")
