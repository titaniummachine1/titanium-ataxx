"""Scrape atx and top-bot Ataxx matches from Botzone and extract positions for NNUE training."""

import json
import os
import re
import struct
import time
import requests

ATAXX_GAME_ID = "5809c7647f65182b044b5e3b"
ATX_BOT_ID = "6aa07f142418e3081fb60ede"
TOP_BOT_IDS = {
    "6aa07f142418e3081fb60ede": "atx",
    "6ad3afdef80b7d76046b6c37": "Ataxx_Python",
    "6a3bd61a7e8f174717a20396": "aaa",
    "5af3313d1c45dc7bf64a16f7": "Master",
}

OUT_DIR = r"C:\gitProjects\titanium-ataxx\data\nnue\atx_scraped"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            r = s.post(
                "https://www.botzone.org.cn/login",
                data={"email": "terminatort8004@gmail.com", "password": "beniamin2005"},
                timeout=20,
            )
            r.raise_for_status()
            return s
        except Exception as e:
            print(f"Login retry {attempt+1}: {e}")
            time.sleep(2.0)
    return s


def safe_get(s: requests.Session, url: str):
    for attempt in range(5):
        try:
            r = s.get(url, timeout=20)
            if r.status_code == 200:
                return r
        except Exception as e:
            print(f"  GET retry {attempt+1} for {url}: {e}")
        time.sleep(1.5 * (attempt + 1))
    return None


def replay_match_positions(log_entries: list, final_scores: list):
    """Replay a Botzone Ataxx match log and yield (ply, occ0, occ1, blockers, turn, outcome_black)."""
    grid = [[0] * 7 for _ in range(7)]
    grid[0][0] = 1
    grid[6][6] = 1
    grid[6][0] = 2
    grid[0][6] = 2
    blockers = 0

    if len(final_scores) >= 2:
        if final_scores[0] > final_scores[1]:
            outcome_black = 1.0
        elif final_scores[0] < final_scores[1]:
            outcome_black = 0.0
        else:
            outcome_black = 0.5
    else:
        outcome_black = 0.5

    def grid_to_bbs():
        occ0 = 0
        occ1 = 0
        for x in range(7):
            for y in range(7):
                sq = y * 7 + x
                if grid[x][y] == 1:
                    occ0 |= 1 << sq
                elif grid[x][y] == 2:
                    occ1 |= 1 << sq
        return occ0, occ1

    def apply_move(color: int, mv: dict):
        x0, y0, x1, y1 = mv["x0"], mv["y0"], mv["x1"], mv["y1"]
        if x0 < 0 or x1 < 0:
            return False
        if not (0 <= x0 < 7 and 0 <= y0 < 7 and 0 <= x1 < 7 and 0 <= y1 < 7):
            return False
        if max(abs(x1 - x0), abs(y1 - y0)) > 1:
            grid[x0][y0] = 0
        grid[x1][y1] = color
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x1 + dx, y1 + dy
                if 0 <= nx < 7 and 0 <= ny < 7 and grid[nx][ny] == (3 - color):
                    grid[nx][ny] = color
        return True

    ply = 0
    positions = []
    for entry in log_entries:
        if "0" in entry or "1" in entry:
            side_key = "0" if "0" in entry else "1"
            turn = 0 if side_key == "0" else 1
            color = 1 if turn == 0 else 2
            side_data = entry[side_key]
            verdict = side_data.get("verdict")
            resp = side_data.get("response")
            if verdict != "OK" or not isinstance(resp, dict):
                break
            occ0, occ1 = grid_to_bbs()
            positions.append((ply + 1, occ0, occ1, blockers, turn, outcome_black))
            if not apply_move(color, resp):
                break
            ply += 1
    return positions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_jsonl_path = os.path.join(OUT_DIR, "atx_matches_raw.jsonl")
    pos_jsonl_path = os.path.join(OUT_DIR, "atx_positions.jsonl")
    pos_bin_path = os.path.join(OUT_DIR, "atx_positions.bin")

    seen_match_ids = set()
    existing_records = []
    last_seen_id = None
    if os.path.exists(raw_jsonl_path):
        with open(raw_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                m_id = rec["match_id"]
                seen_match_ids.add(m_id)
                existing_records.append(rec)
                last_seen_id = m_id
        print(f"Loaded {len(existing_records)} existing matches (oldest id: {last_seen_id})")

    s = make_session()

    start_id = last_seen_id
    max_pages = 120
    downloaded = 0

    with open(raw_jsonl_path, "a", encoding="utf-8") as f_raw:
        for page in range(1, max_pages + 1):
            url = f"https://www.botzone.org.cn/globalmatchlist?game={ATAXX_GAME_ID}"
            if start_id:
                url += f"&startid={start_id}"
            r = safe_get(s, url)
            if r is None:
                print(f"Page {page} failed after retries, stopping.")
                break
            parts = r.text.split("<!--DYNLOADBEGIN-->")
            content = parts[5].split("<!--DYNLOADEND-->")[0] if len(parts) > 5 else r.text
            rows = content.split("<tr>")[2:]
            if not rows:
                break

            page_targets = []
            oldest_on_page = None
            for row in rows:
                m_match = re.search(r"/match/([0-9a-f]{24})", row)
                if not m_match:
                    continue
                m_id = m_match.group(1)
                oldest_on_page = m_id
                if m_id in seen_match_ids:
                    continue
                seen_match_ids.add(m_id)

                has_atx = ATX_BOT_ID in row
                has_top = any(bid in row for bid in TOP_BOT_IDS)
                bots = re.findall(r'<a class="botname[^"]*">([^<]+)<span[^>]*>(\d+)</span>', row)
                scores = re.findall(r'<div class="score pull-right">(\d+)</div>', row)
                if has_atx:
                    page_targets.append(("atx", m_id, bots, scores))
                elif has_top:
                    page_targets.append(("top", m_id, bots, scores))

            for tag, m_id, bots, scores in page_targets:
                m_url = f"https://www.botzone.org.cn/match/{m_id}?lite=true"
                rm = safe_get(s, m_url)
                if rm is None:
                    continue
                try:
                    data = rm.json()
                except Exception:
                    continue
                rec = {
                    "match_id": m_id,
                    "tag": tag,
                    "bots": bots,
                    "scores": data.get("scores", []),
                    "logs": data.get("logs", []),
                }
                f_raw.write(json.dumps(rec) + "\n")
                f_raw.flush()
                existing_records.append(rec)
                downloaded += 1
                time.sleep(0.12)

            if page % 5 == 0 or page == 1:
                print(
                    f"Page {page}: downloaded +{len(page_targets)} (session new={downloaded}, total={len(existing_records)}, oldest={oldest_on_page})",
                    flush=True,
                )
            if oldest_on_page and oldest_on_page[:8] < "6aa00000":
                print(f"Reached matches older than atx creation ({oldest_on_page}), stopping pagination.")
                break
            next_match = re.search(
                r"/globalmatchlist\?startid=([0-9a-f]{24})&game=" + ATAXX_GAME_ID, content
            )
            if not next_match:
                break
            start_id = next_match.group(1)
            time.sleep(0.2)

    unique_positions = {}
    atx_games_count = 0
    top_games_count = 0
    for rec in existing_records:
        if rec.get("tag") == "atx":
            atx_games_count += 1
        else:
            top_games_count += 1
        m_id = rec["match_id"]
        positions = replay_match_positions(rec.get("logs", []), rec.get("scores", []))
        for ply, occ0, occ1, blockers, turn, outcome_black in positions:
            if ply <= 2:
                continue
            key = (occ0, occ1, blockers, turn)
            if key not in unique_positions:
                unique_positions[key] = {
                    "match_id": m_id,
                    "tag": rec.get("tag", "atx"),
                    "ply": ply,
                    "occ0": occ0,
                    "occ1": occ1,
                    "blockers": blockers,
                    "turn": turn,
                    "outcome_black": outcome_black,
                }

    with open(pos_jsonl_path, "w", encoding="utf-8") as f_jsonl, open(pos_bin_path, "wb") as f_bin:
        for key, item in unique_positions.items():
            f_jsonl.write(json.dumps(item) + "\n")
            occ0, occ1, blockers, turn = key
            f_bin.write(struct.pack("<QQQB", occ0, occ1, blockers, turn))

    print(
        f"\nAll done! Total matches in archive: {len(existing_records)} (atx={atx_games_count}, other_top={top_games_count})."
    )
    print(f"Total unique positions extracted: {len(unique_positions)}")
    print(f"  - {pos_bin_path} ({len(unique_positions) * 25} bytes)")
    print(f"  - {pos_jsonl_path}")
    print(f"  - {raw_jsonl_path}")


if __name__ == "__main__":
    main()
