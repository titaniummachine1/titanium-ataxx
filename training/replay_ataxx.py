import sys
import json

# Ataxx board simulator to reconstruct board at any ply
# 7x7 board, 0=empty, 1=P0, 2=P1, 3=blocked

MATCH_ID = "6ae22cc69eb7df09d05b6256"
sys.path.insert(0, 'C:/gitProjects/titanium-ataxx/training')
from botzone_elo_climber import get_session

s = get_session()
r = s.get(f"https://www.botzone.org.cn/match/{MATCH_ID}")

idx = r.text.find("var _rawLogJSON = ")
sub = r.text[idx + len("var _rawLogJSON = "):]
end_idx = sub.find(";")
match_data = json.loads(json.loads(sub[:end_idx].strip()))

# Standard Ataxx board init:
# P0 at (0,0) and (6,6)
# P1 at (0,6) and (6,0)
board = [[0]*7 for _ in range(7)]
board[0][0] = 1
board[6][6] = 1
board[0][6] = 2
board[6][0] = 2

def print_board(b, ply, last_move, eval_str):
    p0_c = sum(row.count(1) for row in b)
    p1_c = sum(row.count(2) for row in b)
    print(f"\n--- Ply {ply} (P0: {p0_c}, P1: {p1_c}) | {last_move} | {eval_str} ---")
    print("  0 1 2 3 4 5 6")
    for y in range(7):
        row_str = " ".join("." if b[x][y] == 0 else ("X" if b[x][y] == 1 else "O") for x in range(7))
        print(f"{y} {row_str}")

ply = 0
for step in match_data:
    for pid in ("0", "1"):
        if pid in step and isinstance(step[pid], dict) and "response" in step[pid]:
            ply += 1
            resp = step[pid]["response"]
            dbg = step[pid].get("debug", "")
            x0, y0 = resp["x0"], resp["y0"]
            x1, y1 = resp["x1"], resp["y1"]
            p_val = 1 if pid == "0" else 2
            opp_val = 2 if pid == "0" else 1

            if x0 != -1 and x1 != -1:
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                dist = max(dx, dy)
                if dist == 2:
                    board[x0][y0] = 0  # jump
                board[x1][y1] = p_val  # land
                # convert adjacent
                for nx in range(max(0, x1-1), min(7, x1+2)):
                    for ny in range(max(0, y1-1), min(7, y1+2)):
                        if board[nx][ny] == opp_val:
                            board[nx][ny] = p_val
            
            if ply in (78, 80, 82, 84, 86, 88, 90, 96, 108):
                print_board(board, ply, f"P{pid}: ({x0},{y0})->({x1},{y1})", dbg)
