"""Game DAG (user design 2026-10-24, after Quoridor_best_AI): every position reached in our played games becomes a node,
every move an edge, with W/D/L counts from the side to move's view. Gates and matches FEED it after they finish (read-only
on their logs: it never touches a running gate). Later: opening WDL statistics (book + training targets) and self-play that
walks the known DAG with temperature and plays the engine's best once off-DAG.

Storage: SQLite data/dag/games_dag.db (on disk, RAM-safe).
  games(src, idx, header, book, white_is, result, plies)           one row per ingested game; (src, idx) unique -> idempotent
  nodes(o0, o1, blk, stm, n, w, d, l, min_ply)                      w/d/l from stm's view
  edges(o0, o1, blk, stm, mv, c0, c1, cblk, cstm, n, w, d, l)       w/d/l from the PARENT stm's view
Positions are exact bitboards (occ0 = 'x', occ1 = 'o', our row-major index), blk = blocked squares, stm 0 = x to move.
Usage:
  python training/dag.py ingest "logs/gate_*.txt" "logs/trip_*.txt"     (globs; already-ingested games are skipped)
  python training/dag.py stats
"""
import glob
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ataxx_move import apply_move, parse_board  # noqa: E402

ROOT = "C:/gitProjects/titanium-ataxx"
DB = ROOT + "/data/dag/games_dag.db"
BOOK = ROOT + "/data/nnue/book50k.txt"
GAME = re.compile(r"^-- game (\d+): book (\d+) titanium=(\w) opp=(\w): (titanium wins|opponent wins|draw)\D*(\d+)-(\d+) \((\d+) plies\)")


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS games(src TEXT, idx INTEGER, header TEXT, book INTEGER, titanium_is TEXT, result TEXT,
        plies INTEGER, PRIMARY KEY(src, idx));
    CREATE TABLE IF NOT EXISTS nodes(o0 INTEGER, o1 INTEGER, blk INTEGER, stm INTEGER, n INTEGER, w INTEGER, d INTEGER,
        l INTEGER, min_ply INTEGER, PRIMARY KEY(o0, o1, blk, stm));
    CREATE TABLE IF NOT EXISTS edges(o0 INTEGER, o1 INTEGER, blk INTEGER, stm INTEGER, mv TEXT, c0 INTEGER, c1 INTEGER,
        cblk INTEGER, cstm INTEGER, n INTEGER, w INTEGER, d INTEGER, l INTEGER, PRIMARY KEY(o0, o1, blk, stm, mv));
    """)
    return c


def parse_log(path):
    """yield (idx, header, book_line, titanium_colour, result_for_x, plies, moves)"""
    header = ""
    cur = None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        if line.startswith("Match:"):
            header = line.strip()
            continue
        m = GAME.match(line)
        if m:
            idx, book, ti = int(m.group(1)), int(m.group(2)), m.group(3)
            res = m.group(5)
            # winner colour: titanium side ti, opponent the other
            if res == "draw":
                rx = 0.5
            else:
                winner = ti if res == "titanium wins" else ("o" if ti == "x" else "x")
                rx = 1.0 if winner == "x" else 0.0
            cur = [idx, header, book, ti, rx, int(m.group(8))]
            continue
        if cur is not None and line.strip() and not line.startswith("--") and not line.startswith("total"):
            yield (*cur, line.split())
            cur = None


def ingest(patterns):
    book = open(BOOK, encoding="utf-8").read().splitlines()
    c = connect()
    # pooled copies and determinism tripwires repeat games already in the leg logs: never ingest them
    files = sorted(set(f for p in patterns for f in glob.glob(p)
                       if not any(t in os.path.basename(f) for t in ("_pooled", "_trip", "trip_"))))
    new_games = new_pos = bad = 0
    for f in files:
        src = os.path.basename(f)
        done = set(r[0] for r in c.execute("SELECT idx FROM games WHERE src=?", (src,)))
        rows = []
        for idx, header, bk, ti, rx, plies, moves in parse_log(f):
            if idx in done:
                continue
            if bk >= len(book):
                continue
            st = parse_board(book[bk])
            o0, o1, blk, stm = st
            node_upd, edge_upd = [], []
            ply = 0
            ok = True
            for tok in moves:
                rs = rx if stm == 0 else 1.0 - rx              # result from the side to move's view
                w, d, l = int(rs == 1.0), int(rs == 0.5), int(rs == 0.0)
                node_upd.append((o0, o1, blk, stm, w, d, l, ply))
                try:
                    n0, n1, nb, ns = apply_move(o0, o1, blk, stm, tok)
                except Exception:
                    ok = False
                    break
                edge_upd.append((o0, o1, blk, stm, tok, n0, n1, nb, ns, w, d, l))
                o0, o1, blk, stm = n0, n1, nb, ns
                ply += 1
            if not ok:
                bad += 1
                continue
            rs = rx if stm == 0 else 1.0 - rx
            node_upd.append((o0, o1, blk, stm, int(rs == 1.0), int(rs == 0.5), int(rs == 0.0), ply))
            c.executemany("""INSERT INTO nodes VALUES(?,?,?,?,1,?,?,?,?) ON CONFLICT(o0,o1,blk,stm) DO UPDATE SET
                n=n+1, w=w+excluded.w, d=d+excluded.d, l=l+excluded.l, min_ply=MIN(min_ply, excluded.min_ply)""", node_upd)
            c.executemany("""INSERT INTO edges VALUES(?,?,?,?,?,?,?,?,?,1,?,?,?) ON CONFLICT(o0,o1,blk,stm,mv) DO UPDATE SET
                n=n+1, w=w+excluded.w, d=d+excluded.d, l=l+excluded.l""", edge_upd)
            rows.append((src, idx, header, bk, ti, "x" if rx == 1.0 else ("o" if rx == 0.0 else "draw"), plies))
            new_pos += len(node_upd)
        c.executemany("INSERT INTO games VALUES(?,?,?,?,?,?,?)", rows)
        c.commit()
        new_games += len(rows)
        if rows:
            print("  %-40s +%d games" % (src, len(rows)), flush=True)
    print("ingested %d new games (%d position visits) from %d files; %d games skipped (illegal replay)" % (new_games, new_pos, len(files), bad))
    stats(c)


def stats(c=None):
    c = c or connect()
    g = c.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    n, visits = c.execute("SELECT COUNT(*), COALESCE(SUM(n),0) FROM nodes").fetchone()
    e = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    multi = c.execute("SELECT COUNT(*) FROM nodes WHERE n >= 2").fetchone()[0]
    roots = c.execute("SELECT COUNT(*) FROM nodes WHERE min_ply = 0").fetchone()[0]
    print("DAG: {} games | {} nodes ({} visits, {} seen >= 2x) | {} edges | {} root positions".format(g, n, visits, multi, e, roots))
    for ply in (0, 1, 2, 4, 8):
        k = c.execute("SELECT COUNT(*), COALESCE(SUM(n),0) FROM nodes WHERE min_ply = ?", (ply,)).fetchone()
        print("   ply %d: %d distinct positions, %d visits" % (ply, k[0], k[1]))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest(sys.argv[2:] or ["logs/gate_*.txt"])
    else:
        stats()
