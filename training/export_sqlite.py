import argparse
import hashlib
import os
import sqlite3
import sys
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    occ0 INTEGER NOT NULL,
    occ1 INTEGER NOT NULL,
    blk INTEGER NOT NULL,
    stm INTEGER NOT NULL,
    teacher REAL,
    outcome REAL,
    sc REAL,
    wsum REAL,
    source TEXT,
    nodes_used INTEGER,
    depth INTEGER,
    best_move_share REAL,
    engine_sha256 TEXT,
    net_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS idx_pos_key ON positions(occ0, occ1, blk, stm);
CREATE INDEX IF NOT EXISTS idx_pos_source ON positions(source);
"""


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--db', required=True)
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--source', default='cache')
    ap.add_argument('--nodes-used', type=int, default=5000)
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--best-move-share', type=float, default=0.0)
    ap.add_argument('--engine', default='target/release/titanium-cli.exe')
    ap.add_argument('--net', default='data/nnue/tupS55tree.tup')
    ap.add_argument('--batch', type=int, default=50000)
    a = ap.parse_args()

    import numpy as np
    z = np.load(a.cache)
    n = len(z['occ0'])
    rows = min(a.rows, n) if a.rows else n
    eng = sha256_file(a.engine)
    net = sha256_file(a.net)
    print('cache %d rows, exporting %d -> %s' % (n, rows, a.db), flush=True)
    print('engine %s net %s' % ((eng or '?')[:12], (net or '?')[:12]), flush=True)

    con = sqlite3.connect(a.db)
    con.execute('pragma journal_mode=WAL')
    con.executescript(SCHEMA)
    occ0 = z['occ0']; occ1 = z['occ1']; blk = z['blk']; stm = z['stm']
    teacher = z['teacher'] if 'teacher' in z else None
    out = z['out'] if 'out' in z else None
    sc = z['sc'] if 'sc' in z else None
    wsum = z['wsum'] if 'wsum' in z else None
    t0 = time.time()
    cur = con.cursor()
    for lo in range(0, rows, a.batch):
        hi = min(rows, lo + a.batch)
        vals = []
        for i in range(lo, hi):
            vals.append((int(occ0[i]), int(occ1[i]), int(blk[i]), int(stm[i]),
                         float(teacher[i]) if teacher is not None else None,
                         float(out[i]) if out is not None else None,
                         float(sc[i]) if sc is not None else None,
                         float(wsum[i]) if wsum is not None else None,
                         a.source, a.nodes_used, a.depth, a.best_move_share, eng, net))
        cur.executemany('INSERT INTO positions (occ0,occ1,blk,stm,teacher,outcome,sc,wsum,'
                        'source,nodes_used,depth,best_move_share,engine_sha256,net_sha256) '
                        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', vals)
        con.commit()
        print('  %d/%d (%.0fs)' % (hi, rows, time.time() - t0), flush=True)
    con.close()
    print('done %d rows -> %s %.1fMB' % (rows, a.db, os.path.getsize(a.db) / 1e6), flush=True)


if __name__ == '__main__':
    main()
