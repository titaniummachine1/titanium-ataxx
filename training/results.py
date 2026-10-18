#!/usr/bin/env python3
"""Experiment results store. sqlite3 today, Turso/libsql tomorrow: same SQL
dialect, swap the driver in connect() — one line. NO JSON anywhere: configs
are COLUMNS, notes are TEXT.

Tables:
  gates(id, ts, kind, engine_a, engine_b, tc, nodes, ms, games, w, l, d,
        score, ci_lo, ci_hi, log, note)
  trainings(id, ts, name, cache, teacher, blend, ow, lr, dirs, seed, epochs,
        best_ep, val, val0, out, log, note)
Usage:
  python training/results.py init
  python training/results.py backfill   # parse logs/*.txt totals
  python training/results.py gates      # print recent gates
"""
import math
import os
import re
import sqlite3
import sys
import time

DB = os.path.join('data', 'nnue', 'results.db')


def connect():
    try:
        import libsql_experimental as libsql
        return libsql.connect(DB)
    except ImportError:
        return sqlite3.connect(DB)


SCHEMA = """
CREATE TABLE IF NOT EXISTS gates(
  id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, engine_a TEXT, engine_b TEXT,
  tc TEXT, nodes INTEGER, ms INTEGER, games INTEGER, w INTEGER, l INTEGER,
  d INTEGER, score REAL, ci_lo REAL, ci_hi REAL, log TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS trainings(
  id INTEGER PRIMARY KEY, ts TEXT, name TEXT, cache TEXT, teacher TEXT,
  blend REAL, ow REAL, lr REAL, dirs TEXT, seed INTEGER, epochs INTEGER,
  best_ep INTEGER, val REAL, val0 REAL, out TEXT, log TEXT, note TEXT);
"""


def cmd_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = connect()
    c.executescript(SCHEMA)
    c.commit()
    c.close()
    print('init %s' % DB)


def add_gate(kind, a, b, tc, nodes, ms, games, w, l, d, log, note=''):
    n = w + l + d
    s = (w + d / 2) / n if n else 0.0
    se = math.sqrt(s * (1 - s) / n) if n else 0.0
    c = connect()
    c.execute(
        'INSERT INTO gates(ts,kind,engine_a,engine_b,tc,nodes,ms,games,w,l,d,'
        'score,ci_lo,ci_hi,log,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (time.strftime('%Y-%m-%d %H:%M:%S'), kind, a, b, tc, nodes, ms,
         games, w, l, d, round(s, 4), round(s - 1.96 * se, 4),
         round(s + 1.96 * se, 4), log, note))
    c.commit()
    c.close()
    return s


def add_training(name, cache='', teacher='', blend=0.0, ow=0.0, lr=0.0,
                 dirs='', seed=0, epochs=0, best_ep=0, val=0.0, val0=0.0,
                 out='', log='', note=''):
    c = connect()
    c.execute(
        'INSERT INTO trainings(ts,name,cache,teacher,blend,ow,lr,dirs,seed,'
        'epochs,best_ep,val,val0,out,log,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (time.strftime('%Y-%m-%d %H:%M:%S'), name, cache, teacher, blend, ow,
         lr, dirs, seed, epochs, best_ep, val, val0, out, log, note))
    c.commit()
    c.close()


TOTAL_RE = re.compile(
    r'total: titanium (\d+) - .*? (\d+), draws (\d+)')
BOOK_TOTAL_RE = re.compile(
    r'total: titan (\d+) - moon (\d+) draws (\d+)')
HEADER_RE = re.compile(r'titanium limits: (.*?), net: (.*?) vs (.*)')


def parse_log(path):
    """(games, w, l, d, tc_desc) from a match or book log. None on failure."""
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return None
    m = TOTAL_RE.search(txt)
    if m:
        w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        h = HEADER_RE.search(txt)
        tc = h.group(1).strip() if h else '?'
        games = len(re.findall(r'^-- game \d+:', txt, re.M))
        return games, w, l, d, tc
    m = BOOK_TOTAL_RE.search(txt)
    if m:
        w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        games = len(re.findall(r'^game \S+ \[titan=', txt, re.M))
        return games, w, l, d, 'book'
    return None


KEEP_PREFIX = ('s30', 's31', 's32', 's33', 's34', 's35', 's36', 's37',
                 's38', 's39', 's40', 's41', 's42', 's43', 's44', 's45',
                 's46', 's47', 's48', 'book_gate', 'opening_gate')


def cmd_backfill():
    import glob
    c = connect()
    seen = {r[0].replace('/', '\\') for r in c.execute('SELECT log FROM gates')}
    c.close()
    n = skip = 0
    for path in sorted(glob.glob('logs/*.txt')):
        if path.replace('/', '\\') in seen:
            skip += 1
            continue
        r = parse_log(path)
        if not r:
            continue
        base = os.path.basename(path)
        games, w, l, d, tc = r
        nodes = ms = 0
        m = re.search(r'(\d+) nodes', tc)
        if m:
            nodes = int(m.group(1))
        m = re.search(r'(\d+) ms', tc)
        if m:
            ms = int(m.group(1))
        add_gate('match', 'titanium', 'opp', tc, nodes, ms, games,
                 w, l, d, path, note='backfill:' + base)
        n += 1
    print('backfilled %d gates (%d already present)' % (n, skip))


def cmd_gates():
    c = connect()
    for r in c.execute(
            'SELECT id,tc,games,w,l,d,score,ci_lo,ci_hi,log,note FROM gates '
            'ORDER BY id DESC LIMIT 30'):
        print('%s | %s %dg | %d-%d-%d s=%.3f [%.3f,%.3f] %s %s' % r)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'gates'
    if cmd == 'init':
        cmd_init()
    elif cmd == 'backfill':
        cmd_backfill()
    elif cmd == 'gates':
        cmd_gates()
    else:
        print('usage: results.py [init|backfill|gates]')


if __name__ == '__main__':
    main()
