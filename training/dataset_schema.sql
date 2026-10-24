PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    occ0 INTEGER NOT NULL,
    occ1 INTEGER NOT NULL,
    blk INTEGER NOT NULL DEFAULT 0,
    stm INTEGER NOT NULL,
    empty INTEGER NOT NULL,
    phase INTEGER NOT NULL,
    outcome REAL NOT NULL,
    source TEXT NOT NULL,
    source_file TEXT,
    source_row INTEGER,
    split TEXT NOT NULL,
    old_teacher REAL,
    engine_cp REAL,
    engine_depth INTEGER,
    engine_nodes INTEGER,
    deep_cp REAL,
    deep_depth INTEGER,
    deep_nodes INTEGER,
    regret INTEGER NOT NULL DEFAULT 0,
    target_cp REAL,
    weight REAL NOT NULL DEFAULT 1.0,
    relabel_status TEXT NOT NULL DEFAULT 'pending',
    deep_status TEXT NOT NULL DEFAULT 'none',
    label_engine_sha TEXT,
    label_net_sha TEXT,
    labeled_at INTEGER,
    quality_flags INTEGER NOT NULL DEFAULT 0,
    UNIQUE(occ0, occ1, blk, stm)
);

CREATE INDEX IF NOT EXISTS idx_pos_split ON positions(split);
CREATE INDEX IF NOT EXISTS idx_pos_source ON positions(source);
CREATE INDEX IF NOT EXISTS idx_pos_relabel ON positions(relabel_status, id);
CREATE INDEX IF NOT EXISTS idx_pos_deep ON positions(deep_status, id);
CREATE INDEX IF NOT EXISTS idx_pos_phase ON positions(phase);
CREATE INDEX IF NOT EXISTS idx_pos_regret ON positions(regret);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
