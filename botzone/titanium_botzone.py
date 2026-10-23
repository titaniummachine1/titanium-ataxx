"""Titanium Ataxx -- Botzone Python (long-live, Simple IO) driver.

All Botzone logic lives HERE: request parsing, game replay after a restart, the clock, the safety
margin and the watchdog. The engine is a plain shared library (titanium_ffi) that only knows
new_game / apply / search(budget_ms) / stop. Layout expected on Botzone (data files are uploaded
next to the bot):
    data/titanium_v2.so         engine library (Linux, glibc >= 2.14)  [new name: the old bot's data/titanium.so stays intact]
    data/nnue/tupS57v2.tup      evaluation net (726440 B)
Local testing overrides (environment): TI_LIB, TI_NET, TI_TT_BITS, TI_TIME_SCALE.

Time policy: for every move the driver knows its limit (first move 12 s, later moves 4 s on the
Python long-live setting), subtracts a 50 ms safety margin and the time already spent since the
request arrived, and searches for the rest. A watchdog thread calls ti_stop shortly after the
budget as a last resort.
"""
import ctypes
import os
import sys
import threading
import time

T_START = time.perf_counter()

MARKER = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
FIRST_LIMIT_MS = 12000      # first move (site Python config [6, 4]: 6 s normal, first doubled)
TURN_LIMIT_MS = 4000        # later moves in long-live mode
SAFETY_MS = 50              # always finish at least this much before the limit
STARTUP_MS = 120            # interpreter start + imports happen before T_START; charged to move 1
OUT_RESERVE_MS = 2          # printing / pipe latency
WATCHDOG_GRACE_MS = 12      # ti_stop is sent this long after the budget if the search is still running
SCALE = float(os.environ.get("TI_TIME_SCALE", "1.0"))   # tests only: shrink all limits
DEBUG = os.environ.get("TI_DEBUG") == "1"               # per-move timing detail on stderr

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    sys.stderr.write("[titanium] %s\n" % msg)
    sys.stderr.flush()


def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def load_library():
    cands = [os.environ.get("TI_LIB"), os.path.join("data", "titanium_v2.so"), os.path.join(HERE, "data", "titanium_v2.so"),
             os.path.join(HERE, "titanium_ffi.dll"), os.path.join(HERE, "libtitanium_ffi.so")]
    path = first_existing(cands)
    if path is None:
        raise SystemExit("titanium library not found; tried %r" % (cands,))
    lib = ctypes.CDLL(os.path.abspath(path))
    lib.ti_abi_version.restype = ctypes.c_uint32
    lib.ti_new.restype = ctypes.c_void_p
    lib.ti_new.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.ti_free.argtypes = [ctypes.c_void_p]
    lib.ti_new_game.argtypes = [ctypes.c_void_p]
    lib.ti_apply.restype = ctypes.c_int
    lib.ti_apply.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
    lib.ti_legal_count.restype = ctypes.c_int
    lib.ti_legal_count.argtypes = [ctypes.c_void_p]
    lib.ti_stop.argtypes = [ctypes.c_void_p]
    lib.ti_search.restype = ctypes.c_int
    lib.ti_search.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
    if lib.ti_abi_version() != 1:
        raise SystemExit("unexpected titanium ABI version %d" % lib.ti_abi_version())
    return lib, path


class Stats(ctypes.Structure):
    _fields_ = [("nodes", ctypes.c_uint64), ("elapsed_us", ctypes.c_uint64), ("score", ctypes.c_int32),
                ("depth", ctypes.c_uint32), ("stopped_by", ctypes.c_uint32), ("repetition_override", ctypes.c_uint32)]


def make_engine(lib):
    tt_bits = int(os.environ.get("TI_TT_BITS", "22"))
    nets = [os.environ.get("TI_NET"), os.path.join("data", "nnue", "tupS57v2.tup"),
            os.path.join(HERE, "data", "nnue", "tupS57v2.tup"), os.path.join("data", "nnue", "tupS56twostage.tup"),
            os.path.join("data", "nnue", "tupS55tree.tup")]
    for p in nets:
        if p and os.path.exists(p):
            e = lib.ti_new(os.path.abspath(p).encode("utf-8"), tt_bits)
            if e:
                log("net %s, tt_bits %d" % (p, tt_bits))
                return e
            log("net %s failed to load" % p)
    log("WARNING: no net loaded, classical eval (weak)")
    return lib.ti_new(None, tt_bits)


class Reader:
    """Token reader over stdin that remembers when the first byte of the current request arrived."""

    def __init__(self):
        self.buf = []
        self.t_req = T_START

    def ints(self, n):
        got = []
        first = True
        while len(got) < n:
            if not self.buf:
                line = sys.stdin.readline()
                if not line:
                    return None
                if first and line.strip():
                    self.t_req = time.perf_counter()
                self.buf = line.split()
                continue
            tok = self.buf.pop(0)
            try:
                got.append(int(tok))
                first = False
            except ValueError:
                pass
        return got


def main():
    lib, libpath = load_library()
    engine = make_engine(lib)
    lib.ti_new_game(engine)
    out = (ctypes.c_int * 4)()
    stats = Stats()
    lib.ti_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(Stats)]
    rd = Reader()
    history = []            # every move applied this game, both sides: (x0, y0, x1, y1)

    def apply_move(m):
        r = lib.ti_apply(engine, *m)
        if r == 1:
            history.append(tuple(m))
            return True
        if r == 0:
            return True      # start marker, nothing to apply
        # desync: rebuild the position from our own record and retry once
        log("ti_apply rejected %r; replaying %d moves" % (m, len(history)))
        lib.ti_new_game(engine)
        for h in history:
            lib.ti_apply(engine, *h)
        if lib.ti_apply(engine, *m) == 1:
            history.append(tuple(m))
            return True
        log("resync failed for %r" % (m,))
        return False

    # ---- turn 1: "T" then 2*T-1 moves (request/response pairs ending with the current request)
    head = rd.ints(1)
    if head is None:
        return
    turns = head[0]
    flat = rd.ints(4 * (2 * turns - 1))
    if flat is None:
        return
    for i in range(0, len(flat), 4):
        apply_move(flat[i:i + 4])
    first = True
    t_origin = T_START - STARTUP_MS / 1000.0     # the clock for move 1 started before Python did
    while True:
        limit_ms = (FIRST_LIMIT_MS if first else TURN_LIMIT_MS) * SCALE
        t0 = t_origin if first else rd.t_req
        spent_ms = (time.perf_counter() - t0) * 1000.0
        budget = int(limit_ms - SAFETY_MS * min(1.0, SCALE * 4) - OUT_RESERVE_MS - spent_ms)
        budget = max(5, budget)
        fired = []

        def watchdog():
            fired.append(1)
            lib.ti_stop(engine)

        timer = threading.Timer((budget + WATCHDOG_GRACE_MS) / 1000.0, watchdog)
        timer.daemon = True
        timer.start()
        ts = time.perf_counter()
        res = lib.ti_search(engine, budget, out)
        search_ms = (time.perf_counter() - ts) * 1000.0
        timer.cancel()
        if DEBUG:
            log("move budget=%d search_ms=%.1f over=%.1f watchdog=%d" % (budget, search_ms, search_ms - budget, len(fired)))
        if res == 1:
            mv = (out[0], out[1], out[2], out[3])
            history.append(mv)
            lib.ti_stats(engine, ctypes.byref(stats))
            dbg = "d%d n%d t%.0f s%d%s" % (stats.depth, stats.nodes, (time.perf_counter() - t0) * 1000.0, stats.score,
                                           " rep" if stats.repetition_override else "")
            sys.stdout.write("%d %d %d %d\n%s\n\n\n%s\n" % (mv[0], mv[1], mv[2], mv[3], dbg, MARKER))
        else:
            # no legal move: the judge ends the game before asking; answer anyway so we never hang
            sys.stdout.write("-1 -1 -1 -1\nno-move\n\n\n%s\n" % MARKER)
        sys.stdout.flush()
        first = False
        req = rd.ints(4)
        if req is None:
            break
        apply_move(req)


if __name__ == "__main__":
    main()
