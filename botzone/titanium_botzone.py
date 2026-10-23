"""Titanium Ataxx -- Botzone Python (Simple IO) driver. Default mode: ONE-SHOT (a fresh process per move,\n6 s limit, the whole game history is sent every time). Set TI_MODE=long for the long-live variant.\n\nOriginal description:

All Botzone logic lives HERE: request parsing, game replay after a restart, the clock, the safety
margin and the watchdog. The engine is a plain shared library (titanium_ffi) that only knows
new_game / apply / search(budget_ms) / stop. Layout expected on Botzone (data files are uploaded
next to the bot):
    data/titanium_v2.so         engine library (Linux, glibc >= 2.14)  [new name: the old bot's data/titanium.so stays intact]
    data/nnue/tupS57v2.tup      evaluation net (726440 B)
Local testing overrides (environment): TI_LIB, TI_NET, TI_TT_BITS, TI_TIME_SCALE.

Time policy: for every move the driver knows its limit (one-shot: 6 s every move; long-live: 12 s first,
4 s later), subtracts a 50 ms safety margin and the time already spent (one-shot: since the process
started), and searches for the rest. A watchdog thread calls ti_stop shortly after the budget as a last resort.
"""
import ctypes
import os
import sys
import threading
import time

T_START = time.perf_counter()

MARKER = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
MODE = os.environ.get("TI_MODE", "oneshot")   # "oneshot" (default) or "long"
if MODE == "long":
    FIRST_LIMIT_MS, TURN_LIMIT_MS = 12000, 4000    # long-live: first move doubled, later moves 4 s
else:
    # One-shot: 6 s per move. The first move is deliberately NOT given the doubled 12 s seen in older
    # notes (unverified): a wrong guess there is a time-out loss. Set TI_FIRST_MS=12000 to use it.
    FIRST_LIMIT_MS = int(os.environ.get("TI_FIRST_MS", "6000"))
    TURN_LIMIT_MS = 6000
SAFETY_MS = 50              # always finish at least this much before the limit
STARTUP_MS = 120            # interpreter start + imports happen before T_START; charged to move 1
OUT_RESERVE_MS = 2          # printing / pipe latency
WATCHDOG_GRACE_MS = 12      # ti_stop is sent this long after the budget if the search is still running
SCALE = float(os.environ.get("TI_TIME_SCALE", "1.0"))   # tests only: shrink all limits
DEBUG = os.environ.get("TI_DEBUG") == "1"               # per-move timing detail on stderr

HERE = os.path.dirname(os.path.abspath(__file__))


def process_age_ms():
    """Wall-clock age of this process in ms (interpreter start-up counts against the move clock).
    Windows: GetProcessTimes creation time. Linux: /proc/self/stat start time vs /proc/uptime.
    Returns None if unavailable (a fixed allowance is charged instead)."""
    try:
        if os.name == "nt":
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(k32.GetCurrentProcess(), ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
                return None
            created = ((c.dwHighDateTime << 32) | c.dwLowDateTime) / 1e7 - 11644473600.0   # FILETIME -> unix seconds
            return max(0.0, (time.time() - created) * 1000.0)
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])                      # field 22 overall (starttime), 0-based 19 after the comm field
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0])
        return max(0.0, (up - start_ticks / os.sysconf("SC_CLK_TCK")) * 1000.0)
    except Exception:
        return None


PROC_AGE_MS = process_age_ms()


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
    # The clock started before Python did: charge the real process age when the OS can tell us.
    startup_ms = PROC_AGE_MS if PROC_AGE_MS is not None else STARTUP_MS
    t_origin = T_START - startup_ms / 1000.0
    if DEBUG:
        log("process age at start %.0f ms (%s)" % (startup_ms, "measured" if PROC_AGE_MS is not None else "assumed"))
    oneshot = MODE != "long"
    while True:
        limit_ms = (FIRST_LIMIT_MS if first else TURN_LIMIT_MS) * SCALE
        t0 = t_origin if (first or oneshot) else rd.t_req
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
            tail = "" if oneshot else "\n\n\n%s" % MARKER
            sys.stdout.write("%d %d %d %d\n%s%s\n" % (mv[0], mv[1], mv[2], mv[3], dbg, tail))
        else:
            # no legal move: the judge ends the game before asking; answer anyway so we never hang
            sys.stdout.write("-1 -1 -1 -1\nno-move%s\n" % ("" if oneshot else "\n\n\n" + MARKER))
        sys.stdout.flush()
        if oneshot:
            return
        first = False
        req = rd.ints(4)
        if req is None:
            break
        apply_move(req)


if __name__ == "__main__":
    main()
