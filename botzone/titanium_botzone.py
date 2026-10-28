"""Titanium Ataxx -- Botzone Python (Simple IO) driver, hardened (exp/botzone-robust).

Default mode: ONE-SHOT (a fresh process per move, the whole game history is sent every time; 6 s per move, 12 s on
this bot's first turn). TI_MODE=long = the long-live variant (>>>BOTZONE_REQUEST_KEEP_RUNNING<<<).
Layout on Botzone (storage files next to the bot):
    data/titanium_v2.so          engine library (Linux, glibc >= 2.14)
    The main big NNUE is embedded in titanium_v2.so; TI_NET may override it.
Local testing overrides (environment): TI_LIB, TI_NET, TI_TT_BITS, TI_TIME_SCALE, TI_DEBUG, TI_FAULT (tests only: lib_fail2, engine_fail1, engine_fail, search_fail1, search_fail, no_lib, age_huge).

Why (2026-10-26, match 6ade99b3...: RE after 101 ms, exit status 120, no stdout, no stderr, 144 MB): the engine had
loaded, then the process died on a broken standard stream (CPython exits 120 when it cannot flush them; the old driver
wrote "net ..." to stderr right after loading the net). So:
  * no stream problem may cost the move: stderr writes are guarded, diagnostics go into the DEBUG line of stdout
    (line 2, shown in the Botzone match log), the answer is written with os.write (retried) and the process leaves with
    os._exit(0) so interpreter shutdown cannot fail after the move was sent;
  * the engine is RESUSCITATED at FULL strength only: library load and engine creation (big NNUE, normal hash table)
    are retried, a failed search rebuilds the engine, replays the game and searches again with the time left (same
    engine, same net, less time); every recovery and its cost in ms is listed in the debug line ("ev ...");
    NEVER a weaker net, the classical eval or a smaller hash table (user order 2026-10-26);
  * NO backup move generator (user order 2026-10-26): if the engine cannot answer after the retries the bot answers
    "-1 -1 -1 -1" with ENGINE_FAILED and the reason in the debug line, so a failure is visible as a failure
    (INVALIDMOVE) and is never mistaken for a weak move;
  * a measured process age (clock start) that is impossible (< 0 or > 3 s) is ignored.
A native crash inside the library (segfault) cannot be caught in-process; a separate engine process was NOT used
because Botzone's sandbox may forbid or kill on process creation, which cannot be tested here.
"""
import os
import sys
import time

T_START = time.perf_counter()

MARKER = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
MODE = os.environ.get("TI_MODE", "long")
if MODE == "long":
    FIRST_LIMIT_MS, TURN_LIMIT_MS = 12000, 4000
else:
    FIRST_LIMIT_MS = int(os.environ.get("TI_FIRST_MS", "12000"))
    TURN_LIMIT_MS = 6000
SAFETY_MS = 100
STARTUP_MS = 120            # assumed interpreter start-up when the process age cannot be measured
MAX_SANE_AGE_MS = 3000      # a measured process age above this (or below 0) is a measurement error
OUT_RESERVE_MS = 5
WATCHDOG_GRACE_MS = 12
RETRY_RESERVE_MS = 400      # a resuscitated search needs at least this much time left
SCALE = float(os.environ.get("TI_TIME_SCALE", "1.0"))
DEBUG = os.environ.get("TI_DEBUG") == "1"
BOTZONE_MAX_DEPTH = 255
FAULT = os.environ.get("TI_FAULT", "")   # tests only, see the docstring
HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = []                 # recoveries of this turn -> debug line


def now_ms():
    return (time.perf_counter() - T_START) * 1000.0


def event(what, t0=None):
    EVENTS.append("%s%s" % (what, "" if t0 is None else ":%.0fms" % (now_ms() - t0)))


def log(msg):
    """stderr is only a convenience: a closed or broken stderr must never cost the move."""
    try:
        sys.stderr.write("[titanium] %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass


def write_out(text):
    """The answer via raw os.write (no Python buffering), retrying partial / interrupted writes."""
    data = text.encode("ascii", "replace")
    for _ in range(50):
        try:
            n = os.write(1, data)
            data = data[n:]
            if not data:
                return True
        except InterruptedError:
            continue
        except Exception:
            time.sleep(0.002)
    return False


def process_age_ms():
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(k32.GetCurrentProcess(), ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
                return None
            created = ((c.dwHighDateTime << 32) | c.dwLowDateTime) / 1e7 - 11644473600.0
            return (time.time() - created) * 1000.0
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0])
        return (up - start_ticks / os.sysconf("SC_CLK_TCK")) * 1000.0
    except Exception:
        return None


class Engine:
    """ctypes wrapper with resuscitation (load / create / replay retried)."""

    def __init__(self):
        self.lib = None
        self.e = None
        self.net = "-"
        self.injected = 0

    def load(self):
        import ctypes
        self.ct = ctypes
        cands = [os.environ.get("TI_LIB"), os.path.join("data", "titanium_v2.so"), os.path.join(HERE, "data", "titanium_v2.so"),
                 os.path.join(HERE, "titanium_ffi.dll"), os.path.join(HERE, "libtitanium_ffi.so")]
        if FAULT == "no_lib":
            cands = []
        path = next((p for p in cands if p and os.path.exists(p)), None)
        if path is None:
            raise RuntimeError("library_not_found")
        for attempt in range(3):
            t0 = now_ms()
            try:
                if FAULT == "lib_fail2" and attempt < 2:
                    raise OSError("injected")
                lib = ctypes.CDLL(os.path.abspath(path))
                lib.ti_abi_version.restype = ctypes.c_uint32
                lib.ti_new.restype = ctypes.c_void_p
                lib.ti_new.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
                lib.ti_free.argtypes = [ctypes.c_void_p]
                lib.ti_new_game.argtypes = [ctypes.c_void_p]
                lib.ti_apply.restype = ctypes.c_int
                lib.ti_apply.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
                lib.ti_stop.argtypes = [ctypes.c_void_p]
                lib.ti_search.restype = ctypes.c_int
                lib.ti_search.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
                lib.ti_search_with_limits.restype = ctypes.c_int
                lib.ti_search_with_limits.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                                      ctypes.c_uint8, ctypes.POINTER(ctypes.c_int)]
                lib.ti_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                if lib.ti_abi_version() != 2:
                    raise RuntimeError("abi_%d" % lib.ti_abi_version())
                self.lib = lib
                if attempt:
                    event("lib_ok_after_%d_retries" % attempt, t0)
                return
            except Exception as ex:
                event("lib_fail%d(%s)" % (attempt + 1, type(ex).__name__), t0)
                time.sleep(0.02)
        raise RuntimeError("library_load_failed_3x")

    def create(self):
        """Full strength or nothing (user order 2026-10-26): the big NNUE with the normal hash table, retried; NEVER a
        weaker net, the classical eval or a smaller hash table just to stay alive - that would hide a failure as a
        weak game. If it cannot be created, the turn fails visibly (ENGINE_FAILED)."""
        tt_bits = int(os.environ.get("TI_TT_BITS", "22"))
        net = os.environ.get("TI_NET")
        if net and not os.path.isfile(net):
            raise RuntimeError("custom_network_not_found")
        for attempt in range(3):
            t0 = now_ms()
            inject = (FAULT == "engine_fail") or (FAULT == "engine_fail1" and attempt == 0)
            net_arg = os.path.abspath(net).encode("utf-8") if net else None
            e = None if inject else self.lib.ti_new(net_arg, tt_bits)
            if e:
                label = "custom:%s" % os.path.basename(net) if net else "bundled-main"
                self.e, self.net = e, "%s/%d" % (label, tt_bits)
                if attempt:
                    event("engine_ok_after_%d_retries" % attempt, t0)
                return
            event("engine_create_fail%d" % (attempt + 1), t0)
            time.sleep(0.02)
        raise RuntimeError("engine_create_failed_3x")

    def setup(self, history):
        self.lib.ti_new_game(self.e)
        for m in history:
            if self.lib.ti_apply(self.e, *m) != 1:
                raise RuntimeError("history_move_rejected_%d_%d_%d_%d" % m)

    def search(self, budget):
        import threading
        out = (self.ct.c_int * 4)()

        def watchdog():
            try:
                self.lib.ti_stop(self.e)
            except Exception:
                pass
        timer = threading.Timer((budget + WATCHDOG_GRACE_MS) / 1000.0, watchdog)
        timer.daemon = True
        timer.start()
        try:
            if FAULT == "search_fail" or (FAULT == "search_fail1" and self.injected == 0):
                self.injected += 1
                raise RuntimeError("injected")
            res = self.lib.ti_search_with_limits(self.e, budget, 0, BOTZONE_MAX_DEPTH, out)
        finally:
            timer.cancel()
        st = Stats()
        try:
            self.lib.ti_stats(self.e, self.ct.byref(st))
        except Exception:
            pass
        if res != 1:
            raise RuntimeError("ti_search_returned_%d" % res)
        return (out[0], out[1], out[2], out[3]), st


def _stats_struct():
    import ctypes

    class S(ctypes.Structure):
        _fields_ = [("nodes", ctypes.c_uint64), ("elapsed_us", ctypes.c_uint64), ("score", ctypes.c_int32),
                    ("depth", ctypes.c_uint32), ("stopped_by", ctypes.c_uint32), ("repetition_override", ctypes.c_uint32)]
    return S


Stats = _stats_struct()


class Reader:
    def __init__(self):
        self.buf = []
        self.t_req = T_START

    def ints(self, n):
        got, first = [], True
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


def play_turn(eng, history, limit_ms, t0):
    """The engine's move, resuscitating it up to twice. Returns (move or None, debug line)."""
    deadline = limit_ms - SAFETY_MS * min(1.0, SCALE * 4) - OUT_RESERVE_MS

    def left():
        return deadline - (time.perf_counter() - t0) * 1000.0
    for attempt in range(3):
        if attempt and left() < RETRY_RESERVE_MS * min(1.0, SCALE * 4):
            event("no_time_for_retry")
            break
        ta = now_ms()
        try:
            if eng.lib is None:
                eng.load()
            if eng.e is None:
                eng.create()
                eng.setup(history)
            mv, st = eng.search(max(5, int(left())))
            if attempt:
                event("resuscitated_on_try%d" % (attempt + 1), ta)
            return mv, "d%d n%d t%.0f s%d%s %s" % (st.depth, st.nodes, (time.perf_counter() - t0) * 1000.0, st.score,
                                                  " rep" if st.repetition_override else "", eng.net)
        except Exception as ex:
            event("try%d_fail(%s:%s)" % (attempt + 1, type(ex).__name__, str(ex)[:48].replace(" ", "_")), ta)
            eng.e = None          # rebuild the engine and replay the game on the next try
    return None, "ENGINE_FAILED t%.0f" % ((time.perf_counter() - t0) * 1000.0)


def main():
    age = process_age_ms()
    if FAULT == "age_huge":
        age = 1e9
    if age is None or age < 0 or age > MAX_SANE_AGE_MS:
        if age is not None:
            event("age_ignored(%.0f)" % age)
        age = STARTUP_MS
    t_origin = T_START - age / 1000.0
    rd = Reader()
    head = rd.ints(1)
    if head is None:
        return
    turns = head[0]
    flat = rd.ints(4 * (2 * turns - 1))
    if flat is None:
        return
    history = [tuple(flat[i:i + 4]) for i in range(0, len(flat), 4) if flat[i] >= 0]
    eng = Engine()
    first = turns == 1
    oneshot = MODE != "long"
    while True:
        limit_ms = (FIRST_LIMIT_MS if first else TURN_LIMIT_MS) * SCALE
        t0 = t_origin if (first or oneshot) else rd.t_req
        mv, dbg = play_turn(eng, history, limit_ms, t0)
        if mv is not None:
            history.append(mv)
        line1 = "%d %d %d %d" % (mv if mv is not None else (-1, -1, -1, -1))
        if EVENTS:
            dbg += " | ev " + ",".join(EVENTS)
            del EVENTS[:]
        if DEBUG:
            log(dbg)
        write_out("%s\n%s%s\n" % (line1, dbg, "" if oneshot else "\n\n\n" + MARKER))
        if oneshot:
            return
        first = False
        req = rd.ints(4)
        if req is None:
            return
        history.append(tuple(req))
        try:
            if eng.e is not None and eng.lib.ti_apply(eng.e, *req) != 1:
                event("request_move_rejected")
                eng.e = None          # replay from history on the next turn
        except Exception:
            eng.e = None


if __name__ == "__main__":
    try:
        main()
    except BaseException as ex:          # never die silently: say why in the debug line (the move is then -1s)
        write_out("-1 -1 -1 -1\nFATAL %s:%s | ev %s\n" % (type(ex).__name__, str(ex)[:80].replace("\n", " "), ",".join(EVENTS)))
    # leave without interpreter shutdown: a broken stream at exit must not turn a sent move into a runtime error
    os._exit(0)
