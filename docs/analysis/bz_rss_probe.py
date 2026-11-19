# D606: peak memory of the Botzone library during real 3.5 s searches (self-play, both sides by the engine).
# Run inside Docker python:3.6 (Botzone-like glibc): python bz_rss_probe.py /path/titanium_v2.so MOVES
# Prints per move: move, nodes, VmRSS and VmHWM (peak) in MB. Not a tuning or training script (A7).
import ctypes, sys

def mem():
    d = {}
    for l in open("/proc/self/status"):
        if l.startswith(("VmRSS", "VmHWM")):
            k, v = l.split(":")
            d[k] = int(v.split()[0]) // 1024
    return d

class Stats(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_uint64 * 32)]

lib = ctypes.CDLL(sys.argv[1])
moves = int(sys.argv[2]) if len(sys.argv) > 2 else 12
lib.ti_new.restype = ctypes.c_void_p
lib.ti_new.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.ti_new_game.argtypes = [ctypes.c_void_p]
lib.ti_apply.restype = ctypes.c_int
lib.ti_apply.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
lib.ti_search_with_limits.restype = ctypes.c_int
lib.ti_search_with_limits.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint8,
                                      ctypes.POINTER(ctypes.c_int)]
print("load", mem())
e = lib.ti_new(None, 22)
lib.ti_new_game(e)
print("new", mem())
out = (ctypes.c_int * 4)()
for i in range(moves):
    r = lib.ti_search_with_limits(e, 3500, 0, 255, out)
    if r != 1:
        print("end", r)
        break
    lib.ti_apply(e, out[0], out[1], out[2], out[3])
    print("move %d %s" % (i + 1, list(out)), mem())
sys.stdout.flush()
