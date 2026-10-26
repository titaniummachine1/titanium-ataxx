"""Report of an exp/eval-probe counter dump (BACKLOG A1): where the big-net evals of a search go and what a two-sided
small-net bound could skip at leaves (A3) and RFP (A4). Layout = `pub mod probe` in exp/eval-probe search.rs.
Make a dump: EVAL_PROBE_OUT=dump.txt <evalprobe exe> label --nodes 5000 --tuple none --nnue <net> < roots.txt
Usage: python training/eval_probe_report.py dump.txt [more dumps]
"""
import sys

DB = ["d0", "d1", "d2", "d3", "d4", "d5+"]
SRC = ["root", "first move", "late move (zw)", "re-search", "null move"]
BK = ["<25", "25-50", "50-100", "100-200", "200-400", ">=400"]


def report(path):
    c = [0] * 320
    for line in open(path):
        i, v = line.split()
        c[int(i)] = int(v)
    big = c[154]
    print("\n== %s" % path)
    print("big-net evals %d, %.2f us each, %.1f%% of search time; small-net filter %d calls, %.2f us each, %.1f%%"
          % (big, c[152] / max(big, 1) / 1e3, 100 * c[152] / c[156], c[155], c[153] / max(c[155], 1) / 1e3,
             100 * c[153] / c[156]))
    ne_miss = sum(c[24:30])
    fc_miss = c[146] + c[147]
    print("big-net evals by site: node entry %.1f%%, futility child %.1f%% (pruned anyway %.1f%%, searched %.1f%%), "
          "no-move %d, guard %d" % (100 * ne_miss / big, 100 * fc_miss / big, 100 * c[146] / big, 100 * c[147] / big,
                                    c[150], c[151]))
    print("\nnode entry by depth:      " + "".join("%10s" % d for d in DB))
    for name, o in (("TT return (no eval)", 0), ("eval from parent", 8), ("cache hit", 16), ("BIG NET", 24),
                    ("-> RFP return", 32), ("-> NMP cutoff", 40), ("-> searched", 56),
                    ("miss -> RFP return", 64), ("miss -> NMP cutoff", 72), ("miss -> searched", 88)):
        print("%-26s" % name + "".join("%10d" % c[o + i] for i in range(6)))
    print("%-26s%10d" % ("depth-0 return", c[48]))
    print("%-26s%10d" % ("miss -> depth-0 return", c[80]))
    print("\nnode-entry big-net evals by caller:")
    for s in range(5):
        row = c[96 + s * 8: 96 + s * 8 + 6]
        if sum(row):
            print("  %-16s" % SRC[s] + "".join("%10d" % x for x in row) + "   = %.1f%% of all big-net evals"
                  % (100 * sum(row) / big))
    print("\nfutility step: entered %d, cached %d, small net pruned %d, passed %d; LMR-only prunes %d"
          % (c[141], c[142], c[143], c[144], c[140]))
    print("  big net after a pass: pruned anyway %d (%.1f%% of passes), searched %d" % (
        c[146], 100 * c[146] / max(c[144], 1), c[147]))
    print("  pruned-anyway gap (big eval - small-net bound): " + ", ".join(
        "%s %d" % (BK[i], c[230 + i]) for i in range(6)))
    d0 = c[80]
    print("\ndepth-0 big-net evals %d (%.1f%% of all): where the eval sits vs the window (A3 lazy leaf)" % (
        d0, 100 * d0 / big))
    for zw, nm in ((1, "zero window"), (0, "PV window")):
        above = c[160 + zw * 16: 166 + zw * 16]
        below = c[168 + zw * 16: 174 + zw * 16]
        tot = sum(above) + sum(below) + c[200 + zw]
        if not tot:
            continue
        print("  %s %d: inside %d (%.1f%%)" % (nm, tot, c[200 + zw], 100 * c[200 + zw] / tot))
        print("    >= beta by  " + "  ".join("%s %.1f%%" % (BK[i], 100 * above[i] / tot) for i in range(6)))
        print("    <= alpha by " + "  ".join("%s %.1f%%" % (BK[i], 100 * below[i] / tot) for i in range(6)))
        # skippable with a two-sided bound of half-width E: |outside distance| >= E
        for e, k in ((50, 2), (100, 3), (200, 4)):
            sk = sum(above[k:]) + sum(below[k:])
            print("    outside by >= %3d cp: %.1f%% of these evals = %.1f%% of all big-net evals"
                  % (e, 100 * sk / tot, 100 * sk / big))
    rf = c[210:216]
    rfire = c[218:224]
    tot = sum(rf) + sum(rfire)
    if tot:
        print("\nRFP-eligible depth 1-4 big-net evals %d (%.1f%% of all), fired %d (%.1f%%)"
              % (tot, 100 * tot / big, sum(rfire), 100 * sum(rfire) / tot))
        print("  fired by     " + "  ".join("%s %d" % (BK[i], rfire[i]) for i in range(6)))
        print("  missed by    " + "  ".join("%s %d" % (BK[i], rf[i]) for i in range(6)))
        for e, k in ((50, 2), (100, 3), (200, 4)):
            print("  fired by >= %3d cp (an upper bound with that error still fires): %.1f%% of all big-net evals"
                  % (e, 100 * sum(rfire[k:]) / big))


for p in sys.argv[1:]:
    report(p)
