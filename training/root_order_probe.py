"""Would ordering the ROOT moves by a policy net help Titanium? READ-ONLY measurement, no engine change.

Truth = Titanium main S57 best move at a deep node budget (default 1,000,000). For each position we compare it with
  * Titanium's own pick at small budgets (1k, 5k nodes) = what its early iterations would search first
  * Ataxx_CPP's policy top move (one net forward pass, engine-ctl 'policy')
Every position: uainewgame first (clean TT/killers/history). Clones compared by destination, jumps by (from, to).
If the policy's top move equals the deep best move more often than Titanium's own shallow pick does, root ordering
by the policy can help short time controls; if not, it cannot.
Usage: python training/root_order_probe.py [n_positions] [deep_nodes]
"""
import sys

import numpy as np

sys.path.insert(0, "C:/gitProjects/titanium-ataxx/training")
import ax_vs_ti as A  # noqa: E402

FULL = (1 << 49) - 1


def canon(tok):
    """'e4' (clone) or 'a1c3' -> ('c', to) / ('j', from, to); a 4-char move with distance <= 1 is a clone"""
    if tok is None:
        return None
    if len(tok) == 2:
        return ("c", tok)
    f, t = tok[:2], tok[2:4]
    d = max(abs(ord(f[0]) - ord(t[0])), abs(int(f[1]) - int(t[1])))
    return ("c", t) if d <= 1 else ("j", f, t)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    deep = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    z = np.load("C:/gitProjects/titanium-ataxx/data/nnue/final/final_val_v3.npz")
    o0, o1, st, blk = z["occ0"], z["occ1"], z["stm"], z["blk"]
    emp = 49 - np.array([bin(int(a | b)).count("1") for a, b in zip(o0[:200000], o1[:200000])])
    ok = np.nonzero((emp >= 10) & (emp <= 36) & (blk[:200000] == 0))[0]
    rng = np.random.default_rng(8)
    idx = rng.choice(ok, n, replace=False)
    ti = A.Ti("go nodes %d" % deep)
    cpp = A.CppSide()
    res = []
    for k, j in enumerate(idx):
        state = (int(o0[j]), int(o1[j]), 0, int(st[j]))
        picks = {}
        for name, go in (("deep", "go nodes %d" % deep), ("ti1k", "go nodes 1000"), ("ti5k", "go nodes 5000"), ("ti50k", "go nodes 50000")):
            ti.send("uainewgame")
            ti.go = go
            picks[name] = canon(ti.move(state))
        cpp.set_state(state)
        picks["cpp_policy"] = canon(cpp.go_cmd("policy"))
        res.append((int(emp[j]), picks))
        if (k + 1) % 50 == 0:
            print("positions %d/%d" % (k + 1, n), flush=True)
    ti.close()
    cpp.close()
    print("\nagreement with Titanium's DEEP best move (%d nodes), %d positions, 10-36 empties:" % (deep, len(res)))
    for name in ("ti1k", "ti5k", "ti50k", "cpp_policy"):
        a = np.mean([p["deep"] == p[name] for _, p in res])
        print("   %-12s %.3f" % (name, a))
    a = np.mean([p["cpp_policy"] == p["ti5k"] for _, p in res])
    print("   cpp_policy == ti5k  %.3f" % a)
    both_wrong = np.mean([(p["ti1k"] != p["deep"]) and (p["cpp_policy"] == p["deep"]) for _, p in res])
    print("   positions where Titanium@1k misses the deep move but the policy has it: %.3f" % both_wrong)
    both_wrong5 = np.mean([(p["ti5k"] != p["deep"]) and (p["cpp_policy"] == p["deep"]) for _, p in res])
    print("   positions where Titanium@5k misses the deep move but the policy has it: %.3f" % both_wrong5)
    for lo, hi in ((10, 18), (19, 27), (28, 36)):
        sub = [p for e, p in res if lo <= e <= hi]
        if sub:
            print("   empties %d-%d (n=%d): ti1k %.2f ti5k %.2f ti50k %.2f policy %.2f" % (
                lo, hi, len(sub), np.mean([p["deep"] == p["ti1k"] for p in sub]), np.mean([p["deep"] == p["ti5k"] for p in sub]),
                np.mean([p["deep"] == p["ti50k"] for p in sub]), np.mean([p["deep"] == p["cpp_policy"] for p in sub])))


if __name__ == "__main__":
    main()
