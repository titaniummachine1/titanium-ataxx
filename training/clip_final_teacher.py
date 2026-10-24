import os

import numpy as np

FINAL = r"C:\gitProjects\titanium-ataxx\data\nnue\final"


def main():
    for split in ("train", "val", "test"):
        p = os.path.join(FINAL, "final_%s.npz" % split)
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        z.close()
        t = np.clip(d["teacher"], -1500.0, 1500.0).astype(np.float32)
        d["teacher"] = t
        d["sc"] = (t / 2000.0).astype(np.float32)
        tmp = p + ".tmp.npz"
        np.savez_compressed(tmp, **d)
        os.replace(tmp, p)
        print(split, len(t), float(t.min()), float(t.max()), os.path.getsize(p), flush=True)


if __name__ == "__main__":
    main()
