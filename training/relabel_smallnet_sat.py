import os
import subprocess
import time
import numpy as np

D = "C:/gitProjects/titanium-ataxx/data/nnue/smallnet"
EXE = "C:/gitProjects/main-ataxx/target/release/titanium-cli.exe"
TNN = "C:/gitProjects/titanium-ataxx/data/nnue/avx2_w256_saturated.tnn"


def relabel(name, chunk_size=8_000_000):
    src = f"{D}/sn_{name}.npz"
    dst = f"{D}/sn_{name}_sat.npz"
    print(f"Loading {src}...", flush=True)
    z = np.load(src)
    own = z["own"]
    opp = z["opp"]
    blk = z["blk"]
    n = len(own)
    print(f"{name}: {n} rows", flush=True)

    all_scores = np.empty(n, dtype=np.int32)
    tmp_in = f"{D}/tmp_{name}_boards.bin"
    tmp_out = f"{D}/tmp_{name}_scores.bin"

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        c_own = own[i:end]
        c_opp = opp[i:end]
        c_blk = blk[i:end]
        packed = np.column_stack([c_own, c_opp, c_blk]).astype("<u8").tobytes()
        with open(tmp_in, "wb") as f:
            f.write(packed)

        cmd = [EXE, "snevalbatch", "--network", TNN, "--in", tmp_in, "--out", tmp_out, "--threads", "10"]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr)
            raise RuntimeError(f"snevalbatch failed on chunk {i}..{end}")
        dt = time.time() - t0
        scores = np.fromfile(tmp_out, dtype=np.int32)
        assert len(scores) == end - i
        all_scores[i:end] = scores
        print(f"Chunk {i}..{end} ({end-i} rows) in {dt:.1f}s ({(end-i)/max(0.1, dt):.0f} pos/s)", flush=True)

    if os.path.exists(tmp_in):
        os.remove(tmp_in)
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    print(f"Saving {dst}...", flush=True)
    np.savez(
        dst,
        own=own,
        opp=opp,
        blk=blk,
        nn=all_scores,
        hand=z["hand"],
        pst=z["pst"],
        alpha=z["alpha"],
        nd0=z["nd0"],
        cached=z["cached"],
    )
    print(f"Done {dst}!", flush=True)


if __name__ == "__main__":
    t_start = time.time()
    relabel("eval")
    relabel("train")
    print(f"All relabeling completed in {time.time() - t_start:.1f}s", flush=True)
