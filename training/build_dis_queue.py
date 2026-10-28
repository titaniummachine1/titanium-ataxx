"""Build a current-main relabel queue ranked by static/search disagreement.

The stale 5k scores in labels.npz are in packed-row order. prio_rows.npy and
dis_rows.npy contain global dataset row IDs, so every score lookup must pass
through labels.npz['rows'] (packed index -> global row).

Only complete chunks with a matching exe and NNUE hash are excluded. The old
out_dis queue is mapped through its saved dis_rows.npy before this script
replaces that queue, so already-correct labels remain excluded after resorting.
"""
import atexit
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = r"C:\gitProjects\titanium-ataxx"
MAIN = r"C:\gitProjects\main-ataxx"
D = os.path.join(ROOT, "data", "nnue", "relabel")
PRIO_BIN = os.path.join(D, "prio_positions.bin")
PRIO_ROWS = os.path.join(D, "prio_rows.npy")  # raw int64, not an .npy container
PACK_ROWS = os.path.join(D, "rows.npy")
LABELS = os.path.join(D, "labels.npz")
EXE = os.path.join(MAIN, "target", "release", "titanium-cli.exe")
NET = os.path.join(MAIN, "data", "nnue", "nnue_main.tnn")
REC = np.dtype([("o0", "<u8"), ("o1", "<u8"), ("blk", "<u8"), ("stm", "u1")])
CHUNK = 5000
BATCH = 100_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\r\n").split("=", 1)
                    out[k] = v
    except OSError:
        pass
    return out


def board_text(recs):
    out = []
    for r in recs:
        o0, o1, blk, stm = int(r["o0"]), int(r["o1"]), int(r["blk"]), int(r["stm"])
        chars = []
        for i in range(49):
            bit = 1 << i
            chars.append("x" if o0 & bit else "o" if o1 & bit else "#" if blk & bit else ".")
        out.append("".join(chars) + (" w" if stm else " b"))
    return "\n".join(out) + "\n"


def completed_chunk(path, expected):
    """Return True only for a complete, parseable label chunk of expected rows."""
    n = 0
    try:
        with open(path, encoding="ascii") as f:
            for line in f:
                p = line.split()
                if len(p) < 3:
                    return False
                int(p[0]); int(p[1]); int(p[2])
                n += 1
                if n > expected:
                    return False
    except (OSError, UnicodeError, ValueError):
        return False
    return n == expected


def mark_existing(done_rows, directory, row_map, exe_hash, net_hash, label):
    """Mark global rows represented by valid chunks from this exact engine/net."""
    manifest = read_manifest(os.path.join(directory, "MANIFEST.txt"))
    if manifest.get("exe_sha", "").lower() != exe_hash[:16].lower() or \
       manifest.get("net_sha", "").lower() != net_hash[:16].lower():
        print(f"ignoring {label}: manifest does not match exe/net", flush=True)
        return 0
    count = 0
    for path in glob.glob(os.path.join(directory, "c_*.txt")):
        m = re.fullmatch(r"c_(\d+)\.txt", os.path.basename(path))
        if not m:
            continue
        chunk_i = int(m.group(1))
        lo = chunk_i * CHUNK
        if lo >= len(row_map):
            continue
        hi = min(lo + CHUNK, len(row_map))
        if not completed_chunk(path, hi - lo):
            continue
        rows = row_map[lo:hi]
        done_rows[rows] = True
        count += len(rows)
    print(f"excluding {count:,} verified {label} labels", flush=True)
    return count


def save_npy_atomic(path, array):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    lock_path = os.path.join(D, "dis_queue_rebuild.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit("queue rebuild lock exists; another rebuild may be active")
    os.close(fd)
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))

    for p in (PRIO_BIN, PRIO_ROWS, PACK_ROWS, LABELS, EXE, NET):
        if not os.path.isfile(p):
            sys.exit(f"required file missing: {p}")

    exe_hash = sha256(EXE)
    net_hash = sha256(NET)
    print(f"label engine exe={exe_hash[:16]} net={net_hash[:16]}", flush=True)

    recs = np.memmap(PRIO_BIN, dtype=REC, mode="r")
    n = len(recs)
    prio_rows = np.fromfile(PRIO_ROWS, dtype=np.int64)
    pack_rows = np.load(PACK_ROWS, mmap_mode="r")
    with np.load(LABELS) as z:
        label_rows = z["rows"]
        stale = z["score"]
    if not (len(prio_rows) == len(pack_rows) == len(label_rows) == len(stale) == n):
        sys.exit(f"row-count mismatch: prio={len(prio_rows)} pack={len(pack_rows)} "
                 f"labels.rows={len(label_rows)} labels.score={len(stale)} positions={n}")
    if not np.array_equal(pack_rows, label_rows):
        sys.exit("rows.npy differs from labels.npz['rows']; refusing an ambiguous score join")
    if label_rows.min() != 0 or label_rows.max() != n - 1:
        sys.exit("labels.npz['rows'] is not a full 0..N-1 dataset-row mapping")

    # label_rows[packed_index] == global_dataset_row; invert it before joining.
    row_to_pack = np.empty(n, dtype=np.int64)
    row_to_pack[label_rows] = np.arange(n, dtype=np.int64)
    # Prefer the durable mapping for old completed out_dis files. The queue
    # itself is about to be resorted, so its old dis_rows array is transient.
    old_dis_rows_path = os.path.join(D, "out_dis", "row_ids.bin")
    if not os.path.exists(old_dis_rows_path):
        old_dis_rows_path = os.path.join(D, "dis_rows.npy")
    old_dis_rows = np.fromfile(old_dis_rows_path, dtype=np.int64) if os.path.exists(old_dis_rows_path) else np.empty(0, np.int64)

    # Skip only completed outputs with a manifest matching this exact binary/net.
    already = np.zeros(n, dtype=np.bool_)
    skipped = 0
    exact_skip_path = os.path.join(D, "dis_samehash_rows.bin")
    have_exact_skip_set = os.path.exists(exact_skip_path)
    if have_exact_skip_set:
        exact_skip = np.fromfile(exact_skip_path, dtype=np.int64)
        if np.any(exact_skip < 0) or np.any(exact_skip >= n):
            sys.exit("dis_samehash_rows.bin contains invalid global row IDs")
        already[exact_skip] = True
        print(f"loaded {len(exact_skip):,} exact same-hash exclusions from prior queue", flush=True)
    skipped += mark_existing(already, os.path.join(D, "out_v2"), prio_rows,
                             exe_hash, net_hash, "out_v2/prio-order")
    if len(old_dis_rows) and not have_exact_skip_set:
        skipped += mark_existing(already, os.path.join(D, "out_dis"), old_dis_rows,
                                 exe_hash, net_hash, "out_dis/old-disorder")
    elif not len(old_dis_rows) and not have_exact_skip_set:
        print("no old dis_rows mapping; out_dis cannot safely be excluded", flush=True)
    current_manifest_path = os.path.join(D, "dis_queue_manifest.json")
    if os.path.exists(current_manifest_path):
        with open(current_manifest_path, encoding="utf-8") as f:
            previous = json.load(f)
        previous_out = previous.get("output_dir")
        if previous_out:
            previous_map = np.fromfile(os.path.join(D, "dis_rows.npy"), dtype=np.int64)
            skipped += mark_existing(already, os.path.join(D, previous_out), previous_map,
                                     exe_hash, net_hash, "previous disagreement queue")

    eligible = np.flatnonzero(~already[prio_rows])
    if len(eligible) == 0:
        sys.exit("no unlabeled positions remain")
    skipped = int(already.sum())
    print(f"positions={n:,} same-hash skipped={skipped:,} eligible={len(eligible):,}", flush=True)

    # Join the stale 5k score by global row -> packed score index.
    stale_eligible = stale[row_to_pack[prio_rows[eligible]]].astype(np.float32, copy=False)
    del row_to_pack, stale, label_rows, pack_rows
    stat_path = os.path.join(D, "dis_static_eligible.npy")
    delta_path = os.path.join(D, "dis_delta_eligible.npy")
    stat_tmp = stat_path + ".tmp"
    delta_tmp = delta_path + ".tmp"
    stat = np.lib.format.open_memmap(stat_tmp, mode="w+", dtype=np.float32, shape=(len(eligible),))

    boards_path = os.path.join(D, "_dis_boards.txt")
    for lo in range(0, len(eligible), BATCH):
        hi = min(len(eligible), lo + BATCH)
        ids = eligible[lo:hi]
        batch_recs = recs[ids]
        with open(boards_path, "w", encoding="ascii", newline="\n") as f:
            f.write(board_text(batch_recs))
        r = subprocess.run([EXE, "nnevaldump", "--network", "bundled", "--boards", boards_path],
                           capture_output=True, text=True, cwd=MAIN)
        if r.returncode:
            sys.exit(f"nnevaldump failed at eligible row {lo}: {r.stderr[-500:]}")
        # nnevaldump prints u32 f32::to_bits() values. Parse as unsigned;
        # int32 parsing saturates values above INT32_MAX and creates NaNs.
        vals = np.fromstring(r.stdout, dtype=np.uint32, sep="\n")
        if len(vals) != hi - lo:
            sys.exit(f"count mismatch at eligible row {lo}: {len(vals)} vs {hi - lo}")
        stat[lo:hi] = vals.view(np.float32) * 400.0
        if lo % (BATCH * 4) == 0:
            print(f"static evaluated {hi:,}/{len(eligible):,}", flush=True)

    stat.flush()
    if not np.isfinite(stat).all():
        sys.exit("static eval contains non-finite values; queue not replaced")
    delta = np.lib.format.open_memmap(delta_tmp, mode="w+", dtype=np.float32, shape=(len(eligible),))
    np.subtract(stat, stale_eligible, out=delta)
    np.abs(delta, out=delta)
    delta.flush()

    # Strictly rank by the requested disagreement. Ties prioritize more empties,
    # then preserve the prior prio order. Do not special-case mate scores here.
    empties = np.empty(len(eligible), dtype=np.uint8)
    for lo in range(0, len(eligible), BATCH):
        hi = min(len(eligible), lo + BATCH)
        batch_recs = recs[eligible[lo:hi]]
        occ = batch_recs["o0"] | batch_recs["o1"] | batch_recs["blk"]
        empties[lo:hi] = np.fromiter(
            (49 - int(x).bit_count() for x in occ), dtype=np.uint8, count=hi - lo)
    order = np.lexsort((eligible, -empties.astype(np.int16), -delta))
    head = order
    tail = np.empty(0, dtype=np.int64)
    queue_rows = prio_rows[eligible[order]]

    # Replace the queue only after every label and score has been validated.
    pos_tmp = os.path.join(D, "dis_positions.bin.tmp")
    rows_tmp = os.path.join(D, "dis_rows.npy.tmp")
    with open(pos_tmp, "wb") as fp, open(rows_tmp, "wb") as fr:
        for lo in range(0, len(order), BATCH):
            hi = min(len(order), lo + BATCH)
            ix = eligible[order[lo:hi]]
            recs[ix].tofile(fp)
            queue_rows[lo:hi].astype(np.int64, copy=False).tofile(fr)
        fp.flush(); os.fsync(fp.fileno())
        fr.flush(); os.fsync(fr.fileno())
    os.replace(pos_tmp, os.path.join(D, "dis_positions.bin"))
    os.replace(rows_tmp, os.path.join(D, "dis_rows.npy"))
    top_delta = float(delta[order[0]])
    delta_queue_tmp = os.path.join(D, "dis_delta_queue.npy.tmp")
    delta_queue = np.lib.format.open_memmap(
        delta_queue_tmp, mode="w+", dtype=np.float32, shape=(len(order),))
    for lo in range(0, len(order), BATCH):
        hi = min(len(order), lo + BATCH)
        delta_queue[lo:hi] = delta[order[lo:hi]]
    delta_queue.flush()
    delta_queue._mmap.close()
    del delta_queue
    stat.flush(); stat._mmap.close()
    delta.flush(); delta._mmap.close()
    del stat, delta
    os.replace(stat_tmp, stat_path)
    os.replace(delta_tmp, delta_path)
    os.replace(delta_queue_tmp, os.path.join(D, "dis_delta_queue.npy"))
    eligible_rows_path = os.path.join(D, "dis_eligible_rows.bin")
    eligible_rows_tmp = eligible_rows_path + ".tmp"
    prio_rows[eligible].astype(np.int64, copy=False).tofile(eligible_rows_tmp)
    os.replace(eligible_rows_tmp, eligible_rows_path)

    commit = subprocess.run(["git", "-C", MAIN, "log", "-1", "--format=%H"],
                            capture_output=True, text=True, check=True).stdout.strip()
    queue_hash = sha256(os.path.join(D, "dis_positions.bin"))
    manifest = {
        "schema": 2,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine_commit": commit,
        "exe_sha256": exe_hash,
        "net_sha256": net_hash,
        "stale_labels_sha256": sha256(LABELS),
        "stale_labels_order": "labels.npz rows maps packed score index to global dataset row",
        "ranking": "abs(current_static_cp - stale_5k_cp), descending; higher empties first on ties",
        "positions_total": int(n),
        "same_hash_rows_skipped": int(skipped),
        "eligible_rows": int(len(eligible)),
        "queue_rows": int(len(order)),
        "head_rows": int(len(head)),
        "mate_tail_rows": 0,
        "same_hash_sources": ["out_v2", "out_dis"],
        "same_hash_exe_prefix": exe_hash[:16],
        "same_hash_net_prefix": net_hash[:16],
        "static_values_file": os.path.basename(stat_path),
        "delta_values_file": os.path.basename(delta_path),
        "queue_delta_file": "dis_delta_queue.npy",
        "eligible_rows_file": os.path.basename(eligible_rows_path),
        "queue_sha256": queue_hash,
        "output_dir": f"out_dis_remaining_{exe_hash[:8]}_{net_hash[:8]}_{queue_hash[:8]}",
    }
    manifest_tmp = os.path.join(D, "dis_queue_manifest.json.tmp")
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(manifest_tmp, os.path.join(D, "dis_queue_manifest.json"))
    legacy_delta = os.path.join(D, "dis_delta_prio.npy")
    if os.path.exists(legacy_delta) and not os.path.exists(legacy_delta + ".pre-fix"):
        os.replace(legacy_delta, legacy_delta + ".pre-fix")
    print(f"queue rebuilt: {len(order):,} rows; strict disagreement order", flush=True)
    print(f"top disagreement={top_delta:.1f} cp; queue manifest written", flush=True)


if __name__ == "__main__":
    main()
