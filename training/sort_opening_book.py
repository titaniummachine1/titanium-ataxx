"""Sort Ataxx opening books from best/strongest/most solid to worst/most lopsided.

Uses champion NNUE static eval + material/structure metrics:
1. Balance: |Eval_STM| -> 0 cp (both sides have equal fighting chance)
2. Material Parity: |diff| = |nx - no| <= 1
3. Phase Validity: 4 <= stones <= 18 (true opening positions, not degenerate starts/endgames)
4. Score: Penalty = |Eval_cp| + 35.0 * diff + stone_penalty

Positions sorted with Penalty ascending: Rank 0 = purest balanced grandmaster openings.
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import time

ROOT = "C:/gitProjects/titanium-ataxx"
MAIN = "C:/gitProjects/main-ataxx"
DEFAULT_EXE = MAIN + "/target/release/titanium-cli.exe"
DEFAULT_TNN = ROOT + "/data/nnue/w512_d4_v7_ep7_disagree.tnn"
DEFAULT_BOOK = ROOT + "/data/nnue/book50k.txt"


def main():
    parser = argparse.ArgumentParser(description="Sort opening book by solidity/balance")
    parser.add_argument("--book", default=DEFAULT_BOOK, help="Path to input opening book")
    parser.add_argument("--out", default=None, help="Path to output sorted opening book (default: same as --book)")
    parser.add_argument("--nnue", default=DEFAULT_TNN, help="Path to .tnn network for evaluation")
    parser.add_argument("--exe", default=DEFAULT_EXE, help="Path to titanium-cli.exe")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.book

    assert os.path.exists(args.book), f"Book missing: {args.book}"
    assert os.path.exists(args.nnue), f"NNUE missing: {args.nnue}"
    assert os.path.exists(args.exe), f"Exe missing: {args.exe}"

    print(f"Loading {args.book}...")
    with open(args.book, "r", encoding="utf-8") as f:
        fens = [line.strip() for line in f if line.strip()]
    total_fens = len(fens)
    print(f"Loaded {total_fens:,} openings.")

    # If overwriting, create a backup
    if os.path.abspath(args.book) == os.path.abspath(args.out):
        backup = args.book.replace(".txt", "_unsorted_backup.txt")
        if not os.path.exists(backup):
            print(f"Creating backup at {backup}...")
            shutil.copyfile(args.book, backup)

    # 1. Run titanium-cli nnevaldump
    print(f"Evaluating {total_fens:,} openings via {os.path.basename(args.exe)} with {os.path.basename(args.nnue)}...")
    t0 = time.time()
    res = subprocess.run(
        [args.exe, "nnevaldump", "--nnue", args.nnue, "--boards", args.book],
        capture_output=True,
        text=True,
        check=True
    )
    dt = time.time() - t0
    raw_lines = res.stdout.splitlines()
    assert len(raw_lines) == total_fens, f"Eval count mismatch: got {len(raw_lines)}, expected {total_fens}"
    print(f"Computed {total_fens:,} evaluations in {dt:.2f}s ({total_fens / dt:.0f} evals/sec).")

    # 2. Score openings
    print("Scoring openings by solidity and balance...")
    scored = []
    for fen, raw_u in zip(fens, raw_lines):
        u = int(raw_u.strip())
        logit = struct.unpack(">f", struct.pack(">I", u))[0]
        eval_cp = logit * 400.0
        abs_eval = abs(eval_cp)

        parts = fen.split()
        board_str = parts[0]
        turn = parts[1] if len(parts) > 1 else "b"

        nx = board_str.count("x")
        no = board_str.count("o")
        stones = nx + no
        diff = abs(nx - no)

        # Composite Penalty Score: lower is better/more solid
        penalty = abs_eval + diff * 35.0
        if stones < 4:
            penalty += 500.0
        elif stones > 18:
            penalty += (stones - 18) * 40.0

        scored.append((penalty, abs_eval, eval_cp, diff, stones, fen))

    # 3. Sort ascending by penalty
    scored.sort(key=lambda x: x[0])

    print("\n" + "=" * 70)
    print("TOP 10 STRONGEST & MOST BALANCED OPENINGS (Rank 0..9):")
    for i, item in enumerate(scored[:10]):
        print(f"  #{i:02d}: penalty={item[0]:5.1f} | eval={item[2]:+6.1f} cp | diff={item[3]} | stones={item[4]:2d} | {item[5]}")

    print("\nBOTTOM 5 MOST LOPSIDED OPENINGS (Tail):")
    for i, item in enumerate(scored[-5:]):
        print(f"  #{-5+i}: penalty={item[0]:5.1f} | eval={item[2]:+6.1f} cp | diff={item[3]} | stones={item[4]:2d} | {item[5]}")

    # 4. Decile statistics
    print("\n" + "=" * 70)
    print("DECILE PROFILE OF SORTED BOOK:")
    step = total_fens // 10
    for d in range(10):
        start = d * step
        end = (d + 1) * step if d < 9 else total_fens
        sub = scored[start:end]
        avg_eval = sum(x[1] for x in sub) / len(sub)
        max_eval = max(x[1] for x in sub)
        avg_diff = sum(x[3] for x in sub) / len(sub)
        print(f"  Decile {d+1:2d} (Games {start:5d}..{end:5d}): avg |eval| = {avg_eval:5.1f} cp (max {max_eval:5.1f} cp), avg diff = {avg_diff:.2f}")

    # 5. Write out
    print(f"\nWriting sorted opening book to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        for item in scored:
            f.write(item[5] + "\n")

    print(f"Done! {total_fens:,} openings successfully sorted from strongest to weakest.")


if __name__ == "__main__":
    main()
