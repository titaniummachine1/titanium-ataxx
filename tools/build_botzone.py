"""One command: build the Botzone bot from main and leave a folder that is ready to upload.

    python tools/build_botzone.py              # pull main (fast-forward), build, check, assemble, test in Linux
    python tools/build_botzone.py --no-pull    # build the current checkout as it is (must still be clean)
    python tools/build_botzone.py --full-time  # also play one real 12 s first move in the Linux test

Steps (each one stops the script on failure, nothing half-built is left as "ready"):
  1. git: refuse a dirty tree (a build must be reproducible from a commit); on main, `git pull --ff-only`.
  2. build the engine library for Linux with zig for glibc 2.14 (Botzone runs Ubuntu 16.04 = glibc 2.23; a
     library built on a modern system needs glibc 2.34 and crashes the bot on its first move).
  3. check the library: ELF x86-64, every ti_* function exported, newest glibc symbol <= 2.17.
  4. assemble botzone/upload/: the driver and data/titanium_v2.so (main's big NNUE is embedded in the library)
     and MANIFEST.txt (no fallback net: the driver plays at full strength or fails visibly) (commit, sha256 of every file, where each file goes).
  5. test (needs Docker, local only): the exact upload folder plays turn 1 as black, turn 1 as white, turn 2, with
     stderr closed and through every injected fault (TI_FAULT: library fails twice, engine creation fails, first search
     fails, impossible clock -> must still move AT FULL STRENGTH (NNUE, normal hash) and name the recovery; engine never
     created / search fails for good / library missing -> must
     fail VISIBLY with -1s + ENGINE_FAILED/FATAL),
     in a python:3.6 container with an older glibc, at a scaled clock; must load the NNUE and answer in time.
Uploading stays manual (your Botzone login never leaves your machine): see MANIFEST.txt.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "botzone", "upload")
TARGET = "x86_64-unknown-linux-gnu"
GLIBC_BUILD = "2.14"
GLIBC_MAX = (2, 17)
SO = os.path.join(ROOT, "target", TARGET, "release", "libtitanium_ffi.so")
EXPORTS = [b"ti_abi_version", b"ti_new", b"ti_free", b"ti_new_game", b"ti_apply", b"ti_legal_count", b"ti_search",
           b"ti_search_with_limits", b"ti_stop", b"ti_stats"]
ZIG_DIRS = ["C:/gitProjects/toolchains/zig-0.13.0/zig-windows-x86_64-0.13.0"]
FILES = {  # upload-folder path -> source path in the repo
    "titanium_botzone.py": "botzone/titanium_botzone.py",
}


def step(msg):
    print("\n== " + msg, flush=True)


def fail(msg):
    sys.exit("FAILED: " + msg)


def run(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, **kw)
    if r.returncode:
        fail("%s exited with %d" % (cmd[0], r.returncode))
    return r


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true", help="build the current checkout without pulling")
    ap.add_argument("--no-test", action="store_true", help="skip the Docker Linux test")
    ap.add_argument("--full-time", action="store_true", help="Linux test also plays one real 12 s first move")
    ap.add_argument("--cpu", default="x86-64-v3", choices=["x86-64-v3", "x86-64-v4"],
                    help="Target microarchitecture level (default x86-64-v3 AVX2 + runtime AVX-512 dispatch; x86-64-v4 = full AVX-512)")
    a = ap.parse_args()

    step("1/5 git")
    if git("status", "--porcelain", "--untracked-files=no"):
        fail("working tree has uncommitted changes; commit or stash them first")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if not a.no_pull:
        if branch != "main":
            fail("not on main (on %s); check out main or pass --no-pull" % branch)
        run(["git", "pull", "--ff-only"])
    commit = git("rev-parse", "--short", "HEAD")
    print("building %s @ %s" % (branch, commit))

    step("2/5 build the Linux library (zig, glibc %s)" % GLIBC_BUILD)
    env = dict(os.environ)
    if not shutil.which("zig", path=env.get("PATH")):
        for d in ZIG_DIRS:
            if os.path.exists(os.path.join(d, "zig.exe")) or os.path.exists(os.path.join(d, "zig")):
                env["PATH"] = d + os.pathsep + env.get("PATH", "")
                break
        else:
            fail("zig not found (install zig 0.13 and cargo-zigbuild: `cargo install cargo-zigbuild`)")
    env["CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS"] = "-C target-cpu=%s" % a.cpu
    run(["cargo", "zigbuild", "--release", "-p", "titanium-ffi", "--target", "%s.%s" % (TARGET, GLIBC_BUILD)], env=env)

    step("3/5 check the library")
    d = open(SO, "rb").read()
    if d[:4] != b"\x7fELF" or d[4] != 2 or int.from_bytes(d[18:20], "little") != 62:
        fail("%s is not an x86-64 ELF" % SO)
    missing = [e.decode() for e in EXPORTS if e not in d]
    if missing:
        fail("missing exports: %s" % missing)
    vers = [tuple(int(x) for x in v.decode().split(".")) for v in set(re.findall(rb"GLIBC_(2\.[0-9]+(?:\.[0-9]+)?)", d))]
    newest = max(vers) if vers else (0,)
    if newest > GLIBC_MAX:
        fail("library needs glibc %s; Botzone (Ubuntu 16.04) only has 2.23" % ".".join(map(str, newest)))
    print("ok: x86-64 ELF, %d exports, newest glibc symbol %s, %d bytes" % (len(EXPORTS), ".".join(map(str, newest)), len(d)))

    step("4/5 assemble %s" % OUT)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "data"))
    shutil.copy2(SO, os.path.join(OUT, "data", "titanium_v2.so"))
    for dst, src in FILES.items():
        s = os.path.join(ROOT, src)
        if not os.path.exists(s):
            fail("missing %s" % src)
        shutil.copy2(s, os.path.join(OUT, dst))
    if os.path.exists(os.path.join(OUT, "data", "nnue", "nnue_main.tnn")):
        fail("the bundled-NNUE package must not depend on an external network file")
    lines = ["Titanium Ataxx Botzone upload, built from %s @ %s" % (branch, commit), "",
             "Upload (Botzone -> avatar menu -> manage storage), then submit the bot:",
             "  data/titanium_v2.so       -> storage /titanium_v2.so",
             "Embedded main NNUE: data/nnue/nnue_main.tnn (sha256 %s)" % sha256(os.path.join(ROOT, "data", "nnue", "nnue_main.tnn")),
             "Botzone search limits: time budget with full u8 depth range (255); solved mates/terminals stop automatically.",
             "  titanium_botzone.py       -> new bot version: Python 3, simple IO, long-running OFF", "",
             "sha256:"]
    for root, _, fs in os.walk(OUT):
        for f in sorted(fs):
            p = os.path.join(root, f)
            lines.append("  %s  %10d  %s" % (sha256(p), os.path.getsize(p), os.path.relpath(p, OUT).replace(os.sep, "/")))
    open(os.path.join(OUT, "MANIFEST.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if a.no_test:
        print("\nREADY (Linux test skipped): %s" % OUT)
        return
    step("5/5 Linux test (Docker, python:3.6, older glibc)")
    if not shutil.which("docker"):
        fail("docker not found; rerun with --no-test to skip (not recommended)")
    # (name, stdin, env, shell redirect, expect): "move" = legal move whose debug line names the bundled network,
    # "move+ev:<x>" = legal move AND recovery event <x> in the debug line, "fail" = -1s + ENGINE_FAILED/FATAL (a VISIBLE failure)
    t1, t1w, t2 = "1\\n-1 -1 -1 -1\\n", "1\\n0 0 1 1\\n", "2\\n-1 -1 -1 -1\\n6 6 5 5\\n6 0 5 1\\n"
    cases = [("black turn 1", t1, "", "", "move"), ("white turn 1", t1w, "", "", "move"), ("turn 2", t2, "", "", "move"),
             ("stderr closed", t1, "", "2>&-", "move"),
             ("library fails twice", t1, "TI_FAULT=lib_fail2", "", "move+ev:lib_ok_after_2_retries"),
             ("engine creation fails once", t1, "TI_FAULT=engine_fail1", "", "move+ev:engine_ok_after_1_retries"),
             ("engine creation fails for good", t1, "TI_FAULT=engine_fail", "", "fail"),
             # real clock (scale 1): at 1/10 the 600 ms test turn is shorter than one engine (re)load on a busy machine
             ("first search fails", t2, "TI_FAULT=search_fail1 TI_TIME_SCALE=1", "", "move+ev:resuscitated_on_try2"),
             ("impossible process age", t1, "TI_FAULT=age_huge", "", "move+ev:age_ignored"),
             ("engine fails for good", t1, "TI_FAULT=search_fail", "", "fail"),
             ("library missing", t1, "TI_FAULT=no_lib", "", "fail")]
    if a.full_time:
        cases.append(("real 12 s first move", t1, "TI_TIME_SCALE=1", "", "move"))
    script = "ldd --version | head -1; python3 --version; "
    for name, inp, env, redir, _ in cases:
        script += "echo '== %s'; printf '%s' | env TI_TIME_SCALE=0.1 %s python3 titanium_botzone.py %s; echo \"== rc=$?\"; " % (name, inp, env, redir)
    r = subprocess.run(["docker", "run", "--rm", "-v", "%s:/bot" % OUT, "-w", "/bot", "python:3.6-slim-buster", "bash", "-c", script],
                       capture_output=True, text=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    print(r.stdout + r.stderr)
    blocks = re.split(r"^== (?!rc=)", r.stdout, flags=re.M)[1:]
    bad = []
    for (name, _, _, _, expect), blk in zip(cases, blocks):
        lines = [l for l in blk.splitlines()[1:] if not l.startswith("== rc=")]
        rc = re.search(r"== rc=(\d+)", blk)
        mv = lines[0] if lines else ""
        dbg = lines[1] if len(lines) > 1 else ""
        if rc is None or rc.group(1) != "0":
            bad.append("%s: exit code %s" % (name, rc.group(1) if rc else "?"))
        elif expect == "fail":
            if mv != "-1 -1 -1 -1" or not re.search(r"ENGINE_FAILED|FATAL", dbg):
                bad.append("%s: expected a VISIBLE failure, got %r / %r" % (name, mv, dbg))
        else:
            if re.fullmatch(r"[0-6] [0-6] [0-6] [0-6]", mv) is None or "bundled-main" not in dbg:
                bad.append("%s: expected a legal NNUE move, got %r / %r" % (name, mv, dbg))
            if expect.startswith("move+ev:") and expect[8:] not in dbg:
                bad.append("%s: recovery event %s missing in %r" % (name, expect[8:], dbg))
    if r.returncode or bad or len(blocks) < len(cases):
        fail("Linux test: " + ("; ".join(bad) if bad else "only %d of %d cases ran" % (len(blocks), len(cases))))
    print("ok: %d cases passed (normal turns, stderr closed, every recovery path, visible failures)" % len(cases))
    print("\nREADY TO UPLOAD: %s  (see MANIFEST.txt)" % OUT)


if __name__ == "__main__":
    main()
