"""One command: build the Botzone bot from main and leave a folder that is ready to upload.

    python tools/build_botzone.py              # pull main (fast-forward), build, check, assemble, test in Linux
    python tools/build_botzone.py --no-pull    # build the current checkout as it is (must still be clean)
    python tools/build_botzone.py --full-time  # also play one real 12 s first move in the Linux test

Steps (each one stops the script on failure, nothing half-built is left as "ready"):
  1. git: refuse a dirty tree (a build must be reproducible from a commit); on main, `git pull --ff-only`.
  2. build the engine library for Linux with zig for glibc 2.14 (Botzone runs Ubuntu 16.04 = glibc 2.23; a
     library built on a modern system needs glibc 2.34 and crashes the bot on its first move).
  3. check the library: ELF x86-64, every ti_* function exported, newest glibc symbol <= 2.17.
  4. assemble botzone/upload/: the driver, data/titanium_v2.so, data/nnue/nnue_main.tnn (main's default net),
     data/nnue/tupS57v2.tup (fallback) and MANIFEST.txt (commit, sha256 of every file, where each file goes).
  5. test (needs Docker, local only): the exact upload folder plays turn 1 as black, turn 1 as white and turn 2
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
           b"ti_stop", b"ti_stats"]
ZIG_DIRS = ["C:/gitProjects/toolchains/zig-0.13.0/zig-windows-x86_64-0.13.0"]
FILES = {  # upload-folder path -> source path in the repo
    "titanium_botzone.py": "botzone/titanium_botzone.py",
    "data/nnue/nnue_main.tnn": "data/nnue/nnue_main.tnn",
    "data/nnue/tupS57v2.tup": "data/nnue/tupS57v2.tup",
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
    os.makedirs(os.path.join(OUT, "data", "nnue"))
    shutil.copy2(SO, os.path.join(OUT, "data", "titanium_v2.so"))
    for dst, src in FILES.items():
        s = os.path.join(ROOT, src)
        if not os.path.exists(s):
            fail("missing %s" % src)
        shutil.copy2(s, os.path.join(OUT, dst))
    lines = ["Titanium Ataxx Botzone upload, built from %s @ %s" % (branch, commit), "",
             "Upload (Botzone -> avatar menu -> manage storage), then submit the bot:",
             "  data/titanium_v2.so       -> storage /titanium_v2.so",
             "  data/nnue/nnue_main.tnn   -> storage /nnue/nnue_main.tnn   (big NNUE, main's default eval)",
             "  data/nnue/tupS57v2.tup    -> storage /nnue/tupS57v2.tup    (fallback if the NNUE is missing)",
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
    turns = [("black turn 1", "1\\n-1 -1 -1 -1\\n"), ("white turn 1", "1\\n0 0 1 1\\n"),
             ("turn 2", "2\\n-1 -1 -1 -1\\n6 6 5 5\\n6 0 5 1\\n")]
    script = "set -e; ldd --version | head -1; python3 --version; "
    for name, inp in turns:
        script += "echo '-- %s'; printf '%s' | TI_TIME_SCALE=0.1 TI_DEBUG=1 python3 titanium_botzone.py; " % (name, inp)
    if a.full_time:
        script += "echo '-- black turn 1, real 12 s clock'; printf '1\\n-1 -1 -1 -1\\n' | TI_DEBUG=1 python3 titanium_botzone.py; "
    r = subprocess.run(["docker", "run", "--rm", "-v", "%s:/bot" % OUT, "-w", "/bot", "python:3.6-slim-buster", "bash", "-c", script],
                       capture_output=True, text=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    print(r.stdout + r.stderr)
    moves = re.findall(r"^(-?\d+) (-?\d+) (-?\d+) (-?\d+)$", r.stdout, re.M)
    want = len(turns) + (1 if a.full_time else 0)
    if r.returncode or len(moves) != want or any("-1" in m for m in moves):
        fail("Linux test: expected %d legal moves, got %d (exit %d)" % (want, len(moves), r.returncode))
    if (r.stdout + r.stderr).count("nnue_main.tnn") < want:
        fail("Linux test: the NNUE was not loaded on every turn")
    over = [float(x) for x in re.findall(r"over=(-?[\d.]+)", r.stdout + r.stderr)]
    print("ok: %d turns played, NNUE loaded every turn, worst overshoot past the engine budget %.1f ms" % (want, max(over)))
    print("\nREADY TO UPLOAD: %s  (see MANIFEST.txt)" % OUT)


if __name__ == "__main__":
    main()
