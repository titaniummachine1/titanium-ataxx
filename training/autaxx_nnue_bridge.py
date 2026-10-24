import os
import subprocess
import sys

ROOT = r"C:\gitProjects\titanium-ataxx"
D = os.path.join(ROOT, "data", "engines", "autaxx")
EXE = os.path.join(D, "build", "src", "autaxx", "autaxx.exe")
NNUE = os.path.join(D, "weights", "weights-8b5d1db5176c.nnue")
TOOLCHAIN = r"C:\gitProjects\toolchains\llvm-mingw-20261006-ucrt-x86_64\bin"
os.environ["PATH"] = TOOLCHAIN + os.pathsep + os.environ.get("PATH", "")


def main():
    p = subprocess.Popen(
        [EXE],
        cwd=D,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def read_until(token):
        out = []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if token in line:
                break
        return out

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "uai":
            send("uai")
            out = read_until("uaiok")
            print("\n".join(out), flush=True)
            send("setoption name search value nnue")
            send("setoption name nnue-path value " + NNUE)
            send("setoption name hash value 32")
            send("isready")
            out = read_until("readyok")
            print("\n".join(out), flush=True)
        elif line == "quit":
            send("quit")
            break
        elif line.startswith("go"):
            send(line)
            out = read_until("bestmove")
            print("\n".join(out), flush=True)
        else:
            send(line)


if __name__ == "__main__":
    main()
