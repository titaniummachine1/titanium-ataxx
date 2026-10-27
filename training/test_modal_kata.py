import modal

app = modal.App("test-kata-build")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "git", "libzip-dev", "zlib1g-dev", "build-essential")
    .run_commands(
        "git clone --depth 1 --branch Ataxx2026-lightvector-cuda https://github.com/hzyhhzy/KataGomo.git /eng/KataGomo",
        "cd /eng/KataGomo/cpp && cmake -B build -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)",
    )
)

@app.function(image=image, gpu="T4")
def check():
    import subprocess
    out = subprocess.run(["/eng/KataGomo/cpp/build/katago", "version"], capture_output=True, text=True)
    return out.stdout + out.stderr

@app.local_entrypoint()
def main():
    print(check.remote())
