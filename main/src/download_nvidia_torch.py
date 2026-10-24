#!/usr/bin/env python3
import concurrent.futures
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

URL = "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
DEST = Path("/mnt/thesis/torch-2.5.0-jetson.whl")
NUM_THREADS = 8

def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as r:
        return int(r.headers.get("Content-Length", 0))

def download_chunk(url: str, start: int, end: int, dest_path: str, part_num: int):
    part_path = f"{dest_path}.part_{part_num}"
    existing_bytes = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    expected_bytes = end - start + 1

    if existing_bytes >= expected_bytes:
        print(f"Part {part_num} already complete ({existing_bytes}/{expected_bytes} bytes)")
        return existing_bytes

    cur_start = start + existing_bytes
    req = urllib.request.Request(url, headers={"Range": f"bytes={cur_start}-{end}"})
    with urllib.request.urlopen(req, timeout=45) as resp, open(part_path, "ab") as f:
        while True:
            chunk = resp.read(128 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return os.path.getsize(part_path)

def main():
    if DEST.exists():
        print(f"File {DEST} already exists. Verifying zip...")
        try:
            with zipfile.ZipFile(DEST, "r") as zf:
                if zf.testzip() is None:
                    print("✅ Existing wheel is valid!")
                    install_wheel()
                    return
        except Exception:
            print("Existing wheel is incomplete/corrupt. Re-downloading...")
            DEST.unlink()

    total_size = get_remote_file_size(URL)
    print(f"Total NVIDIA PyTorch wheel size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")

    chunk_size = total_size // NUM_THREADS
    ranges = []
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < (NUM_THREADS - 1) else (total_size - 1)
        ranges.append((start, end, str(DEST), i))

    t0 = time.time()
    last_print = t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(download_chunk, URL, r[0], r[1], r[2], r[3]): r for r in ranges}

        while any(not f.done() for f in futures):
            time.sleep(2.0)
            now = time.time()
            if now - last_print >= 5.0:
                downloaded = sum(
                    os.path.getsize(f"{DEST}.part_{i}")
                    for i in range(NUM_THREADS)
                    if os.path.exists(f"{DEST}.part_{i}")
                )
                pct = (downloaded / total_size) * 100.0
                elapsed = now - t0
                speed = downloaded / max(0.001, elapsed) / (1024 * 1024)
                eta = (total_size - downloaded) / max(0.001, downloaded / max(0.001, elapsed))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {pct:5.1f}% | {downloaded/(1024**2):.1f}/{total_size/(1024**2):.1f} MB | {speed:.2f} MB/s | ETA: {int(eta//60)}m {int(eta%60)}s")
                last_print = now

        # Wait for all futures and check for exceptions
        for f in futures:
            f.result()

    # Merge parts
    print("\nMerging downloaded parts...")
    with open(DEST, "wb") as out_f:
        for i in range(NUM_THREADS):
            part_path = f"{DEST}.part_{i}"
            with open(part_path, "rb") as in_f:
                while True:
                    buf = in_f.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)
            os.remove(part_path)

    print(f"Merged wheel created at {DEST} ({DEST.stat().st_size} bytes)")
    print("Testing wheel zip integrity...")
    with zipfile.ZipFile(DEST, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt zipfile entry: {bad}")
    print("✅ Wheel zip integrity 100% verified!")

    install_wheel()

def install_wheel():
    print("\n=======================================================")
    print("📦 Installing NVIDIA JetPack PyTorch wheel into .venv...")
    print("=======================================================")
    pip_path = "/mnt/thesis/main/.venv/bin/pip"
    # Remove existing torch completely first to avoid stale .so mixing
    subprocess.check_call(["rm", "-rf", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch-*.dist-info", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch.libs"])
    subprocess.check_call([pip_path, "install", "--no-deps", str(DEST)])
    print("✅ NVIDIA Jetson PyTorch installed successfully!")

if __name__ == "__main__":
    main()
