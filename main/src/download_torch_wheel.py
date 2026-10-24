#!/usr/bin/env python3
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

URL = "https://files.pythonhosted.org/packages/69/f0/46e74e0d145f43fa506cb336eaefb2d240547e4ce1f496e442711093ab25/torch-2.5.1-cp310-cp310-manylinux2014_aarch64.whl"
DEST = Path("/tmp/torch-2.5.1-cp310-cp310-manylinux2014_aarch64.whl")
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
    print(f"Part {part_num} downloading bytes {cur_start} to {end}...")
    req = urllib.request.Request(url, headers={"Range": f"bytes={cur_start}-{end}"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "ab") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)

    final_size = os.path.getsize(part_path)
    print(f"Part {part_num} finished ({final_size} bytes)")
    return final_size

def main():
    total_size = get_remote_file_size(URL)
    print(f"Total remote size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)")
    
    chunk_size = total_size // NUM_THREADS
    ranges = []
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < (NUM_THREADS - 1) else (total_size - 1)
        ranges.append((start, end, str(DEST), i))

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(download_chunk, URL, r[0], r[1], r[2], r[3]) for r in ranges]
        concurrent.futures.wait(futures)

    # Check that all parts exist and match exact byte lengths
    for i in range(NUM_THREADS):
        part_path = f"{DEST}.part_{i}"
        start, end = ranges[i][0], ranges[i][1]
        expected = end - start + 1
        actual = os.path.getsize(part_path)
        if actual != expected:
            raise ValueError(f"Part {i} size mismatch: got {actual}, expected {expected}")
        print(f"Verified part {i}: {actual} bytes")

    # Merge parts
    print("Merging parts...")
    with open(DEST, "wb") as out_f:
        for i in range(NUM_THREADS):
            part_path = f"{DEST}.part_{i}"
            with open(part_path, "rb") as in_f:
                while True:
                    buf = in_f.read(1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)

    print(f"Merged wheel created at {DEST} ({os.path.getsize(DEST)} bytes)")
    
    # Test zipfile
    print("Testing wheel zip integrity...")
    with zipfile.ZipFile(DEST, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Zipfile test failed on file: {bad}")
    print("✅ Wheel zip integrity verified!")

    # Clean up part files
    for i in range(NUM_THREADS):
        os.remove(f"{DEST}.part_{i}")

    # Install wheel into virtualenv
    print("Installing wheel into /mnt/thesis/main/.venv...")
    pip_path = "/mnt/thesis/main/.venv/bin/pip"
    subprocess.check_call([pip_path, "install", "--force-reinstall", "--no-deps", str(DEST)])
    print("✅ Wheel installed successfully!")

if __name__ == "__main__":
    main()
