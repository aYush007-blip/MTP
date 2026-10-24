#!/usr/bin/env python3
"""
Verify and repair each 40MB chunk of torch-2.5.0-jetson.whl against NVIDIA CDN.
Re-downloads only chunks that differ, then re-assembles and tests zip integrity.
"""

from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

URL = "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
DEST = Path("/mnt/thesis/torch-2.5.0-jetson.whl")
PARTS_DIR = Path("/mnt/thesis/torch_parts")
CHUNK_SIZE = 40 * 1024 * 1024

def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as r:
        return int(r.headers.get("Content-Length", 0))

def fetch_range(url: str, start: int, end: int, max_retries: int = 3) -> bytes:
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                expected = end - start + 1
                if len(data) == expected:
                    return data
                print(f"    Short read: {len(data)}/{expected}. Retrying ({attempt}/{max_retries})...")
        except Exception as e:
            print(f"    Fetch error: {e}. Retrying ({attempt}/{max_retries})...")
            time.sleep(1.5)
    raise RuntimeError(f"Failed to fetch range {start}-{end}")

def main():
    total_size = get_remote_file_size(URL)
    print(f"Total remote size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")
    
    chunks = []
    offset = 0
    chunk_idx = 0
    while offset < total_size:
        end = min(offset + CHUNK_SIZE - 1, total_size - 1)
        part_file = PARTS_DIR / f"part_{chunk_idx:03d}.bin"
        chunks.append((chunk_idx, offset, end, part_file))
        offset = end + 1
        chunk_idx += 1

    repaired = 0
    for idx, start, end, part_path in chunks:
        expected_len = end - start + 1
        t0 = time.time()
        remote_data = fetch_range(URL, start, end)
        remote_sha = hashlib.sha256(remote_data).hexdigest()

        local_sha = None
        if part_path.exists() and part_path.stat().st_size == expected_len:
            with open(part_path, "rb") as f:
                local_sha = hashlib.sha256(f.read()).hexdigest()

        if local_sha == remote_sha:
            print(f"[{idx+1}/{len(chunks)}] Chunk {part_path.name}: MATCH ✅ ({time.time()-t0:.1f}s)")
        else:
            print(f"[{idx+1}/{len(chunks)}] Chunk {part_path.name}: MISMATCH! Repairing with verified data... 🔄")
            with open(part_path, "wb") as f:
                f.write(remote_data)
                f.flush()
                os.fsync(f.fileno())
            repaired += 1
            print(f"[{idx+1}/{len(chunks)}] Chunk {part_path.name}: Repaired and saved ✅")

    print(f"\nVerification complete! {repaired} chunks repaired out of {len(chunks)}.")

    print("Assembling final wheel...")
    with open(DEST, "wb") as out_f:
        for idx, start, end, part_path in chunks:
            with open(part_path, "rb") as in_f:
                while True:
                    b = in_f.read(4 * 1024 * 1024)
                    if not b:
                        break
                    out_f.write(b)
        out_f.flush()
        os.fsync(out_f.fileno())

    print(f"Merged wheel created: {DEST} ({DEST.stat().st_size} bytes).")
    print("Testing wheel zip integrity...")
    with zipfile.ZipFile(DEST, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt zip entry remaining: {bad}")
    print("🎉 100% PERFECT ZIP INTEGRITY VERIFIED!")

    # Clean up parts
    for idx, start, end, part_path in chunks:
        part_path.unlink()
    PARTS_DIR.rmdir()

    # Install into virtual environment
    print("\n📦 Installing NVIDIA PyTorch wheel into .venv...")
    pip_path = "/mnt/thesis/main/.venv/bin/pip"
    subprocess.check_call(["rm", "-rf", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch-*.dist-info", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch.libs"])
    subprocess.check_call([pip_path, "install", "--no-deps", str(DEST)])
    print("🎉 PyTorch 2.5.0 (CUDA enabled) installed successfully!")

if __name__ == "__main__":
    main()
