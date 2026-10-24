#!/usr/bin/env python3
"""
Robust Chunked Downloader for Large Wheels on Flaky/Timeout-Prone Networks.
Downloads in strictly bounded 40MB slices with fresh connections and strict verification.
"""

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
CHUNK_SIZE = 40 * 1024 * 1024  # 40 MB chunks (finishes in ~4s at 10MB/s)

def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as r:
        return int(r.headers.get("Content-Length", 0))

def download_chunk_with_retries(url: str, start: int, end: int, part_path: Path, max_retries: int = 5):
    expected_len = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected_len:
        print(f"  Chunk {part_path.name} already verified ({expected_len} bytes).")
        return True

    # If partial, delete and re-download cleanly
    if part_path.exists():
        part_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "wb") as f:
                while True:
                    buf = resp.read(256 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                f.flush()
                os.fsync(f.fileno())

            actual_len = part_path.stat().st_size
            if actual_len == expected_len:
                speed = actual_len / max(0.001, time.time() - t0) / (1024 * 1024)
                print(f"  Chunk {part_path.name} ({actual_len/(1024*1024):.1f} MB) in {time.time()-t0:.1f}s [{speed:.2f} MB/s] ✅")
                return True
            else:
                print(f"  Chunk {part_path.name} size mismatch: got {actual_len}, expected {expected_len}. Retrying ({attempt}/{max_retries})...")
                part_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  Chunk {part_path.name} attempt {attempt} failed: {e}. Retrying...")
            part_path.unlink(missing_ok=True)
            time.sleep(2)

    raise RuntimeError(f"Failed to download chunk {part_path.name} after {max_retries} attempts.")

def main():
    total_size = get_remote_file_size(URL)
    print(f"Target size: {total_size} bytes ({total_size / (1024*1024):.1f} MB)")
    
    parts_dir = Path("/mnt/thesis/torch_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    offset = 0
    chunk_idx = 0
    while offset < total_size:
        end = min(offset + CHUNK_SIZE - 1, total_size - 1)
        part_file = parts_dir / f"part_{chunk_idx:03d}.bin"
        chunks.append((offset, end, part_file))
        offset = end + 1
        chunk_idx += 1

    print(f"Divided into {len(chunks)} chunks of ~{CHUNK_SIZE/(1024*1024):.0f} MB each.")
    t_start = time.time()

    for idx, (c_start, c_end, c_path) in enumerate(chunks):
        pct = (c_start / total_size) * 100.0
        print(f"[{idx+1}/{len(chunks)}] ({pct:4.1f}%) Downloading {c_path.name} (bytes {c_start}-{c_end})...")
        download_chunk_with_retries(URL, c_start, c_end, c_path)

    print("\nAll chunks downloaded and verified. Assembling final wheel...")
    if DEST.exists():
        DEST.unlink()

    with open(DEST, "wb") as out_f:
        for _, _, c_path in chunks:
            with open(c_path, "rb") as in_f:
                while True:
                    b = in_f.read(4 * 1024 * 1024)
                    if not b:
                        break
                    out_f.write(b)
        out_f.flush()
        os.fsync(out_f.fileno())

    print(f"Final wheel assembled: {DEST} ({DEST.stat().st_size} bytes).")
    assert DEST.stat().st_size == total_size, f"Total size mismatch: {DEST.stat().st_size} vs {total_size}"

    print("Verifying complete wheel zip integrity...")
    with zipfile.ZipFile(DEST, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt zip entry found: {bad}")
    print("🎉 100% PERFECT ZIP INTEGRITY VERIFIED!")

    # Clean up part files
    for _, _, c_path in chunks:
        c_path.unlink()
    parts_dir.rmdir()

    # Install into virtual environment
    print("\n📦 Installing NVIDIA PyTorch wheel into .venv...")
    pip_path = "/mnt/thesis/main/.venv/bin/pip"
    subprocess.check_call(["rm", "-rf", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch-*.dist-info", "/mnt/thesis/main/.venv/lib/python3.10/site-packages/torch.libs"])
    subprocess.check_call([pip_path, "install", "--no-deps", str(DEST)])
    print("🎉 PyTorch 2.5.0 (CUDA enabled) installed successfully!")

if __name__ == "__main__":
    main()
