#!/usr/bin/env python3
"""
High-Speed Parallel Multi-Threaded Model Downloader for Jetson Orin.
Uses HTTP Range requests with parallel threads to saturate network bandwidth.
Supports resuming interrupted downloads.
"""

import argparse
import concurrent.futures
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import urllib.request

MODELS = {
    "qwen1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors",
        "filename": "model.safetensors",
        "dir": "/mnt/thesis/models/Qwen2.5-1.5B-Instruct",
    },
    "qwen1.5b-gguf": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "dir": "/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF",
    },
    "qwen0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
        "filename": "model.safetensors",
        "dir": "/mnt/thesis/models/Qwen2.5-0.5B-Instruct",
    },
    "qwen0.5b-gguf": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "dir": "/mnt/thesis/models/Qwen2.5-0.5B-Instruct-GGUF",
    },
}


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as r:
        return int(r.headers.get("Content-Length", 0))


def download_chunk(url: str, start: int, end: int, dest_path: str, part_num: int):
    part_path = f"{dest_path}.part_{part_num}"
    existing_bytes = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    if existing_bytes >= (end - start + 1):
        return existing_bytes

    cur_start = start + existing_bytes
    req = urllib.request.Request(url, headers={"Range": f"bytes={cur_start}-{end}"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "ab") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return os.path.getsize(part_path)


def download_file_parallel(url: str, dest_file: Path, num_threads: int = 8):
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if dest_file.exists():
        print(f"✅ Target file already exists: {dest_file}")
        return

    print(f"Querying file size for {dest_file.name}...")
    total_size = get_remote_file_size(url)
    if total_size <= 0:
        raise ValueError(f"Could not determine remote file size from {url}")

    print(f"Total file size: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.1f} MB)")
    print(f"Spawning {num_threads} parallel download threads...")

    chunk_size = total_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < (num_threads - 1) else (total_size - 1)
        ranges.append((start, end, str(dest_file), i))

    t0 = time.time()
    last_print = t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_chunk, url, r[0], r[1], r[2], r[3]): r for r in ranges}

        while any(not f.done() for f in futures):
            time.sleep(2.0)
            now = time.time()
            if now - last_print >= 5.0:
                # Calculate downloaded bytes
                downloaded = 0
                for i in range(num_threads):
                    p_file = f"{dest_file}.part_{i}"
                    if os.path.exists(p_file):
                        downloaded += os.path.getsize(p_file)
                pct = (downloaded / total_size) * 100.0
                elapsed = now - t0
                speed = downloaded / max(0.001, elapsed) / (1024 * 1024)
                eta = (total_size - downloaded) / max(0.001, downloaded / max(0.001, elapsed))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {pct:5.1f}% | {downloaded/(1024**2):.1f}/{total_size/(1024**2):.1f} MB | {speed:.2f} MB/s | ETA: {int(eta//60)}m {int(eta%60)}s")
                last_print = now

    # Merge parts
    print("\nMerging downloaded parts into final file...")
    with open(dest_file, "wb") as out_f:
        for i in range(num_threads):
            part_path = f"{dest_file}.part_{i}"
            with open(part_path, "rb") as in_f:
                while True:
                    buf = in_f.read(1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)
            os.remove(part_path)

    total_time = time.time() - t0
    final_size = dest_file.stat().st_size
    print(f"🎉 Successfully downloaded {dest_file.name} ({final_size / (1024**2):.1f} MB) in {total_time:.1f}s ({final_size/total_time/(1024**2):.2f} MB/s)!")


def download_model(model_key: str, threads: int = 8):
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
        sys.exit(1)

    cfg = MODELS[model_key]
    out_dir = Path(cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # First, download metadata files using huggingface_hub snapshot_download
    print(f"\n=======================================================")
    print(f"⬇️ Downloading Metadata for {cfg['repo']}...")
    print(f"=======================================================")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=cfg["repo"],
        local_dir=str(out_dir),
        allow_patterns=["*.json", "*.txt"],
    )

    # Second, download the large weights file using verified hf_hub_download
    print(f"\n=======================================================")
    print(f"⬇️ Downloading Weights File ({cfg['filename']}) via verified hf_hub_download...")
    print(f"=======================================================")
    from huggingface_hub import hf_hub_download
    dest_file = hf_hub_download(
        repo_id=cfg["repo"],
        filename=cfg["filename"],
        local_dir=str(out_dir),
    )
    print(f"✅ Verified weights downloaded to: {dest_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded Model Downloader for Jetson")
    parser.add_argument("model", choices=list(MODELS.keys()), default="qwen1.5b", nargs="?")
    parser.add_argument("--threads", type=int, default=8, help="Number of parallel download threads")
    args = parser.parse_args()

    download_model(args.model, threads=args.threads)
