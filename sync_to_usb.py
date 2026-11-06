#!/usr/bin/env python3
"""
One-click / Background USB Sync Utility for Speech2LaTeX Lectures.
Detects USB drive AYUSH, mounts it, copies all lectures, and syncs filesystem buffers.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/thesis/headless_agent/src")
from usb_exporter import find_usb_mount, sync_all_to_usb

def main():
    print("Checking for USB drive labeled 'AYUSH'...")
    root = find_usb_mount()
    if not root:
        print("USB drive is not currently detected. Waiting up to 60s for insertion...")
        for _ in range(60):
            time.sleep(1.0)
            root = find_usb_mount()
            if root:
                break

    if not root:
        print("[Error] USB drive 'AYUSH' was not detected. Please plug in the USB flash drive and re-run.")
        sys.exit(1)

    print(f"[OK] USB Drive detected and mounted at: {root}")
    print("Syncing all lecture monographs, audio recordings, and transcripts...")
    count = sync_all_to_usb(local_base="/mnt/thesis/live_tests")
    print(f"[Success] {count} lecture sessions successfully exported to {root}/Lectures/")
    print("All buffers flushed (sync). You can now safely remove the USB drive.")

if __name__ == "__main__":
    main()

