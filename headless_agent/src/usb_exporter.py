#!/usr/bin/env python3
"""
USB Auto-Export Manager for Offline Jetson Orin Nano Speech-to-LaTeX.

Automatically locates the inserted USB drive labeled 'AYUSH' (typically mounted
at /media/ayush02/AYUSH), creates an organized lecture directory, exports all
deliverables (.pdf, .tex, .md, .srt, .vtt, .wav, .zip), generates checksums,
and safely flushes storage buffers (sync) so the system can be turned off safely.
"""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple
import zipfile

USB_MOUNT_PREFERRED = Path("/media/ayush02/AYUSH")


def find_usb_mount() -> Optional[Path]:
    """
    Locate the mount directory of USB drive AYUSH.
    Checks preferred path first (verifying true filesystem mount),
    then /dev/disk/by-label/AYUSH, then lsblk.
    Automatically mounts the drive via udisksctl if unmounted.
    """
    # 1. Check if preferred path is actively mounted and writable
    if USB_MOUNT_PREFERRED.is_dir() and os.path.ismount(str(USB_MOUNT_PREFERRED)) and os.access(str(USB_MOUNT_PREFERRED), os.W_OK):
        return USB_MOUNT_PREFERRED

    # 1b. Try unprivileged direct mount (works if configured in /etc/fstab with 'user' option, bypassing Polkit)
    try:
        subprocess.run(["mount", str(USB_MOUNT_PREFERRED)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=5)
        if USB_MOUNT_PREFERRED.is_dir() and os.path.ismount(str(USB_MOUNT_PREFERRED)) and os.access(str(USB_MOUNT_PREFERRED), os.W_OK):
            return USB_MOUNT_PREFERRED
    except Exception:
        pass

    # 2. Check /dev/disk/by-label/AYUSH created by udev
    label_dev = Path("/dev/disk/by-label/AYUSH")
    if label_dev.exists():
        try:
            m_res = subprocess.run(
                ["udisksctl", "mount", "-b", str(label_dev)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False
            )
            if USB_MOUNT_PREFERRED.is_dir() and os.path.ismount(str(USB_MOUNT_PREFERRED)) and os.access(str(USB_MOUNT_PREFERRED), os.W_OK):
                return USB_MOUNT_PREFERRED
            m_match = re.search(r'Mounted .*? at (/[^\s\.]+)', m_res.stdout)
            if m_match and Path(m_match.group(1)).is_dir():
                return Path(m_match.group(1))
        except Exception as e:
            print(f"[USB Exporter] by-label mount warning: {e}")

    # 3. Search through /media and /mnt for any mounted AYUSH
    for base in [Path("/media/ayush02"), Path("/media"), Path("/mnt")]:
        if base.is_dir():
            for p in base.glob("**/AYUSH"):
                if p.is_dir() and os.path.ismount(str(p)) and os.access(str(p), os.W_OK):
                    return p

    # 4. Inspect lsblk block devices for drive labeled AYUSH
    try:
        res = subprocess.run(
            ["lsblk", "-J", "-o", "NAME,LABEL,MOUNTPOINT"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5
        )
        if res.returncode == 0:
            import json
            data = json.loads(res.stdout)
            for dev in data.get("blockdevices", []):
                for child in dev.get("children", [dev]):
                    if (child.get("label") or "").upper() == "AYUSH":
                        mp = child.get("mountpoint")
                        if mp and Path(mp).is_dir() and os.path.ismount(mp):
                            return Path(mp)
                        dev_node = f"/dev/{child.get('name')}"
                        try:
                            m_res = subprocess.run(
                                ["udisksctl", "mount", "-b", dev_node],
                                capture_output=True,
                                text=True,
                                timeout=15,
                                check=False
                            )
                            if USB_MOUNT_PREFERRED.is_dir() and os.path.ismount(str(USB_MOUNT_PREFERRED)):
                                return USB_MOUNT_PREFERRED
                            m_match = re.search(r'Mounted .*? at (/[^\s\.]+)', m_res.stdout)
                            if m_match and Path(m_match.group(1)).is_dir():
                                return Path(m_match.group(1))
                        except Exception:
                            pass
    except Exception:
        pass

    return None


def sync_all_to_usb(local_base: str = "/mnt/thesis/live_tests") -> int:
    """
    Sync all lectures from local SSD to USB drive AYUSH.
    Returns number of lectures synced.
    """
    usb_root = find_usb_mount()
    if not usb_root:
        print("[USB Exporter] USB drive AYUSH is not mounted. Cannot sync.")
        return 0

    usb_lectures = usb_root / "Lectures"
    usb_lectures.mkdir(parents=True, exist_ok=True)
    local_p = Path(local_base)
    synced_count = 0

    for d in sorted(local_p.iterdir()):
        if d.is_dir() and (d.name.startswith("202") or "session" in d.name or "Lecture" in d.name):
            target_d = usb_lectures / d.name
            target_d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    dest_f = target_d / f.name
                    if not dest_f.exists() or dest_f.stat().st_size != f.stat().st_size:
                        shutil.copy2(f, dest_f)
            synced_count += 1

    subprocess.run(["sync"], check=False)
    print(f"[USB Exporter] Successfully synced {synced_count} lecture sessions to {usb_lectures}.")
    return synced_count


def sanitize_filename(name: str) -> str:
    """Clean filename removing invalid characters."""
    name = re.sub(r'[^\w\s-]', '', name).strip()
    return re.sub(r'[-\s]+', '_', name)


def compute_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class LectureExporter:
    """
    Orchestrates the export of all lecture deliverables to local SSD and USB drive.
    """
    def __init__(self, local_base: str = "/mnt/thesis/live_tests"):
        self.local_base = Path(local_base)
        self.local_base.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        lecture_title: str,
        files_dict: Dict[str, Path],
        timestamp: Optional[str] = None
    ) -> Dict:
        """
        Export lecture outputs.
        Args:
            lecture_title: Short title, e.g. 'Z_Transform_Properties'
            files_dict: Map of type to source file path, e.g. {
                'pdf': Path('/tmp/lecture.pdf'),
                'tex': Path('/tmp/lecture.tex'),
                'md': Path('/tmp/lecture.md'),
                'srt': Path('/tmp/lecture.srt'),
                'vtt': Path('/tmp/lecture.vtt'),
                'wav': Path('/tmp/lecture.wav'),
            }
            timestamp: Optional YYYYMMDD_HHMMSS timestamp. If None, uses current.
        Returns:
            Dictionary containing export status, local and USB paths, and checksums.
        """
        if not timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        sanitized_title = sanitize_filename(lecture_title) or "Lecture"
        if sanitized_title.startswith(timestamp):
            dir_name = sanitized_title
        else:
            dir_name = f"{timestamp}_{sanitized_title}"

        # 1. Prepare Local Directory
        local_dir = self.local_base / dir_name
        local_dir.mkdir(parents=True, exist_ok=True)

        local_files = {}
        for ext, src_path in files_dict.items():
            if src_path and Path(src_path).is_file():
                dest_path = local_dir / Path(src_path).name
                shutil.copy2(src_path, dest_path)
                local_files[ext] = dest_path

        # 2. Create ZIP Bundle locally
        zip_path = local_dir / f"{sanitized_title}_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext, p in local_files.items():
                zf.write(p, arcname=p.name)
        local_files["zip"] = zip_path

        # 3. Compute Checksums
        checksum_file = local_dir / "SHA256SUMS.txt"
        with open(checksum_file, "w", encoding="utf-8") as cs_f:
            for p in local_dir.iterdir():
                if p.is_file() and p.name != "SHA256SUMS.txt":
                    cs = compute_sha256(p)
                    cs_f.write(f"{cs}  {p.name}\n")
        local_files["checksums"] = checksum_file

        # 4. Export to USB Drive AYUSH if available (with auto-retry and size verification)
        usb_root = None
        for attempt in range(3):
            usb_root = find_usb_mount()
            if usb_root:
                break
            time.sleep(1.5)

        usb_dir = None
        usb_success = False

        if usb_root:
            try:
                usb_lectures = usb_root / "Lectures"
                usb_lectures.mkdir(parents=True, exist_ok=True)
                usb_dir = usb_lectures / dir_name
                usb_dir.mkdir(parents=True, exist_ok=True)

                all_copied = True
                for p in local_dir.iterdir():
                    if p.is_file():
                        target_f = usb_dir / p.name
                        shutil.copy2(p, target_f)
                        if not target_f.is_file() or target_f.stat().st_size != p.stat().st_size:
                            all_copied = False

                # Ensure FAT32 buffers are committed to flash memory
                subprocess.run(["sync"], check=False)
                usb_success = all_copied
            except Exception as e:
                print(f"[USB Exporter Error] Failed copying to USB: {e}")
                usb_success = False

        return {
            "title": sanitized_title,
            "timestamp": timestamp,
            "local_path": str(local_dir),
            "usb_path": str(usb_dir) if usb_dir else None,
            "usb_success": usb_success,
            "files": {k: str(v) for k, v in local_files.items()},
        }


if __name__ == "__main__":
    exporter = LectureExporter()
    usb = find_usb_mount()
    print(f"USB Mount Point: {usb}")
    if usb:
        print(f"USB Free Space: {shutil.disk_usage(usb).free / (1024**3):.2f} GB")
    else:
        print("USB drive AYUSH not currently detected.")
