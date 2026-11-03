# Models Manifest & Verification Guide

This document records the exact models used by the **Edge Speech-to-LaTeX** system on the NVIDIA Jetson Orin Nano, including cryptographic checksums, authoritative backup locations, and automated restore procedures.

---

## 1. Verified Model Registry

| Model Name | Role | Format & Size | Authoritative Primary Path | Cryptographic SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-1.5B-Instruct** | Mathematical & Prose Typesetting SLM | Q4_K_M GGUF (1.1 GB) | `/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| **Whisper-Small.en** | Real-Time Edge ASR (Flash Attention) | GGML Bin (466 MB) | `/mnt/thesis/models/whisper/ggml-small.en.bin` | `d954925f8ceba9e4ad44302a550b3d044821b356908682c5fac30fbe7b5168d5` |
| **Whisper-Base.en** | Lightweight Fallback ASR | GGML Bin (142 MB) | `/mnt/thesis/models/whisper/ggml-base.bin` | `4b1ff093cc4eed1bee23b5733f1ebc251afd398f4f01e06b7528dc7203d37448` |

---

## 2. Dual-Layer Storage & Backup Architecture

To protect against filesystem sector corruptions or bit-rot on external storage devices under sustained continuous load, models are stored across two independent tiers:

1. **Tier 1 (Authoritative Internal eMMC):**
   * Path: `/home/ayush02/models/backup_working_models/`
   * Device: `/dev/mmcblk0p1` (Rock-solid internal solid-state eMMC)
   * Permissions: `chmod 444` (Strict read-only)
2. **Tier 2 (Workspace Storage & Compatibility Symlinks):**
   * Workspace symlink: `/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf -> /home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
   * Local Whisper models: `/mnt/thesis/models/whisper/`

---

## 3. Automated One-Command Verification & Recovery

An automated verification and recovery script is provided in [`scripts/setup_models.sh`](scripts/setup_models.sh):

```bash
/mnt/thesis/scripts/setup_models.sh
```

### What this script does:
1. Calculates SHA-256 hashes of the active models.
2. If any hash differs (due to write corruption or missing files), automatically restores the verified bit-exact copy from `/home/ayush02/models/backup_working_models/`.
3. If no local backup exists, downloads the pristine weights directly from official Hugging Face repositories and verifies their cryptographic signature.
4. Locks all weights with `chmod 444` (read-only) and refreshes symlinks.

---

## 4. Upstream Upstream Sources

* **Qwen2.5-1.5B-Instruct-GGUF:** [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
* **Whisper GGML Models:** [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)
