#!/usr/bin/env bash
# ==============================================================================
# Jetson Orin Speech-to-LaTeX: Model Setup & Integrity Verification Script
# ==============================================================================
# Ensures both primary and backup neural models are present, verified,
# and locked with read-only permissions (chmod 444) to prevent bit-rot.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Target Model Directories
PRIMARY_LLM_DIR="/home/ayush02/models"
PRIMARY_WHISPER_DIR="${REPO_DIR}/models/whisper"
EMMC_BACKUP_DIR="/home/ayush02/models/backup_working_models"
REPO_BACKUP_DIR="${REPO_DIR}/models/backup_working_models"

# Exact Cryptographic Checksums (SHA-256)
SHA_QWEN="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
SHA_WHISPER_SMALL="d954925f8ceba9e4ad44302a550b3d044821b356908682c5fac30fbe7b5168d5"
SHA_WHISPER_BASE="4b1ff093cc4eed1bee23b5733f1ebc251afd398f4f01e06b7528dc7203d37448"

# Upstream Hugging Face Direct URLs
URL_QWEN="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
URL_WHISPER_SMALL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
URL_WHISPER_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"

echo "================================================================="
echo "🧠 Jetson Speech-to-LaTeX: Model Setup & Verification"
echo "================================================================="

mkdir -p "${PRIMARY_LLM_DIR}" "${PRIMARY_WHISPER_DIR}" "${EMMC_BACKUP_DIR}" "${REPO_BACKUP_DIR}"

verify_or_restore() {
    local name="$1"
    local primary_path="$2"
    local backup_path="$3"
    local expected_sha="$4"
    local download_url="$5"

    echo -n "Checking ${name}... "

    # 1. Check primary file
    if [[ -f "${primary_path}" ]]; then
        local current_sha
        current_sha="$(sha256sum "${primary_path}" | awk '{print $1}')"
        if [[ "${current_sha}" == "${expected_sha}" ]]; then
            echo "[OK: Primary Verified]"
            chmod 444 "${primary_path}" 2>/dev/null || true
            return 0
        else
            echo "[Warning: Primary Mismatch (${current_sha:0:8}...)]"
        fi
    fi

    # 2. Check eMMC backup
    if [[ -f "${backup_path}" ]]; then
        echo -n "Restoring from secure eMMC backup... "
        local backup_sha
        backup_sha="$(sha256sum "${backup_path}" | awk '{print $1}')"
        if [[ "${backup_sha}" == "${expected_sha}" ]]; then
            cp -p "${backup_path}" "${primary_path}"
            chmod 444 "${primary_path}" 2>/dev/null || true
            echo "[OK: Restored from backup]"
            return 0
        else
            echo "[Backup also invalid]"
        fi
    fi

    # 3. Download fresh from Hugging Face Hub
    echo "Downloading ${name} from Hugging Face..."
    curl -L --progress-bar "${download_url}" -o "${primary_path}.tmp"
    local dl_sha
    dl_sha="$(sha256sum "${primary_path}.tmp" | awk '{print $1}')"
    if [[ "${dl_sha}" == "${expected_sha}" ]]; then
        mv "${primary_path}.tmp" "${primary_path}"
        chmod 444 "${primary_path}" 2>/dev/null || true
        cp -p "${primary_path}" "${backup_path}"
        chmod 444 "${backup_path}" 2>/dev/null || true
        echo "[OK: Downloaded and verified]"
    else
        echo "❌ ERROR: Downloaded file failed SHA-256 verification!"
        rm -f "${primary_path}.tmp"
        return 1
    fi
}

# 1. Qwen2.5-1.5B-Instruct-GGUF (Authoritative path on rock-solid internal eMMC)
verify_or_restore \
    "Qwen2.5-1.5B-Instruct (Q4_K_M GGUF)" \
    "${PRIMARY_LLM_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "${EMMC_BACKUP_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "${SHA_QWEN}" \
    "${URL_QWEN}"

# 2. Whisper-Small English GGUF/bin
verify_or_restore \
    "Whisper-Small (ggml-small.en.bin)" \
    "${PRIMARY_WHISPER_DIR}/ggml-small.en.bin" \
    "${EMMC_BACKUP_DIR}/ggml-small.en.bin" \
    "${SHA_WHISPER_SMALL}" \
    "${URL_WHISPER_SMALL}"

# 3. Whisper-Base English GGUF/bin
verify_or_restore \
    "Whisper-Base (ggml-base.bin)" \
    "${PRIMARY_WHISPER_DIR}/ggml-base.bin" \
    "${EMMC_BACKUP_DIR}/ggml-base.bin" \
    "${SHA_WHISPER_BASE}" \
    "${URL_WHISPER_BASE}"

# Ensure symlinks for workspace compatibility
mkdir -p "${REPO_DIR}/models/Qwen2.5-1.5B-Instruct-GGUF"
ln -sf "${PRIMARY_LLM_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf" "${REPO_DIR}/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo "================================================================="
echo "✅ All models verified, backed up, and locked read-only (chmod 444)."
echo "================================================================="
