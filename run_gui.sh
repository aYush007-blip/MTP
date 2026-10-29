#!/usr/bin/env bash
# ==============================================================================
# Jetson Speech-to-LaTeX Showcase Launcher
# 100% Offline Edge Startup Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="${SCRIPT_DIR}/main"
VENV_PYTHON="${MAIN_DIR}/.venv/bin/python"

if [ -f "${VENV_PYTHON}" ]; then
    PYTHON_CMD="${VENV_PYTHON}"
else
    PYTHON_CMD="python3"
fi

echo "======================================================="
echo " Starting Jetson Speech-to-LaTeX Showcase GUI"
echo " Working Directory: ${MAIN_DIR}"
echo " Python Binary:     ${PYTHON_CMD}"
echo " Web GUI URL:       http://localhost:8000"
echo "======================================================="

# Auto-open browser in desktop environment
if [ -n "${DISPLAY}" ] || [ -n "${WAYLAND_DISPLAY}" ]; then
    (sleep 3 && "${PYTHON_CMD}" -m webbrowser "http://localhost:8000" >/dev/null 2>&1) &
fi

cd "${MAIN_DIR}"
exec "${PYTHON_CMD}" gui.py
