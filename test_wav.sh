#!/usr/bin/env bash
# ==============================================================================
# Speech2LaTeX Manual Audio Pipeline Test Launcher
# NVIDIA Jetson Orin Nano
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VENV="$SCRIPT_DIR/headless_agent/.venv/bin/python"

if [ -f "$PYTHON_VENV" ]; then
    PYTHON_EXEC="$PYTHON_VENV"
elif [ -f "$SCRIPT_DIR/main/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/main/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="$SCRIPT_DIR/headless_agent/src:$SCRIPT_DIR/main/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/mnt/thesis/llama.cpp/build/bin:/mnt/thesis/main/whisper.cpp/build-orin/bin:${LD_LIBRARY_PATH:-}"

exec "$PYTHON_EXEC" "$SCRIPT_DIR/test_system_wav.py" "$@"
