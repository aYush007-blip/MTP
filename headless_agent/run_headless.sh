#!/usr/bin/env bash
# ==============================================================================
# Speech2LaTeX Headless Voice-Activated Edge Agent Launcher
# NVIDIA Jetson Orin Nano (100% Offline Edge Operation)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.disable_headless" ]; then
    echo "Headless background agent is disabled by user configuration (.disable_headless present). Exiting cleanly."
    exit 0
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Auto-set USB microphone hardware capture gain and playback volume
amixer -c Audio sset Mic 110 cap unmute >/dev/null 2>&1 || amixer -c 1 sset Mic 110 cap unmute >/dev/null 2>&1 || true
amixer -c Audio sset PCM 95 unmute >/dev/null 2>&1 || amixer -c 1 sset PCM 95 unmute >/dev/null 2>&1 || true

# Check virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "/mnt/thesis/main/.venv/bin/python" ]; then
    PYTHON_BIN="/mnt/thesis/main/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "================================================================================"
echo " Starting Headless Speech2LaTeX Voice Agent on Jetson Orin Nano"
echo " Working Directory : $SCRIPT_DIR"
echo " Python Binary     : $PYTHON_BIN"
echo "================================================================================"

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/src/headless_voice_agent.py" "$@"

