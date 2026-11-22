#!/usr/bin/env bash
# ==============================================================================
# Speech2LaTeX Interactive Voice Agent Launcher (Desktop GUI Mode)
# NVIDIA Jetson Orin Nano
# ==============================================================================

cd /mnt/thesis/headless_agent

# Ensure environment
export PYTHONPATH="/mnt/thesis/headless_agent/src:/mnt/thesis/main/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/mnt/thesis/llama.cpp/build/bin:/usr/local/cuda/lib64:/mnt/thesis/main/whisper.cpp/build-orin/bin:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="/mnt/thesis/headless_agent/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "================================================================================"
echo " 🎙️ Speech2LaTeX Autonomous Voice Agent (Interactive Desktop Mode)"
echo " Working Directory : /mnt/thesis/headless_agent"
echo " Python Binary     : $PYTHON_BIN"
echo " Note              : Running in safe testing mode (--no-auto-shutdown)"
echo " Press Ctrl+C at any time in this window to stop the agent."
echo "================================================================================"

exec "$PYTHON_BIN" -u "/mnt/thesis/headless_agent/src/headless_voice_agent.py" --no-auto-shutdown "$@"

