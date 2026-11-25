#!/usr/bin/env bash
# ==============================================================================
# Disarm Headless Mode for Development & Testing inside Ubuntu GUI
# Creates .disable_headless so systemd exits cleanly at boot
# ==============================================================================

DISABLE_FLAG="/mnt/thesis/headless_agent/.disable_headless"
touch "$DISABLE_FLAG"

# Terminate any running headless instance
pkill -f "headless_voice_agent.py" 2>/dev/null || true

MSG="🛑 HEADLESS AUTO-START IS DISARMED (UBUNTU DEV MODE)!

On next boot:
  • The background voice daemon will NOT run at startup.
  • 100% of CPU, GPU, and RAM remain free for testing, thesis, and GUI coding.

Before bringing the Jetson to the lecture hall, click:
  'Arm Lecture Headless Boot' to re-enable autonomous boot auto-start."

if [ -n "$DISPLAY" ] && command -v zenity >/dev/null 2>&1; then
    zenity --info --title="Speech2LaTeX - Dev Mode (Headless Disarmed)" --text="$MSG" --width=500 --timeout=5 2>/dev/null &
fi

echo -e "\033[1;33m================================================================================\033[0m"
echo -e "\033[1;33m$MSG\033[0m"
echo -e "\033[1;33m================================================================================\033[0m"
