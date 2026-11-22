#!/usr/bin/env bash
# ==============================================================================
# Arm Headless Boot Mode for Real Classroom Lectures
# Removes .disable_headless so systemd automatically boots into Speech2LaTeX
# ==============================================================================

DISABLE_FLAG="/mnt/thesis/headless_agent/.disable_headless"
rm -f "$DISABLE_FLAG"

MSG="✅ LECTURE HEADLESS MODE IS ARMED!

On next power-on (without screen, monitor, or keyboard):
  1. Jetson boots directly into Speech2LaTeX autonomous service.
  2. Offline Whisper & Llama daemons will auto-load on GPU.
  3. Audio prompt will announce: 'Jetson speech agent is ready. Say hey Jetson start recording to begin.'
  4. Say: 'Hey Jetson start recording' to begin recording.
  5. While processing, ask: 'Hey Jetson what's the status?' for vocal progress updates.
  6. Say: 'Hey Jetson stop recording' to finish processing, mirror to USB drive AYUSH, and auto-poweroff.

(To return to Ubuntu Desktop Development Mode, click 'Disarm Headless (Ubuntu Dev)')."

if [ -n "$DISPLAY" ] && command -v zenity >/dev/null 2>&1; then
    zenity --info --title="Speech2LaTeX - Headless Auto-Start Armed" --text="$MSG" --width=520 --timeout=5 2>/dev/null &
fi

echo -e "\033[1;32m================================================================================\033[0m"
echo -e "\033[1;32m$MSG\033[0m"
echo -e "\033[1;32m================================================================================\033[0m"
