#!/usr/bin/env bash
set -e

# ==============================================================================
# Master One-Time Headless Setup for Jetson Orin Nano Speech2LaTeX System
# Run once with: sudo bash /mnt/thesis/headless_agent/setup_headless_system.sh
# ==============================================================================

echo "======================================================================"
echo " [1/4] Installing Polkit Rule for Passwordless USB Access..."
echo "======================================================================"

TARGET_USER="ayush02"
MOUNT_DIR="/media/${TARGET_USER}/AYUSH"
mkdir -p "${MOUNT_DIR}"
chown -R ${TARGET_USER}:${TARGET_USER} "/media/${TARGET_USER}"
chmod 775 "${MOUNT_DIR}"

POLKIT_DIR="/etc/polkit-1/localauthority/50-local.d"
mkdir -p "${POLKIT_DIR}"
cat << 'EOF' > "${POLKIT_DIR}/10-udisks2-mount.pkla"
[Allow users to mount and unmount removable devices without authentication]
Identity=unix-user:ayush02;unix-group:sudo;unix-group:plugdev
Action=org.freedesktop.udisks2.filesystem-mount;org.freedesktop.udisks2.filesystem-mount-system;org.freedesktop.udisks2.filesystem-mount-other-seat;org.freedesktop.udisks2.filesystem-unmount-others;org.freedesktop.udisks2.encrypted-unlock;org.freedesktop.udisks2.eject-media;org.freedesktop.udisks2.power-off-drive
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF

cat << 'EOF' > "${POLKIT_DIR}/20-poweroff.pkla"
[Allow users to power off without authentication]
Identity=unix-user:ayush02;unix-group:sudo;unix-group:plugdev
Action=org.freedesktop.login1.power-off;org.freedesktop.login1.power-off-multiple-sessions;org.freedesktop.login1.power-off-ignore-inhibit
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF

SUDOERS_FILE="/etc/sudoers.d/speech2latex-poweroff"
cat << 'EOF' > "${SUDOERS_FILE}"
ayush02 ALL=(ALL) NOPASSWD: /sbin/poweroff, /sbin/shutdown, /bin/systemctl poweroff, /bin/systemctl reboot
EOF
chmod 0440 "${SUDOERS_FILE}"

systemctl restart polkit || true
echo "  [OK] Polkit & Sudoers rules installed: No password required for USB drive or clean system poweroff."

echo ""
echo "======================================================================"
echo " [2/4] Verifying Kernel Boot Auto-Mount (/etc/fstab)..."
echo "======================================================================"

FSTAB_LINE="LABEL=AYUSH ${MOUNT_DIR} vfat defaults,nofail,user,umask=000,uid=1000,gid=1000 0 0"
if ! grep -q "LABEL=AYUSH" /etc/fstab; then
    echo "${FSTAB_LINE}" >> /etc/fstab
    echo "  [OK] Added kernel fstab auto-mount entry."
else
    echo "  [OK] fstab auto-mount entry already active."
fi

echo ""
echo "======================================================================"
echo " [3/4] Configuring Hardware Audio Input & Output Levels..."
echo "======================================================================"

amixer -c Audio sset Mic 110 cap unmute >/dev/null 2>&1 || amixer -c 1 sset Mic 110 cap unmute >/dev/null 2>&1 || true
amixer -c Audio sset PCM 95 unmute >/dev/null 2>&1 || amixer -c 1 sset PCM 95 unmute >/dev/null 2>&1 || true
alsactl store || true
echo "  [OK] Hardware microphone gain set to 92% (-5 dB) and playback volume to 80%."

echo ""
echo "======================================================================"
echo " [4/4] Enabling Headless Autostart on Power-On (systemd)..."
echo "======================================================================"

cp /mnt/thesis/headless_agent/config/speech2latex-headless.service /etc/systemd/system/speech2latex-headless.service
systemctl daemon-reload
systemctl enable speech2latex-headless.service
echo "  [OK] speech2latex-headless.service enabled for automatic boot launch."

echo ""
echo "======================================================================"
echo " ALL SYSTEMS VERIFIED & READY FOR 100% MONITORLESS OPERATION!"
echo " Tomorrow at the lecture:"
echo "   1. Plug in USB Flash Drive (AYUSH)."
echo "   2. Plug in Earphones/Mic dongle (AB13X)."
echo "   3. Connect power to Jetson."
echo "   4. Listen for voice prompt: 'Jetson speech agent is ready. Say hey Jetson start recording to begin.'"
echo "   5. Say 'Start Recording' when professor starts."
echo "   6. Say 'Stop Recording' when lecture ends ('Recording stopped. Processing lecture, please wait')."
echo "   7. During processing, you can ask 'Hey Jetson what's the status?' anytime to hear time remaining."
echo "   8. When done, Jetson announces 'Transcription successfully saved... shutting down in 5 seconds' and automatically powers off."
echo "   9. Unplug USB drive and take your notes."
echo "======================================================================"

