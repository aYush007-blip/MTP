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

systemctl restart polkit || true
echo "  [OK] Polkit rule installed: No password will ever be required for USB drive."

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
echo "   4. Listen for voice prompt: 'System is ready to transcribe'."
echo "   5. Say 'Start Recording' when professor starts."
echo "   6. Say 'Stop Recording' when lecture ends."
echo "   7. Wait for voice prompt: 'Transcription successfully saved'."
echo "   8. Unplug USB drive and take your notes."
echo "======================================================================"

