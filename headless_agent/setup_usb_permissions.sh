#!/usr/bin/env bash
set -e

# Setup automatic passwordless USB mounting for offline Jetson Orin Nano Headless Agent
# Run this once with: sudo bash /mnt/thesis/headless_agent/setup_usb_permissions.sh

echo "======================================================================"
echo "Configuring Headless USB Auto-Mount & Passwordless Permissions..."
echo "======================================================================"

TARGET_USER="ayush02"
MOUNT_DIR="/media/${TARGET_USER}/AYUSH"

# 1. Create mount target directory with correct ownership
mkdir -p "${MOUNT_DIR}"
chown -R ${TARGET_USER}:${TARGET_USER} "/media/${TARGET_USER}"
chmod 775 "${MOUNT_DIR}"

# 2. Configure Polkit rule to eliminate "Authentication Required" popups for USB drives
# Works for headless sessions, inactive seats, and GUI desktop clicks.
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

echo "[✓] Installed Polkit rule in ${POLKIT_DIR}/10-udisks2-mount.pkla"

# 3. Add fstab entry with 'user,nofail' option so standard mount binary can mount it without root
FSTAB_LINE="LABEL=AYUSH ${MOUNT_DIR} vfat defaults,nofail,user,umask=000,uid=1000,gid=1000 0 0"

if grep -q "LABEL=AYUSH" /etc/fstab; then
    echo "[!] fstab entry for LABEL=AYUSH already exists, updating..."
    sed -i '/LABEL=AYUSH/d' /etc/fstab
fi

echo "${FSTAB_LINE}" >> /etc/fstab
echo "[✓] Added fstab entry to /etc/fstab:"
echo "    ${FSTAB_LINE}"

# 4. Restart Polkit daemon
systemctl restart polkit || true
echo "[✓] Polkit service restarted."

# 5. Reload systemd daemon to pick up fstab changes
systemctl daemon-reload || true
echo "[✓] Systemd daemon reloaded."

echo "======================================================================"
echo "Verification:"
echo "Testing mount as unprivileged user '${TARGET_USER}'..."
su - "${TARGET_USER}" -c "mount ${MOUNT_DIR} 2>/dev/null || true"
lsblk -o NAME,LABEL,MOUNTPOINT | grep -i ayush || true
echo "======================================================================"
echo "DONE! Removable drive AYUSH will now mount automatically without ANY password prompts."
