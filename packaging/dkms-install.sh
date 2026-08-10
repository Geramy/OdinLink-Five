#!/bin/sh
# OdinLink-Five DKMS Install Script
# Run from the repository root:  sudo sh packaging/dkms-install.sh
set -e

PACKAGE_NAME="odl-tb5"
PACKAGE_VERSION="${1:-0.1.0}"
SRC_DIR="/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}"

# Stop anything using the module
for uid_dir in /run/user/*; do
    uid=$(basename "$uid_dir")
    [ -S "${uid_dir}/bus" ] || continue
    user=$(id -nu "$uid" 2>/dev/null) || continue
    su - "$user" -c "DBUS_SESSION_BUS_ADDRESS=unix:path=${uid_dir}/bus \
        systemctl --user stop odl-tb5-daemon.service" 2>/dev/null || true
done
pkill -9 -x odl_tb5_tray 2>/dev/null || true
pkill -9 -x odl_tb5_daemon 2>/dev/null || true
rmmod odl_tb5 2>/dev/null || true

# Remove old DKMS source
rm -rf "$SRC_DIR"

# Copy driver source
mkdir -p "$SRC_DIR/uapi"
cp driver/odl_tb5_service.c     "$SRC_DIR/"
cp driver/odl_tb5_chardev.c     "$SRC_DIR/"
cp driver/odl_tb5_ring_dma.c    "$SRC_DIR/"
cp driver/odl_tb5_proto.c       "$SRC_DIR/"
cp driver/odl_tb5_loopback.c    "$SRC_DIR/"
cp driver/odl_tb5_core.h        "$SRC_DIR/"
cp driver/Kbuild                "$SRC_DIR/"
cp driver/Makefile              "$SRC_DIR/"
cp driver/uapi/odl_tb5_uapi.h   "$SRC_DIR/uapi/"

# Substitute @PROJECT_VERSION@ into dkms.conf and the postinst/prerm helpers.
# (Copying the helpers verbatim left the placeholder literal and made DKMS
# look for /usr/src/odl-tb5-@PROJECT_VERSION@ — issue #23.)
sed "s/@PROJECT_VERSION@/$PACKAGE_VERSION/g" packaging/dkms.conf > "$SRC_DIR/dkms.conf"
sed "s/@PROJECT_VERSION@/$PACKAGE_VERSION/g" packaging/dkms-postinst.sh > "$SRC_DIR/dkms-postinst.sh"
sed "s/@PROJECT_VERSION@/$PACKAGE_VERSION/g" packaging/dkms-prerm.sh > "$SRC_DIR/dkms-prerm.sh"
chmod +x "$SRC_DIR/dkms-postinst.sh" "$SRC_DIR/dkms-prerm.sh"

# Register with DKMS
if [ -x /usr/sbin/dkms ]; then
    dkms add -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION"
    dkms build -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION"
    dkms install -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION"
else
    echo "ERROR: dkms not found. Install it: sudo apt install dkms"
    exit 1
fi

# Install udev rule and load module
cp driver/71-odl-tb5.rules /etc/udev/rules.d/
udevadm control --reload-rules
modprobe odl_tb5

# Enable on boot
echo "odl_tb5" > /etc/modules-load.d/odl_tb5.conf

echo ""
echo "OdinLink-Five DKMS installation complete."
echo "Module loaded. Use: lsmod | grep odl_tb5"
