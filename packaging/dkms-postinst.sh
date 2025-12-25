#!/bin/sh
set -e

PACKAGE_NAME="odl-tb5"
PACKAGE_VERSION="@PROJECT_VERSION@"

# Stop daemon and tray for all logged-in users before module reload
for uid_dir in /run/user/*; do
    uid=$(basename "$uid_dir")
    if [ -S "${uid_dir}/bus" ]; then
        user=$(id -nu "$uid" 2>/dev/null) || continue
        su - "$user" -c \
            "DBUS_SESSION_BUS_ADDRESS=unix:path=${uid_dir}/bus \
             systemctl --user stop odl-tb5-daemon.service" 2>/dev/null || true
    fi
done
pkill -x odl_tb5_tray 2>/dev/null || true
pkill -x odl_tb5_daemon 2>/dev/null || true
sleep 1
pkill -9 -x odl_tb5_tray 2>/dev/null || true
pkill -9 -x odl_tb5_daemon 2>/dev/null || true

# Unload old module if present
rmmod odl_tb5 2>/dev/null || true

# Add and build the DKMS module
if [ -x /usr/sbin/dkms ]; then
    dkms add -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION" || true
    dkms build -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION" || true
    dkms install -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION" || true
fi

# Reload udev rules
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger 2>/dev/null || true

# Ensure module loads on every boot
echo "odl_tb5" > /etc/modules-load.d/odl_tb5.conf

# Load module now
modprobe odl_tb5 2>/dev/null || true
