#!/bin/sh
set -e

PACKAGE_NAME="odl-tb5"
PACKAGE_VERSION="@PROJECT_VERSION@"

# Stop daemon and tray for all logged-in users (they hold the module open)
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
pkill -x odl_tb5_cli 2>/dev/null || true
sleep 1
pkill -9 -x odl_tb5_tray 2>/dev/null || true
pkill -9 -x odl_tb5_daemon 2>/dev/null || true
pkill -9 -x odl_tb5_cli 2>/dev/null || true

# Unload module — retry up to 5 times waiting for refcount 0
for i in 1 2 3 4 5; do
    rmmod odl_tb5 2>/dev/null && break
    sleep 1
done

# Remove boot autoload config
rm -f /etc/modules-load.d/odl_tb5.conf

# Remove DKMS module
if [ -x /usr/sbin/dkms ]; then
    dkms remove -m "$PACKAGE_NAME" -v "$PACKAGE_VERSION" --all 2>/dev/null || true
fi

# Reload udev rules after our rules file is about to be removed
udevadm control --reload-rules 2>/dev/null || true
