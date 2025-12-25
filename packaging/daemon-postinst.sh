#!/bin/sh
set -e

# ---- Stop any running old daemon ----
pkill -x odl_tb5_daemon 2>/dev/null || true
sleep 1
# Force-kill stragglers
pkill -9 -x odl_tb5_daemon 2>/dev/null || true

# ---- Reload udev rules ----
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger 2>/dev/null || true

# ---- Reload systemd user daemon for all logged-in users ----
systemctl --global daemon-reload 2>/dev/null || true

# ---- Restart daemon for the installing user ----
if [ -n "$SUDO_USER" ]; then
    DAEMON_UID=$(id -u "$SUDO_USER" 2>/dev/null) || true
    if [ -n "$DAEMON_UID" ] && [ -S "/run/user/${DAEMON_UID}/bus" ]; then
        su - "$SUDO_USER" -c \
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/${DAEMON_UID}/bus \
             systemctl --user restart odl-tb5-daemon.service" 2>/dev/null || true
    fi
fi
