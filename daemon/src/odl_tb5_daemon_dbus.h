/*
 * OdinLink TB5 Daemon - D-Bus Integration
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_DBUS_H
#define ODL_TB5_DAEMON_DBUS_H

#include <gio/gio.h>
#include <stdint.h>

/* Initialize D-Bus service on session bus. Returns 0 on success. */
int  odl_daemon_dbus_init(GMainLoop *loop);
void odl_daemon_dbus_shutdown(void);

/* Get the GDBus connection (for signal emission from other modules) */
GDBusConnection *odl_daemon_dbus_get_connection(void);

/* Emit signals */
void odl_daemon_dbus_emit_device_added(int index, const char *name);
void odl_daemon_dbus_emit_device_removed(int index);
void odl_daemon_dbus_emit_peer_state_changed(int index, const char *state);
void odl_daemon_dbus_emit_test_progress(const char *test_id, unsigned progress, const char *subtest);
void odl_daemon_dbus_emit_test_completed(const char *test_id, gboolean success, const char *summary);
void odl_daemon_dbus_emit_sync_file_transferred(const char *filename, const char *direction, uint64_t bytes);
void odl_daemon_dbus_emit_sync_conflict(const char *filename, const char *resolution);

#endif
