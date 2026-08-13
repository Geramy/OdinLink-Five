/*
 * OdinLink — Daemon: Watches for Cable Plug/Unplug Events
 *
 * Polls /dev/odl_tb5_N periodically and emits D-Bus signals when
 * a peer connects or disconnects. Also tracks link state transitions
 * (handshake → connected → ready).
 */
#include "odl_tb5_daemon_monitor.h"
#include "odl_tb5_daemon_dbus.h"
#include "odl_tb5_daemon_test.h"
#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_ioctl.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

struct odl_daemon_device_table g_device_table;
static guint monitor_timer_id;

const char *odl_daemon_state_str(uint32_t state)
{
	switch (state) {
	case ODL_TB5_STATE_DISCONNECTED: return "disconnected";
	case ODL_TB5_STATE_HANDSHAKE:    return "handshake";
	case ODL_TB5_STATE_CONNECTED:    return "connected";
	case ODL_TB5_STATE_ERROR:        return "error";
	case ODL_TB5_STATE_READY:        return "ready";
	default:                         return "unknown";
	}
}

static void format_uuid(const uint8_t uuid[16], char *buf, size_t len)
{
	snprintf(buf, len,
		"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
		uuid[0], uuid[1], uuid[2], uuid[3],
		uuid[4], uuid[5], uuid[6], uuid[7],
		uuid[8], uuid[9], uuid[10], uuid[11],
		uuid[12], uuid[13], uuid[14], uuid[15]);
}

static gboolean odl_daemon_monitor_tick(gpointer user_data)
{
	(void)user_data;

	g_mutex_lock(&g_device_table.lock);

	int connected = 0;

	for (int i = 0; i < ODL_DAEMON_MAX_DEVICES; i++) {
		char path[64];
		snprintf(path, sizeof(path), "/dev/%s_%d", ODL_TB5_DEVICE_NAME, i);

		bool was_present = g_device_table.slots[i].present;
		uint32_t old_state = g_device_table.slots[i].state;

		if (access(path, F_OK) != 0) {
			if (was_present) {
				g_printerr("monitor: device %d (%s) "
					   "disappeared (was %s)\n",
					   i, path,
					   odl_daemon_state_str(old_state));

				g_device_table.slots[i].present = false;
				g_device_table.slots[i].state = ODL_TB5_STATE_DISCONNECTED;
				g_device_table.slots[i].has_remote_sysinfo = false;
				g_device_table.slots[i].open_warned = false;
				memset(&g_device_table.slots[i].peer, 0,
				       sizeof(g_device_table.slots[i].peer));
				snprintf(g_device_table.slots[i].state_str,
				         sizeof(g_device_table.slots[i].state_str),
				         "disconnected");

				g_mutex_unlock(&g_device_table.lock);
				odl_daemon_server_stop_for_device(i);
				odl_daemon_dbus_emit_device_removed(i);
				g_mutex_lock(&g_device_table.lock);
			}
			continue;
		}

		odl_tb5_t handle;
		int ret = odl_tb5_open(&handle, i);
		if (ret < 0) {
			if (!g_device_table.slots[i].open_warned) {
				g_printerr("monitor: %s exists but cannot open: %s "
					   "(install driver/71-odl-tb5.rules or "
					   "chmod 660)\n",
					   path, strerror(-ret));
				g_device_table.slots[i].open_warned = true;
			}
			continue;
		}
		g_device_table.slots[i].open_warned = false;

		struct odl_tb5_peer_info info;
		ret = odl_tb5_get_peer(handle, &info);
		odl_tb5_close(handle);

		if (ret < 0)
			continue;

		g_device_table.slots[i].present = true;
		g_device_table.slots[i].index = i;
		g_device_table.slots[i].state = info.state;
		memcpy(&g_device_table.slots[i].peer, &info, sizeof(info));
		snprintf(g_device_table.slots[i].state_str,
		         sizeof(g_device_table.slots[i].state_str),
		         "%s", odl_daemon_state_str(info.state));

		if (info.state == ODL_TB5_STATE_CONNECTED ||
		    info.state == ODL_TB5_STATE_READY)
			connected++;

		if (!was_present) {
			char name[128];
			if (info.device_name[0])
				snprintf(name, sizeof(name), "%s", info.device_name);
			else
				snprintf(name, sizeof(name), "Device %d", i);

			g_printerr("monitor: device %d appeared: "
				   "name=\"%s\", state=%s\n",
				   i, name,
				   odl_daemon_state_str(info.state));

			g_mutex_unlock(&g_device_table.lock);
			odl_daemon_server_start_for_device(i);
			odl_daemon_dbus_emit_device_added(i, name);
			g_mutex_lock(&g_device_table.lock);
		} else if (info.state != old_state) {
			const char *state_str = odl_daemon_state_str(info.state);

			g_printerr("monitor: device %d state: %s -> %s\n",
				   i, odl_daemon_state_str(old_state),
				   state_str);

			g_mutex_unlock(&g_device_table.lock);
			odl_daemon_dbus_emit_peer_state_changed(i, state_str);
			g_mutex_lock(&g_device_table.lock);
		}
	}

	g_device_table.connected_count = connected;
	g_mutex_unlock(&g_device_table.lock);

	return G_SOURCE_CONTINUE;
}

int odl_daemon_monitor_init(void)
{
	g_mutex_init(&g_device_table.lock);
	memset(g_device_table.slots, 0, sizeof(g_device_table.slots));
	g_device_table.connected_count = 0;

	odl_daemon_monitor_tick(NULL);

	monitor_timer_id = g_timeout_add(1000, odl_daemon_monitor_tick, NULL);

	return 0;
}

void odl_daemon_monitor_shutdown(void)
{
	if (monitor_timer_id > 0) {
		g_source_remove(monitor_timer_id);
		monitor_timer_id = 0;
	}
	g_mutex_clear(&g_device_table.lock);
}
