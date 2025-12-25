/*
 * OdinLink TB5 Daemon - Device Monitor
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_MONITOR_H
#define ODL_TB5_DAEMON_MONITOR_H

#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_types.h>
#include <glib.h>
#include <stdbool.h>

#include "odl_tb5_daemon_sysinfo.h"

#define ODL_DAEMON_MAX_DEVICES 16

struct odl_daemon_device_slot {
	bool     present;
	int      index;
	uint32_t state;
	struct odl_tb5_peer_info peer;
	char     state_str[16];

	struct odl_sysinfo remote_sysinfo;
	bool              has_remote_sysinfo;
};

struct odl_daemon_device_table {
	struct odl_daemon_device_slot slots[ODL_DAEMON_MAX_DEVICES];
	int     connected_count;
	GMutex  lock;
};

/* Global device table */
extern struct odl_daemon_device_table g_device_table;

int  odl_daemon_monitor_init(void);
void odl_daemon_monitor_shutdown(void);

/* Get state string from enum value */
const char *odl_daemon_state_str(uint32_t state);

#endif
