/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2025-2026 OdinLink Project */
#ifndef ODL_TB5_DAEMON_CONFIG_H
#define ODL_TB5_DAEMON_CONFIG_H

#include <stdbool.h>

struct odl_daemon_config {
	char sync_folder[512];
	bool sync_enabled;
	int  monitor_interval_ms;
	int  rccl_stats_interval_ms;
};

extern struct odl_daemon_config g_config;

int  odl_daemon_config_load(void);
int  odl_daemon_config_save(void);
void odl_daemon_config_set_defaults(void);

#endif
