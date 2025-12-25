/*
 * OdinLink TB5 Daemon - RCCL Stats Collector
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_RCCL_STATS_H
#define ODL_TB5_DAEMON_RCCL_STATS_H

#include <stdbool.h>
#include <glib.h>
#include <odl_tb5/odl_tb5_rccl_stats.h>

struct odl_daemon_rccl_cache {
	struct odl_rccl_stats stats;
	bool    available;
	GMutex  lock;
};

extern struct odl_daemon_rccl_cache g_rccl_cache;

int  odl_daemon_rccl_init(void);
void odl_daemon_rccl_shutdown(void);

char *odl_daemon_rccl_get_json(void);

#endif
