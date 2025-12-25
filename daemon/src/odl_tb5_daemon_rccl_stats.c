/*
 * OdinLink TB5 Daemon - RCCL Stats Collector
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_daemon_rccl_stats.h"
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

struct odl_daemon_rccl_cache g_rccl_cache;
static guint rccl_timer_id;

static gboolean odl_daemon_rccl_tick(gpointer user_data)
{
	(void)user_data;

	int fd = open(ODL_RCCL_STATS_PATH, O_RDONLY);
	if (fd < 0) {
		g_mutex_lock(&g_rccl_cache.lock);
		g_rccl_cache.available = false;
		g_mutex_unlock(&g_rccl_cache.lock);
		return G_SOURCE_CONTINUE;
	}

	struct stat st;
	if (fstat(fd, &st) < 0 || (size_t)st.st_size < sizeof(struct odl_rccl_stats)) {
		close(fd);
		g_mutex_lock(&g_rccl_cache.lock);
		g_rccl_cache.available = false;
		g_mutex_unlock(&g_rccl_cache.lock);
		return G_SOURCE_CONTINUE;
	}

	void *map = mmap(NULL, sizeof(struct odl_rccl_stats), PROT_READ,
	                 MAP_SHARED, fd, 0);
	close(fd);

	if (map == MAP_FAILED) {
		g_mutex_lock(&g_rccl_cache.lock);
		g_rccl_cache.available = false;
		g_mutex_unlock(&g_rccl_cache.lock);
		return G_SOURCE_CONTINUE;
	}

	struct odl_rccl_stats *stats = (struct odl_rccl_stats *)map;

	g_mutex_lock(&g_rccl_cache.lock);
	if (stats->magic == ODL_RCCL_STATS_MAGIC) {
		memcpy(&g_rccl_cache.stats, stats, sizeof(*stats));
		g_rccl_cache.available = true;
	} else {
		g_rccl_cache.available = false;
	}
	g_mutex_unlock(&g_rccl_cache.lock);

	munmap(map, sizeof(struct odl_rccl_stats));
	return G_SOURCE_CONTINUE;
}

int odl_daemon_rccl_init(void)
{
	g_mutex_init(&g_rccl_cache.lock);
	g_rccl_cache.available = false;

	odl_daemon_rccl_tick(NULL);

	rccl_timer_id = g_timeout_add(5000, odl_daemon_rccl_tick, NULL);

	return 0;
}

void odl_daemon_rccl_shutdown(void)
{
	if (rccl_timer_id > 0) {
		g_source_remove(rccl_timer_id);
		rccl_timer_id = 0;
	}
	g_mutex_clear(&g_rccl_cache.lock);
}

char *odl_daemon_rccl_get_json(void)
{
	g_mutex_lock(&g_rccl_cache.lock);

	if (!g_rccl_cache.available) {
		g_mutex_unlock(&g_rccl_cache.lock);
		return g_strdup("{\"active\": false}");
	}

	struct odl_rccl_stats *s = &g_rccl_cache.stats;

	uint64_t now_ns;
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;

	uint64_t uptime_sec = 0;
	if (now_ns > s->start_time_ns)
		uptime_sec = (now_ns - s->start_time_ns) / 1000000000ULL;

	char *json = g_strdup_printf(
		"{"
		"\"active\": %s, "
		"\"tx_bytes\": %lu, "
		"\"rx_bytes\": %lu, "
		"\"tx_ops\": %lu, "
		"\"rx_ops\": %lu, "
		"\"uptime_sec\": %lu"
		"}",
		s->active ? "true" : "false",
		(unsigned long)s->tx_bytes,
		(unsigned long)s->rx_bytes,
		(unsigned long)s->tx_ops,
		(unsigned long)s->rx_ops,
		(unsigned long)uptime_sec);

	g_mutex_unlock(&g_rccl_cache.lock);
	return json;
}
