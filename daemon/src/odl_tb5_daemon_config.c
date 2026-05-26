/*
 * OdinLink — Daemon: Configuration File Parser
 *
 * Reads and writes the daemon's config file (sync paths, log level,
 * auto-start settings). Uses GLib's key-value file format.
 * SPDX-License-Identifier: MIT
 */
#include "odl_tb5_daemon_config.h"
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

struct odl_daemon_config g_config;

static char *odl_daemon_config_path(void)
{
	return g_build_filename(g_get_user_config_dir(), "odl_tb5", "daemon.conf", NULL);
}

void odl_daemon_config_set_defaults(void)
{
	char *home = g_build_filename(g_get_home_dir(), "OdinLink-Shared", NULL);

	snprintf(g_config.sync_folder, sizeof(g_config.sync_folder), "%s", home);
	g_config.sync_enabled = false;
	g_config.monitor_interval_ms = 1000;
	g_config.rccl_stats_interval_ms = 5000;

	g_free(home);
}

int odl_daemon_config_load(void)
{
	GKeyFile *kf;
	char *path;
	GError *err = NULL;

	odl_daemon_config_set_defaults();

	path = odl_daemon_config_path();
	kf = g_key_file_new();

	if (!g_key_file_load_from_file(kf, path, G_KEY_FILE_NONE, &err)) {
		g_clear_error(&err);
		g_key_file_free(kf);
		g_free(path);
		g_mkdir_with_parents(g_config.sync_folder, 0755);
		return 0;
	}

	char *val;

	val = g_key_file_get_string(kf, "sync", "folder", NULL);
	if (val) {
		snprintf(g_config.sync_folder, sizeof(g_config.sync_folder), "%s", val);
		g_free(val);
	}

	g_config.sync_enabled = g_key_file_get_boolean(kf, "sync", "enabled", NULL);

	int iv;
	iv = g_key_file_get_integer(kf, "monitor", "interval_ms", &err);
	if (!err && iv > 0)
		g_config.monitor_interval_ms = iv;
	g_clear_error(&err);

	iv = g_key_file_get_integer(kf, "rccl", "stats_interval_ms", &err);
	if (!err && iv > 0)
		g_config.rccl_stats_interval_ms = iv;
	g_clear_error(&err);

	g_key_file_free(kf);
	g_free(path);

	g_mkdir_with_parents(g_config.sync_folder, 0755);

	return 0;
}

int odl_daemon_config_save(void)
{
	GKeyFile *kf;
	char *path, *dir;
	GError *err = NULL;

	path = odl_daemon_config_path();
	dir = g_path_get_dirname(path);
	g_mkdir_with_parents(dir, 0755);
	g_free(dir);

	kf = g_key_file_new();

	g_key_file_set_string(kf, "sync", "folder", g_config.sync_folder);
	g_key_file_set_boolean(kf, "sync", "enabled", g_config.sync_enabled);
	g_key_file_set_integer(kf, "monitor", "interval_ms", g_config.monitor_interval_ms);
	g_key_file_set_integer(kf, "rccl", "stats_interval_ms", g_config.rccl_stats_interval_ms);

	if (!g_key_file_save_to_file(kf, path, &err)) {
		g_printerr("odl_tb5_daemon: failed to save config: %s\n", err->message);
		g_error_free(err);
		g_key_file_free(kf);
		g_free(path);
		return -1;
	}

	g_key_file_free(kf);
	g_free(path);
	return 0;
}
