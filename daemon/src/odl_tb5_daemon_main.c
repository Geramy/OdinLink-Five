/*
 * OdinLink TB5 Daemon
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_daemon_dbus.h"
#include "odl_tb5_daemon_monitor.h"
#include "odl_tb5_daemon_config.h"
#include "odl_tb5_daemon_rccl_stats.h"
#include "odl_tb5_daemon_test.h"
#include "odl_tb5_daemon_sync.h"
#include <glib.h>
#include <glib-unix.h>
#include <stdio.h>
#include <stdlib.h>

static GMainLoop *main_loop;

static gboolean on_signal(gpointer user_data)
{
	(void)user_data;
	g_printerr("odl_tb5_daemon: shutting down...\n");
	g_main_loop_quit(main_loop);
	return G_SOURCE_REMOVE;
}

int main(int argc, char *argv[])
{
	gboolean foreground = FALSE;
	GOptionEntry entries[] = {
		{ "foreground", 'f', 0, G_OPTION_ARG_NONE, &foreground,
		  "Run in foreground (don't daemonize)", NULL },
		{ NULL }
	};

	GError *err = NULL;
	GOptionContext *ctx = g_option_context_new("- OdinLink TB5 Daemon");
	g_option_context_add_main_entries(ctx, entries, NULL);
	if (!g_option_context_parse(ctx, &argc, &argv, &err)) {
		g_printerr("Option parsing failed: %s\n", err->message);
		g_error_free(err);
		g_option_context_free(ctx);
		return 1;
	}
	g_option_context_free(ctx);

	g_printerr("odl_tb5_daemon: starting...\n");

	odl_daemon_config_load();

	main_loop = g_main_loop_new(NULL, FALSE);

	g_unix_signal_add(SIGTERM, on_signal, NULL);
	g_unix_signal_add(SIGINT, on_signal, NULL);

	int ret;

	ret = odl_daemon_dbus_init(main_loop);
	if (ret < 0) {
		g_printerr("odl_tb5_daemon: D-Bus init failed\n");
		g_main_loop_unref(main_loop);
		return 1;
	}

	odl_daemon_rccl_init();

	odl_daemon_test_init();

	odl_daemon_monitor_init();

	if (g_config.sync_enabled)
		odl_daemon_sync_init(g_config.sync_folder);

	g_printerr("odl_tb5_daemon: running\n");

	g_main_loop_run(main_loop);

	odl_daemon_sync_shutdown();
	odl_daemon_test_shutdown();
	odl_daemon_rccl_shutdown();
	odl_daemon_monitor_shutdown();
	odl_daemon_dbus_shutdown();

	odl_daemon_config_save();

	g_main_loop_unref(main_loop);
	g_printerr("odl_tb5_daemon: stopped\n");

	return 0;
}
