/*
 * OdinLink TB5 System Tray Application
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>
#include <stdlib.h>

AppIndicator  *g_indicator = NULL;
GDBusProxy    *g_daemon_proxy = NULL;
GtkWidget     *g_peers_window = NULL;
GtkWidget     *g_test_window = NULL;
GtkWidget     *g_rccl_window = NULL;

static guint refresh_timer_id;

static gboolean on_refresh(gpointer user_data)
{
	(void)user_data;
	odl_tray_menu_refresh();
	return G_SOURCE_CONTINUE;
}

static void on_quit(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;
	gtk_main_quit();
}

int main(int argc, char *argv[])
{
	gtk_init(&argc, &argv);

	if (odl_tray_dbus_init() < 0) {
		g_printerr("odl_tb5_tray: failed to connect to daemon\n");
		g_printerr("odl_tb5_tray: is odl_tb5_daemon running?\n");
	}

	g_indicator = app_indicator_new(
		ODL_TRAY_APP_ID,
		"odl_tb5_disconnected",
		APP_INDICATOR_CATEGORY_HARDWARE);

	app_indicator_set_status(g_indicator, APP_INDICATOR_STATUS_ACTIVE);
	app_indicator_set_title(g_indicator, "OdinLink TB5");
	app_indicator_set_icon_theme_path(g_indicator,
		ODL_TRAY_ICON_DIR);

	odl_tray_menu_init();

	refresh_timer_id = g_timeout_add(2000, on_refresh, NULL);

	g_printerr("odl_tb5_tray: running\n");

	gtk_main();

	if (refresh_timer_id > 0)
		g_source_remove(refresh_timer_id);

	odl_tray_dbus_shutdown();

	g_printerr("odl_tb5_tray: stopped\n");
	return 0;
}
