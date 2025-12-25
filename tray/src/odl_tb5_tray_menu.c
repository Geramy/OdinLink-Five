/*
 * OdinLink TB5 Tray - AppIndicator Menu
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static GtkWidget *s_menu           = NULL;
static GtkWidget *s_peers_submenu  = NULL;
static GtkWidget *s_sync_status    = NULL;
static GtkWidget *s_sync_folder    = NULL;
static GtkWidget *s_sync_toggle    = NULL;

static gboolean s_sync_enabled = FALSE;

static guint s_peers_fingerprint = 0;

static void on_quit(GtkMenuItem *item, gpointer user_data);
static void on_about(GtkMenuItem *item, gpointer user_data);
static void on_rccl_stats(GtkMenuItem *item, gpointer user_data);
static void on_browse_files(GtkMenuItem *item, gpointer user_data);
static void on_change_folder(GtkMenuItem *item, gpointer user_data);
static void on_toggle_sync(GtkMenuItem *item, gpointer user_data);
static void on_open_folder(GtkMenuItem *item, gpointer user_data);
static void on_test_item(GtkMenuItem *item, gpointer user_data);
static void on_test_all(GtkMenuItem *item, gpointer user_data);
static void on_peer_details(GtkMenuItem *item, gpointer user_data);

static const char *test_types[] = {
	"bandwidth",
	"latency",
	"jitter",
	"latency_under_load",
	"mimo",
};
#define NUM_TESTS (sizeof(test_types) / sizeof(test_types[0]))

/* Create a disabled label menu item */
static GtkWidget *make_label_item(const char *text)
{
	GtkWidget *item = gtk_menu_item_new_with_label(text);
	gtk_widget_set_sensitive(item, FALSE);
	return item;
}

static GtkWidget *build_peers_submenu(void)
{
	GtkWidget *submenu = gtk_menu_new();

	GtkWidget *placeholder = make_label_item("(scanning...)");
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), placeholder);

	return submenu;
}

static GtkWidget *build_tests_submenu(void)
{
	GtkWidget *submenu = gtk_menu_new();
	GtkWidget *item;

	item = gtk_menu_item_new_with_label("Bandwidth Test");
	g_signal_connect(item, "activate",
	                 G_CALLBACK(on_test_item), (gpointer)test_types[0]);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	item = gtk_menu_item_new_with_label("Latency Test");
	g_signal_connect(item, "activate",
	                 G_CALLBACK(on_test_item), (gpointer)test_types[1]);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	item = gtk_menu_item_new_with_label("Jitter Test");
	g_signal_connect(item, "activate",
	                 G_CALLBACK(on_test_item), (gpointer)test_types[2]);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	item = gtk_menu_item_new_with_label("Latency Under Load");
	g_signal_connect(item, "activate",
	                 G_CALLBACK(on_test_item), (gpointer)test_types[3]);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	item = gtk_menu_item_new_with_label("MIMO Test");
	g_signal_connect(item, "activate",
	                 G_CALLBACK(on_test_item), (gpointer)test_types[4]);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	gtk_menu_shell_append(GTK_MENU_SHELL(submenu),
	                      gtk_separator_menu_item_new());

	item = gtk_menu_item_new_with_label("Run All Tests");
	g_signal_connect(item, "activate", G_CALLBACK(on_test_all), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	return submenu;
}

static GtkWidget *build_sync_submenu(void)
{
	GtkWidget *submenu = gtk_menu_new();

	s_sync_status = make_label_item("Status: unknown");
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), s_sync_status);

	s_sync_folder = make_label_item("Folder: (none)");
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), s_sync_folder);

	gtk_menu_shell_append(GTK_MENU_SHELL(submenu),
	                      gtk_separator_menu_item_new());

	GtkWidget *item;

	item = gtk_menu_item_new_with_label("Change Folder...");
	g_signal_connect(item, "activate", G_CALLBACK(on_change_folder), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	s_sync_toggle = gtk_menu_item_new_with_label("Enable Sync");
	g_signal_connect(s_sync_toggle, "activate",
	                 G_CALLBACK(on_toggle_sync), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), s_sync_toggle);

	item = gtk_menu_item_new_with_label("Open Folder");
	g_signal_connect(item, "activate", G_CALLBACK(on_open_folder), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(submenu), item);

	return submenu;
}

void odl_tray_menu_init(void)
{
	s_menu = gtk_menu_new();

	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu),
	                      make_label_item("OdinLink TB5"));

	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu),
	                      gtk_separator_menu_item_new());

	GtkWidget *peers_item = gtk_menu_item_new_with_label("Peers");
	s_peers_submenu = build_peers_submenu();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(peers_item), s_peers_submenu);
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), peers_item);

	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu),
	                      gtk_separator_menu_item_new());

	GtkWidget *tests_item = gtk_menu_item_new_with_label("Run Tests");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(tests_item),
	                          build_tests_submenu());
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), tests_item);

	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu),
	                      gtk_separator_menu_item_new());

	GtkWidget *sync_item = gtk_menu_item_new_with_label("Shared Folder");
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(sync_item),
	                          build_sync_submenu());
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), sync_item);

	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu),
	                      gtk_separator_menu_item_new());

	GtkWidget *files_item = gtk_menu_item_new_with_label("Browse Files");
	g_signal_connect(files_item, "activate",
	                 G_CALLBACK(on_browse_files), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), files_item);

	GtkWidget *rccl_item = gtk_menu_item_new_with_label("RCCL Stats");
	g_signal_connect(rccl_item, "activate",
	                 G_CALLBACK(on_rccl_stats), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), rccl_item);

	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu),
	                      gtk_separator_menu_item_new());

	GtkWidget *about_item = gtk_menu_item_new_with_label("About");
	g_signal_connect(about_item, "activate", G_CALLBACK(on_about), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), about_item);

	GtkWidget *quit_item = gtk_menu_item_new_with_label("Quit");
	g_signal_connect(quit_item, "activate", G_CALLBACK(on_quit), NULL);
	gtk_menu_shell_append(GTK_MENU_SHELL(s_menu), quit_item);

	gtk_widget_show_all(s_menu);
	app_indicator_set_menu(g_indicator, GTK_MENU(s_menu));
}

void odl_tray_menu_refresh(void)
{
	if (!s_menu)
		return;

	int device_count = 0;
	guint new_fingerprint = 0;
	GVariant *devices = odl_tray_dbus_call_sync("GetDevices", NULL);

	if (devices) {
		gconstpointer data = g_variant_get_data(devices);
		gsize size = g_variant_get_size(devices);
		if (data && size > 0) {
			guint h = 5381;
			const guint8 *p = data;
			for (gsize i = 0; i < size; i++)
				h = ((h << 5) + h) + p[i];
			new_fingerprint = h;
		}
	}

	if (new_fingerprint != s_peers_fingerprint) {
		s_peers_fingerprint = new_fingerprint;

		GList *children = gtk_container_get_children(
			GTK_CONTAINER(s_peers_submenu));
		for (GList *l = children; l; l = l->next)
			gtk_widget_destroy(GTK_WIDGET(l->data));
		g_list_free(children);

		if (devices) {
			GVariantIter *iter = NULL;
			g_variant_get(devices, "(a(iss))", &iter);

			gint32 dev_index;
			const gchar *state_str = NULL;
			const gchar *dev_name = NULL;

			while (g_variant_iter_loop(iter, "(iss)",
			                           &dev_index, &state_str,
			                           &dev_name)) {
				device_count++;

				char label_buf[256];
				snprintf(label_buf, sizeof(label_buf),
				         "%d: %s [%s]",
				         dev_index,
				         dev_name ? dev_name : "Unknown",
				         state_str ? state_str : "?");

				GtkWidget *dev_item =
					gtk_menu_item_new_with_label(label_buf);
				GtkWidget *dev_sub = gtk_menu_new();

				GtkWidget *details_item =
					gtk_menu_item_new_with_label("Details...");
				g_signal_connect(details_item, "activate",
				                 G_CALLBACK(on_peer_details),
				                 GINT_TO_POINTER(dev_index));
				gtk_menu_shell_append(GTK_MENU_SHELL(dev_sub),
				                      details_item);

				gtk_menu_item_set_submenu(
					GTK_MENU_ITEM(dev_item), dev_sub);
				gtk_menu_shell_append(
					GTK_MENU_SHELL(s_peers_submenu),
					dev_item);
			}

			g_variant_iter_free(iter);
		}

		if (device_count == 0) {
			GtkWidget *none_item =
				make_label_item("No devices found");
			gtk_menu_shell_append(
				GTK_MENU_SHELL(s_peers_submenu), none_item);
		}

		gtk_widget_show_all(s_peers_submenu);
	} else if (devices) {
		GVariantIter *iter = NULL;
		g_variant_get(devices, "(a(iss))", &iter);
		gint32 idx;
		const gchar *s = NULL, *n = NULL;
		while (g_variant_iter_loop(iter, "(iss)", &idx, &s, &n))
			device_count++;
		g_variant_iter_free(iter);
	}

	if (devices)
		g_variant_unref(devices);

	GVariant *sync_status = odl_tray_dbus_call_sync("GetSyncStatus", NULL);
	if (sync_status) {
		gboolean enabled;
		guint32 files_pending;
		guint64 bytes_transferred;
		const gchar *last_sync = NULL;

		g_variant_get(sync_status, "(buts)",
		              &enabled, &files_pending,
		              &bytes_transferred, &last_sync);

		s_sync_enabled = enabled;

		char status_buf[128];
		if (enabled) {
			snprintf(status_buf, sizeof(status_buf),
			         "Status: active (%u pending)", files_pending);
		} else {
			snprintf(status_buf, sizeof(status_buf),
			         "Status: disabled");
		}
		gtk_menu_item_set_label(GTK_MENU_ITEM(s_sync_status), status_buf);

		gtk_menu_item_set_label(GTK_MENU_ITEM(s_sync_toggle),
		                        enabled ? "Disable Sync" : "Enable Sync");

		g_variant_unref(sync_status);
	}

	GVariant *folder_result = odl_tray_dbus_call_sync("GetSyncFolder", NULL);
	if (folder_result) {
		const gchar *path = NULL;
		g_variant_get(folder_result, "(s)", &path);

		char folder_buf[256];
		if (path && path[0]) {
			snprintf(folder_buf, sizeof(folder_buf),
			         "Folder: %s", path);
		} else {
			snprintf(folder_buf, sizeof(folder_buf),
			         "Folder: (none)");
		}
		gtk_menu_item_set_label(GTK_MENU_ITEM(s_sync_folder), folder_buf);
		g_variant_unref(folder_result);
	}

	if (!g_daemon_proxy) {
		app_indicator_set_icon(g_indicator, "odl_tb5_error");
	} else if (s_sync_enabled && device_count > 0) {
		app_indicator_set_icon(g_indicator, "odl_tb5_syncing");
	} else if (device_count > 0) {
		app_indicator_set_icon(g_indicator, "odl_tb5_connected");
	} else {
		app_indicator_set_icon(g_indicator, "odl_tb5_disconnected");
	}
}

static void on_quit(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;
	gtk_main_quit();
}

static void on_about(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;

	const char *daemon_version = "unknown";
	GVariant *ver = odl_tray_dbus_call_sync("GetVersion", NULL);
	char ver_buf[64] = {0};
	if (ver) {
		const gchar *v = NULL;
		g_variant_get(ver, "(s)", &v);
		if (v) {
			snprintf(ver_buf, sizeof(ver_buf), "%s", v);
			daemon_version = ver_buf;
		}
		g_variant_unref(ver);
	}

	GtkWidget *dialog = gtk_message_dialog_new(
		NULL,
		GTK_DIALOG_DESTROY_WITH_PARENT,
		GTK_MESSAGE_INFO,
		GTK_BUTTONS_OK,
		"OdinLink TB5 Tray\n\n"
		"System tray application for monitoring and managing\n"
		"Thunderbolt 5 peer-to-peer connections.\n\n"
		"Daemon version: %s\n\n"
		"Copyright (c) 2025-2026 OdinLink Project\n"
		"License: MIT",
		daemon_version);

	gtk_window_set_title(GTK_WINDOW(dialog), "About OdinLink TB5");
	gtk_dialog_run(GTK_DIALOG(dialog));
	gtk_widget_destroy(dialog);
}

static void on_rccl_stats(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;
	odl_tray_rccl_show();
}

static void on_browse_files(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;
	odl_tray_files_show();
}

static void on_change_folder(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;
	odl_tray_sync_choose_folder();
}

static void on_toggle_sync(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;

	gboolean new_state = !s_sync_enabled;

	odl_tray_dbus_call_sync("SetSyncEnabled",
	                        g_variant_new("(b)", new_state));

	g_printerr("odl_tb5_tray: sync %s\n",
	           new_state ? "enabled" : "disabled");

	odl_tray_menu_refresh();
}

static void on_open_folder(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;

	GVariant *result = odl_tray_dbus_call_sync("GetSyncFolder", NULL);
	if (!result)
		return;

	const gchar *path = NULL;
	g_variant_get(result, "(s)", &path);

	if (path && path[0]) {
		char *cmd = g_strdup_printf("xdg-open \"%s\"", path);
		int ret = system(cmd);
		if (ret != 0)
			g_printerr("odl_tb5_tray: xdg-open failed with %d\n", ret);
		g_free(cmd);
	} else {
		GtkWidget *dialog = gtk_message_dialog_new(
			NULL,
			GTK_DIALOG_DESTROY_WITH_PARENT,
			GTK_MESSAGE_WARNING,
			GTK_BUTTONS_OK,
			"No shared folder is configured.\n\n"
			"Use 'Change Folder...' to set one.");
		gtk_window_set_title(GTK_WINDOW(dialog), "OdinLink TB5");
		gtk_dialog_run(GTK_DIALOG(dialog));
		gtk_widget_destroy(dialog);
	}

	g_variant_unref(result);
}

static int get_first_device_index(void)
{
	GVariant *devices = odl_tray_dbus_call_sync("GetDevices", NULL);
	if (!devices)
		return -1;

	GVariantIter *iter = NULL;
	g_variant_get(devices, "(a(iss))", &iter);

	gint32 dev_index = -1;
	const gchar *s = NULL, *n = NULL;
	g_variant_iter_next(iter, "(i&s&s)", &dev_index, &s, &n);

	g_variant_iter_free(iter);
	g_variant_unref(devices);

	return (int)dev_index;
}

static void on_test_item(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	const char *test_type = (const char *)user_data;
	int dev_idx = get_first_device_index();
	if (dev_idx < 0) {
		g_printerr("odl_tb5_tray: no devices available for testing\n");
		return;
	}
	odl_tray_tests_show(dev_idx, test_type);
}

static void on_test_all(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	(void)user_data;
	int dev_idx = get_first_device_index();
	if (dev_idx < 0) {
		g_printerr("odl_tb5_tray: no devices available for testing\n");
		return;
	}
	odl_tray_tests_show(dev_idx, "all");
}

static void on_peer_details(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	int device_index = GPOINTER_TO_INT(user_data);
	odl_tray_peers_show(device_index);
}
