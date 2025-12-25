/*
 * OdinLink TB5 Tray - D-Bus Proxy Client
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>

/* D-Bus signal dispatcher */
static void on_daemon_signal(GDBusProxy  *proxy,
                             const gchar *sender_name,
                             const gchar *signal_name,
                             GVariant    *parameters,
                             gpointer     user_data)
{
	(void)proxy;
	(void)sender_name;
	(void)user_data;

	if (g_strcmp0(signal_name, "DeviceAdded") == 0) {
		gint32 index;
		const gchar *name = NULL;
		g_variant_get(parameters, "(is)", &index, &name);
		g_printerr("odl_tb5_tray: signal DeviceAdded index=%d name=%s\n",
		           index, name ? name : "(null)");
		odl_tray_menu_refresh();

	} else if (g_strcmp0(signal_name, "DeviceRemoved") == 0) {
		gint32 index;
		g_variant_get(parameters, "(i)", &index);
		g_printerr("odl_tb5_tray: signal DeviceRemoved index=%d\n", index);
		odl_tray_menu_refresh();

	} else if (g_strcmp0(signal_name, "PeerStateChanged") == 0) {
		gint32 index;
		const gchar *state = NULL;
		g_variant_get(parameters, "(is)", &index, &state);
		g_printerr("odl_tb5_tray: signal PeerStateChanged index=%d state=%s\n",
		           index, state ? state : "(null)");
		odl_tray_menu_refresh();

	} else if (g_strcmp0(signal_name, "TestProgress") == 0) {
		const gchar *test_id = NULL;
		guint32 progress;
		const gchar *subtest = NULL;
		g_variant_get(parameters, "(sus)", &test_id, &progress, &subtest);
		g_printerr("odl_tb5_tray: signal TestProgress id=%s progress=%u%% subtest=%s\n",
		           test_id ? test_id : "(null)", progress,
		           subtest ? subtest : "(null)");

	} else if (g_strcmp0(signal_name, "TestCompleted") == 0) {
		const gchar *test_id = NULL;
		gboolean success;
		const gchar *summary = NULL;
		g_variant_get(parameters, "(sbs)", &test_id, &success, &summary);
		g_printerr("odl_tb5_tray: signal TestCompleted id=%s success=%d summary=%s\n",
		           test_id ? test_id : "(null)", success,
		           summary ? summary : "(null)");

	} else if (g_strcmp0(signal_name, "SyncFileTransferred") == 0) {
		const gchar *filename = NULL;
		const gchar *direction = NULL;
		guint64 bytes;
		g_variant_get(parameters, "(sst)", &filename, &direction, &bytes);
		g_printerr("odl_tb5_tray: signal SyncFileTransferred file=%s dir=%s bytes=%lu\n",
		           filename ? filename : "(null)",
		           direction ? direction : "(null)",
		           (unsigned long)bytes);

	} else if (g_strcmp0(signal_name, "SyncConflict") == 0) {
		const gchar *filename = NULL;
		const gchar *resolution = NULL;
		g_variant_get(parameters, "(ss)", &filename, &resolution);
		g_printerr("odl_tb5_tray: signal SyncConflict file=%s resolution=%s\n",
		           filename ? filename : "(null)",
		           resolution ? resolution : "(null)");

	} else {
		g_printerr("odl_tb5_tray: unknown signal %s\n", signal_name);
	}
}

int odl_tray_dbus_init(void)
{
	GError *error = NULL;

	g_daemon_proxy = g_dbus_proxy_new_for_bus_sync(
		G_BUS_TYPE_SESSION,
		G_DBUS_PROXY_FLAGS_NONE,
		NULL,
		ODL_DAEMON_BUS_NAME,
		ODL_DAEMON_OBJ_PATH,
		ODL_DAEMON_IFACE,
		NULL,
		&error);

	if (!g_daemon_proxy) {
		g_printerr("odl_tb5_tray: D-Bus proxy creation failed: %s\n",
		           error ? error->message : "unknown error");
		if (error)
			g_error_free(error);
		return -1;
	}

	g_signal_connect(g_daemon_proxy, "g-signal",
	                 G_CALLBACK(on_daemon_signal), NULL);

	g_printerr("odl_tb5_tray: connected to %s on session bus\n",
	           ODL_DAEMON_BUS_NAME);
	return 0;
}

void odl_tray_dbus_shutdown(void)
{
	if (g_daemon_proxy) {
		g_object_unref(g_daemon_proxy);
		g_daemon_proxy = NULL;
	}
}

GVariant *odl_tray_dbus_call_sync(const char *method, GVariant *params)
{
	GError *error = NULL;
	GVariant *result;

	if (!g_daemon_proxy) {
		g_printerr("odl_tb5_tray: dbus_call_sync(%s): no proxy\n", method);
		return NULL;
	}

	result = g_dbus_proxy_call_sync(
		g_daemon_proxy,
		method,
		params,
		G_DBUS_CALL_FLAGS_NONE,
		5000,
		NULL,
		&error);

	if (!result) {
		g_printerr("odl_tb5_tray: dbus_call_sync(%s) failed: %s\n",
		           method, error ? error->message : "unknown error");
		if (error)
			g_error_free(error);
		return NULL;
	}

	return result;
}
