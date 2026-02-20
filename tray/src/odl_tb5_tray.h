/*
 * OdinLink TB5 Tray Application - Main Internal Header
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_TRAY_H
#define ODL_TB5_TRAY_H

#include <gtk/gtk.h>
#include <libayatana-appindicator/app-indicator.h>
#include <gio/gio.h>
#include <stdbool.h>
#include <stdint.h>

#define ODL_TRAY_APP_ID     "com.odinlink.Tb5Tray"

#ifndef ODL_TRAY_ICON_DIR
#define ODL_TRAY_ICON_DIR   "/usr/share/icons/hicolor/scalable/apps"
#endif
#define ODL_DAEMON_BUS_NAME "com.odinlink.Tb5Daemon"
#define ODL_DAEMON_OBJ_PATH "/com/odinlink/Tb5Daemon"
#define ODL_DAEMON_IFACE    "com.odinlink.Tb5Daemon"

extern AppIndicator  *g_indicator;
extern GDBusProxy    *g_daemon_proxy;
extern GtkWidget     *g_peers_window;
extern GtkWidget     *g_test_window;
extern GtkWidget     *g_rccl_window;
extern GtkWidget     *g_overview_window;

/* D-Bus client (odl_tb5_tray_dbus.c) */
int  odl_tray_dbus_init(void);
void odl_tray_dbus_shutdown(void);
GVariant *odl_tray_dbus_call_sync(const char *method, GVariant *params);

/* Menu (odl_tb5_tray_menu.c) */
void odl_tray_menu_init(void);
void odl_tray_menu_refresh(void);

/* Peer list window (odl_tb5_tray_peers.c) */
void odl_tray_peers_show(int device_index);

/* Test runner dialog (odl_tb5_tray_tests.c) */
void odl_tray_tests_show(int device_index, const char *test_type);

/* Sync UI (odl_tb5_tray_sync.c) */
void odl_tray_sync_choose_folder(void);

/* RCCL stats window (odl_tb5_tray_rccl.c) */
void odl_tray_rccl_show(void);

/* Test overview dashboard (odl_tb5_tray_test_overview.c) */
void odl_tray_overview_show(void);

/* File browser window (odl_tb5_tray_files.c) */
extern GtkWidget *g_files_window;
void odl_tray_files_show(void);

#endif
