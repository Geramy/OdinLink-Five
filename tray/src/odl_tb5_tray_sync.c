/*
 * OdinLink TB5 Tray - Sync Folder UI
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>

void odl_tray_sync_choose_folder(void)
{
	GtkWidget *dialog = gtk_file_chooser_dialog_new(
		"Choose OdinLink Shared Folder",
		NULL,
		GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
		"_Cancel", GTK_RESPONSE_CANCEL,
		"_Select", GTK_RESPONSE_ACCEPT,
		NULL);

	GVariant *result = odl_tray_dbus_call_sync("GetSyncFolder", NULL);
	if (result) {
		const char *current_path = NULL;
		g_variant_get(result, "(s)", &current_path);
		if (current_path && current_path[0])
			gtk_file_chooser_set_current_folder(
				GTK_FILE_CHOOSER(dialog), current_path);
		g_variant_unref(result);
	}

	if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
		char *folder = gtk_file_chooser_get_filename(
			GTK_FILE_CHOOSER(dialog));

		if (folder) {
			GVariant *ret = odl_tray_dbus_call_sync(
				"SetSyncFolder",
				g_variant_new("(s)", folder));

			if (ret) {
				gboolean success;
				g_variant_get(ret, "(b)", &success);
				g_variant_unref(ret);

				if (success)
					g_printerr("odl_tb5_tray: sync folder set to %s\n",
					           folder);
				else
					g_printerr("odl_tb5_tray: failed to set sync folder\n");
			}

			g_free(folder);
		}
	}

	gtk_widget_destroy(dialog);
}
