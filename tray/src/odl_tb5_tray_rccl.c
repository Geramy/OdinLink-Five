/*
 * OdinLink TB5 Tray - RCCL Stats Display Window
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

struct rccl_window_ctx {
	GtkWidget  *window;
	GtkWidget  *lbl_active;
	GtkWidget  *lbl_tx_ops;
	GtkWidget  *lbl_rx_ops;
	GtkWidget  *lbl_tx_bytes;
	GtkWidget  *lbl_rx_bytes;
	GtkWidget  *lbl_uptime;
	guint       refresh_timer;
};

static struct rccl_window_ctx *rw_ctx = NULL;

/* Extract a boolean from a JSON string by key */
static gboolean json_get_bool(const char *json, const char *key)
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\":", key);
	const char *pos = strstr(json, pattern);
	if (!pos)
		return FALSE;
	pos += strlen(pattern);
	while (*pos == ' ')
		pos++;
	return (strncmp(pos, "true", 4) == 0);
}

/* Extract a uint64 from a JSON string by key */
static uint64_t json_get_uint64(const char *json, const char *key)
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\":", key);
	const char *pos = strstr(json, pattern);
	if (!pos)
		return 0;
	pos += strlen(pattern);
	uint64_t val = 0;
	sscanf(pos, "%" SCNu64, &val);
	return val;
}

/* Format a byte count into a human-readable string */
static void format_bytes(uint64_t bytes, char *buf, size_t buflen)
{
	double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
	if (gb >= 1.0)
		snprintf(buf, buflen, "%.2f GB", gb);
	else {
		double mb = (double)bytes / (1024.0 * 1024.0);
		if (mb >= 1.0)
			snprintf(buf, buflen, "%.2f MB", mb);
		else
			snprintf(buf, buflen, "%" PRIu64 " bytes", bytes);
	}
}

/* Format seconds into HH:MM:SS */
static void format_uptime_sec(uint64_t uptime_sec, char *buf, size_t buflen)
{
	if (uptime_sec == 0) {
		snprintf(buf, buflen, "-");
		return;
	}

	unsigned hours = (unsigned)(uptime_sec / 3600);
	unsigned mins  = (unsigned)((uptime_sec % 3600) / 60);
	unsigned secs  = (unsigned)(uptime_sec % 60);

	snprintf(buf, buflen, "%02u:%02u:%02u", hours, mins, secs);
}

static void rccl_window_populate(struct rccl_window_ctx *ctx)
{
	GVariant *result = odl_tray_dbus_call_sync("GetRcclStats", NULL);
	if (!result) {
		gtk_label_set_text(GTK_LABEL(ctx->lbl_active), "No");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_tx_ops), "-");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_rx_ops), "-");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_tx_bytes), "-");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_rx_bytes), "-");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_uptime), "-");
		return;
	}

	const gchar *json = NULL;
	g_variant_get(result, "(s)", &json);

	if (!json || !json[0]) {
		gtk_label_set_text(GTK_LABEL(ctx->lbl_active), "No");
		g_variant_unref(result);
		return;
	}

	char buf[128];

	gboolean active = json_get_bool(json, "active");
	gtk_label_set_text(GTK_LABEL(ctx->lbl_active),
	                   active ? "Yes" : "No");

	uint64_t tx_ops = json_get_uint64(json, "tx_ops");
	snprintf(buf, sizeof(buf), "%" PRIu64, tx_ops);
	gtk_label_set_text(GTK_LABEL(ctx->lbl_tx_ops), buf);

	uint64_t rx_ops = json_get_uint64(json, "rx_ops");
	snprintf(buf, sizeof(buf), "%" PRIu64, rx_ops);
	gtk_label_set_text(GTK_LABEL(ctx->lbl_rx_ops), buf);

	uint64_t tx_bytes = json_get_uint64(json, "tx_bytes");
	format_bytes(tx_bytes, buf, sizeof(buf));
	gtk_label_set_text(GTK_LABEL(ctx->lbl_tx_bytes), buf);

	uint64_t rx_bytes = json_get_uint64(json, "rx_bytes");
	format_bytes(rx_bytes, buf, sizeof(buf));
	gtk_label_set_text(GTK_LABEL(ctx->lbl_rx_bytes), buf);

	uint64_t uptime_sec = json_get_uint64(json, "uptime_sec");
	format_uptime_sec(uptime_sec, buf, sizeof(buf));
	gtk_label_set_text(GTK_LABEL(ctx->lbl_uptime), buf);

	g_variant_unref(result);
}

static gboolean on_auto_refresh(gpointer user_data)
{
	struct rccl_window_ctx *ctx = user_data;

	if (!ctx || !ctx->window)
		return G_SOURCE_REMOVE;

	if (!gtk_widget_get_visible(ctx->window))
		return G_SOURCE_CONTINUE;

	rccl_window_populate(ctx);
	return G_SOURCE_CONTINUE;
}

static void on_refresh_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct rccl_window_ctx *ctx = user_data;
	rccl_window_populate(ctx);
}

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
	(void)widget;
	struct rccl_window_ctx *ctx = user_data;

	if (ctx->refresh_timer > 0) {
		g_source_remove(ctx->refresh_timer);
		ctx->refresh_timer = 0;
	}

	g_rccl_window = NULL;
	g_free(rw_ctx);
	rw_ctx = NULL;
}

/* Add a label row to the grid, returning the value widget */
static GtkWidget *add_stat_row(GtkGrid *grid, int row, const char *label_text)
{
	GtkWidget *label = gtk_label_new(NULL);
	gtk_widget_set_halign(label, GTK_ALIGN_END);
	char *markup = g_strdup_printf("<b>%s</b>", label_text);
	gtk_label_set_markup(GTK_LABEL(label), markup);
	g_free(markup);
	gtk_grid_attach(grid, label, 0, row, 1, 1);

	GtkWidget *value = gtk_label_new("-");
	gtk_widget_set_halign(value, GTK_ALIGN_START);
	gtk_label_set_selectable(GTK_LABEL(value), TRUE);
	gtk_grid_attach(grid, value, 1, row, 1, 1);

	return value;
}

void odl_tray_rccl_show(void)
{
	if (g_rccl_window && rw_ctx) {
		rccl_window_populate(rw_ctx);
		gtk_window_present(GTK_WINDOW(g_rccl_window));
		return;
	}

	rw_ctx = g_new0(struct rccl_window_ctx, 1);

	GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window), "OdinLink TB5 - RCCL Stats");
	gtk_window_set_default_size(GTK_WINDOW(window), 380, 280);
	gtk_window_set_resizable(GTK_WINDOW(window), TRUE);
	gtk_container_set_border_width(GTK_CONTAINER(window), 12);
	g_signal_connect(window, "destroy",
	                 G_CALLBACK(on_window_destroy), rw_ctx);

	rw_ctx->window = window;
	g_rccl_window = window;

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
	gtk_container_add(GTK_CONTAINER(window), vbox);

	GtkWidget *title = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(title),
	                     "<span size='large' weight='bold'>"
	                     "RCCL Plugin Statistics</span>");
	gtk_box_pack_start(GTK_BOX(vbox), title, FALSE, FALSE, 4);

	GtkWidget *grid = gtk_grid_new();
	gtk_grid_set_row_spacing(GTK_GRID(grid), 6);
	gtk_grid_set_column_spacing(GTK_GRID(grid), 12);
	gtk_widget_set_halign(grid, GTK_ALIGN_CENTER);
	gtk_box_pack_start(GTK_BOX(vbox), grid, TRUE, FALSE, 0);

	int row = 0;
	rw_ctx->lbl_active   = add_stat_row(GTK_GRID(grid), row++,
	                                     "RCCL Plugin Active:");
	rw_ctx->lbl_tx_ops   = add_stat_row(GTK_GRID(grid), row++,
	                                     "TX Operations:");
	rw_ctx->lbl_rx_ops   = add_stat_row(GTK_GRID(grid), row++,
	                                     "RX Operations:");
	rw_ctx->lbl_tx_bytes = add_stat_row(GTK_GRID(grid), row++,
	                                     "TX Bytes:");
	rw_ctx->lbl_rx_bytes = add_stat_row(GTK_GRID(grid), row++,
	                                     "RX Bytes:");
	rw_ctx->lbl_uptime   = add_stat_row(GTK_GRID(grid), row++,
	                                     "Plugin Uptime:");

	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
	gtk_widget_set_halign(hbox, GTK_ALIGN_END);
	gtk_box_pack_end(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

	GtkWidget *btn_refresh = gtk_button_new_with_label("Refresh");
	g_signal_connect(btn_refresh, "clicked",
	                 G_CALLBACK(on_refresh_clicked), rw_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), btn_refresh, FALSE, FALSE, 0);

	GtkWidget *btn_close = gtk_button_new_with_label("Close");
	g_signal_connect_swapped(btn_close, "clicked",
	                         G_CALLBACK(gtk_widget_destroy), window);
	gtk_box_pack_start(GTK_BOX(hbox), btn_close, FALSE, FALSE, 0);

	rccl_window_populate(rw_ctx);
	gtk_widget_show_all(window);

	rw_ctx->refresh_timer = g_timeout_add(5000, on_auto_refresh, rw_ctx);
}
