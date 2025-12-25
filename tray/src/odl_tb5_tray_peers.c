/*
 * OdinLink TB5 Tray - Peer Detail Window
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>
#include <string.h>

static const char *peer_css =
	".peer-window {"
	"  background-color: #282828;"
	"  color: #E6E6E6;"
	"}"
	".peer-window label {"
	"  color: #E6E6E6;"
	"}"
	".peer-section-title {"
	"  color: #4CAF50;"
	"  font-weight: bold;"
	"  font-size: 14px;"
	"}"
	".peer-field-label {"
	"  color: #9E9E9E;"
	"}"
	".peer-field-value {"
	"  color: #E6E6E6;"
	"}"
	".peer-window button {"
	"  background-image: none;"
	"  background-color: #363636;"
	"  color: #E6E6E6;"
	"  border: 1px solid #555555;"
	"  border-radius: 4px;"
	"  padding: 4px 12px;"
	"  text-shadow: none;"
	"  box-shadow: none;"
	"}"
	".peer-window button label {"
	"  color: #E6E6E6;"
	"}"
	".peer-window button:hover {"
	"  background-image: none;"
	"  background-color: #4CAF50;"
	"  color: #282828;"
	"}"
	".peer-window button:hover label {"
	"  color: #282828;"
	"}"
	".peer-separator {"
	"  background-color: #555555;"
	"  min-height: 1px;"
	"}"
	".peer-log-view text {"
	"  background-color: #292929;"
	"  color: #CCCCCC;"
	"}";

struct peer_window_ctx {
	int         device_index;
	GtkWidget  *window;
	GtkWidget  *content_box;
};

static struct peer_window_ctx *pw_ctx = NULL;

/* Add a label + value row to a grid */
static void add_row(GtkGrid *grid, int row,
                    const char *label_text, const char *value_text)
{
	GtkWidget *label = gtk_label_new(NULL);
	char *markup = g_strdup_printf("<b>%s</b>", label_text);
	gtk_label_set_markup(GTK_LABEL(label), markup);
	g_free(markup);
	gtk_widget_set_halign(label, GTK_ALIGN_END);
	gtk_style_context_add_class(
		gtk_widget_get_style_context(label), "peer-field-label");
	gtk_grid_attach(grid, label, 0, row, 1, 1);

	GtkWidget *value = gtk_label_new(value_text);
	gtk_widget_set_halign(value, GTK_ALIGN_START);
	gtk_label_set_selectable(GTK_LABEL(value), TRUE);
	gtk_label_set_line_wrap(GTK_LABEL(value), TRUE);
	gtk_label_set_max_width_chars(GTK_LABEL(value), 50);
	gtk_style_context_add_class(
		gtk_widget_get_style_context(value), "peer-field-value");
	gtk_grid_attach(grid, value, 1, row, 1, 1);
}

/* Add a section header */
static void add_section(GtkBox *box, const char *title)
{
	GList *children = gtk_container_get_children(GTK_CONTAINER(box));
	if (children) {
		GtkWidget *sep = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
		gtk_style_context_add_class(
			gtk_widget_get_style_context(sep), "peer-separator");
		gtk_box_pack_start(box, sep, FALSE, FALSE, 6);
	}
	g_list_free(children);

	GtkWidget *label = gtk_label_new(title);
	gtk_style_context_add_class(
		gtk_widget_get_style_context(label), "peer-section-title");
	gtk_widget_set_halign(label, GTK_ALIGN_START);
	gtk_box_pack_start(box, label, FALSE, FALSE, 2);
}

/* Build content (called on create and refresh) */
static void peer_window_populate(struct peer_window_ctx *ctx)
{
	char buf[256];

	GList *children = gtk_container_get_children(
		GTK_CONTAINER(ctx->content_box));
	for (GList *l = children; l; l = l->next)
		gtk_widget_destroy(GTK_WIDGET(l->data));
	g_list_free(children);

	GtkBox *box = GTK_BOX(ctx->content_box);

	add_section(box, "Connection Info");

	GtkWidget *conn_grid = gtk_grid_new();
	gtk_grid_set_row_spacing(GTK_GRID(conn_grid), 4);
	gtk_grid_set_column_spacing(GTK_GRID(conn_grid), 16);
	gtk_widget_set_margin_start(conn_grid, 12);
	gtk_box_pack_start(box, conn_grid, FALSE, FALSE, 0);

	int row = 0;

	snprintf(buf, sizeof(buf), "%d", ctx->device_index);
	add_row(GTK_GRID(conn_grid), row++, "Device Index:", buf);

	GVariant *result = odl_tray_dbus_call_sync(
		"GetPeerInfo",
		g_variant_new("(i)", ctx->device_index));

	if (result) {
		const gchar *uuid = NULL;
		guint32 link_speed = 0, link_width = 0;
		const gchar *state = NULL, *vendor = NULL, *devname = NULL;

		g_variant_get(result, "(suuss)",
		              &uuid, &link_speed, &link_width,
		              &state, &vendor, &devname);

		add_row(GTK_GRID(conn_grid), row++, "UUID:",
		        uuid ? uuid : "(unknown)");

		snprintf(buf, sizeof(buf), "%u Gb/s", link_speed);
		add_row(GTK_GRID(conn_grid), row++, "Link Speed:", buf);

		snprintf(buf, sizeof(buf), "x%u", link_width);
		add_row(GTK_GRID(conn_grid), row++, "Link Width:", buf);

		add_row(GTK_GRID(conn_grid), row++, "State:",
		        state ? state : "(unknown)");

		add_row(GTK_GRID(conn_grid), row++, "Vendor:",
		        vendor && vendor[0] ? vendor : "(unknown)");

		add_row(GTK_GRID(conn_grid), row++, "Device:",
		        devname && devname[0] ? devname : "(unknown)");

		g_variant_unref(result);
	} else {
		add_row(GTK_GRID(conn_grid), row++, "Status:",
		        "(device unavailable)");
	}

	GVariant *sysinfo = odl_tray_dbus_call_sync(
		"GetPeerSystemInfo",
		g_variant_new("(i)", ctx->device_index));

	if (sysinfo) {
		GVariantIter *cpu_iter = NULL;
		guint32 ram_total = 0, ram_avail = 0;
		GVariantIter *gpu_iter = NULL;

		g_variant_get(sysinfo, "(a(suuu)uua(suu))",
		              &cpu_iter, &ram_total, &ram_avail, &gpu_iter);

		add_section(box, "Processors");

		const gchar *cpu_model = NULL;
		guint32 cores = 0, threads = 0, freq = 0;
		int cpu_num = 0;

		while (g_variant_iter_loop(cpu_iter, "(suuu)",
		                           &cpu_model, &cores, &threads,
		                           &freq)) {
			GtkWidget *cpu_grid = gtk_grid_new();
			gtk_grid_set_row_spacing(GTK_GRID(cpu_grid), 3);
			gtk_grid_set_column_spacing(GTK_GRID(cpu_grid), 16);
			gtk_widget_set_margin_start(cpu_grid, 12);
			if (cpu_num > 0)
				gtk_widget_set_margin_top(cpu_grid, 6);
			gtk_box_pack_start(box, cpu_grid, FALSE, FALSE, 0);

			row = 0;
			if (cpu_num > 0) {
				snprintf(buf, sizeof(buf), "Socket %d", cpu_num);
				add_row(GTK_GRID(cpu_grid), row++, "Socket:", buf);
			}
			add_row(GTK_GRID(cpu_grid), row++, "Model:",
			        cpu_model ? cpu_model : "(unknown)");

			snprintf(buf, sizeof(buf), "%u cores / %u threads",
			         cores, threads);
			add_row(GTK_GRID(cpu_grid), row++, "Cores:", buf);

			if (freq >= 1000)
				snprintf(buf, sizeof(buf), "%.2f GHz",
				         freq / 1000.0);
			else
				snprintf(buf, sizeof(buf), "%u MHz", freq);
			add_row(GTK_GRID(cpu_grid), row++, "Max Frequency:", buf);

			cpu_num++;
		}
		g_variant_iter_free(cpu_iter);

		add_section(box, "Memory");

		GtkWidget *mem_grid = gtk_grid_new();
		gtk_grid_set_row_spacing(GTK_GRID(mem_grid), 3);
		gtk_grid_set_column_spacing(GTK_GRID(mem_grid), 16);
		gtk_widget_set_margin_start(mem_grid, 12);
		gtk_box_pack_start(box, mem_grid, FALSE, FALSE, 0);

		row = 0;
		if (ram_total >= 1024)
			snprintf(buf, sizeof(buf), "%.1f GB",
			         ram_total / 1024.0);
		else
			snprintf(buf, sizeof(buf), "%u MB", ram_total);
		add_row(GTK_GRID(mem_grid), row++, "Total RAM:", buf);

		if (ram_avail >= 1024)
			snprintf(buf, sizeof(buf), "%.1f GB",
			         ram_avail / 1024.0);
		else
			snprintf(buf, sizeof(buf), "%u MB", ram_avail);
		add_row(GTK_GRID(mem_grid), row++, "Available:", buf);

		if (ram_total > 0) {
			unsigned int used = ram_total - ram_avail;
			snprintf(buf, sizeof(buf), "%.1f GB (%.0f%%)",
			         used / 1024.0,
			         (double)used / ram_total * 100.0);
			add_row(GTK_GRID(mem_grid), row++, "Used:", buf);
		}

		add_section(box, "Graphics");

		const gchar *gpu_name = NULL;
		guint32 vram_total = 0, vram_used = 0;
		int gpu_num = 0;

		while (g_variant_iter_loop(gpu_iter, "(suu)",
		                           &gpu_name, &vram_total,
		                           &vram_used)) {
			GtkWidget *gpu_grid = gtk_grid_new();
			gtk_grid_set_row_spacing(GTK_GRID(gpu_grid), 3);
			gtk_grid_set_column_spacing(GTK_GRID(gpu_grid), 16);
			gtk_widget_set_margin_start(gpu_grid, 12);
			if (gpu_num > 0)
				gtk_widget_set_margin_top(gpu_grid, 6);
			gtk_box_pack_start(box, gpu_grid, FALSE, FALSE, 0);

			row = 0;
			snprintf(buf, sizeof(buf), "GPU %d", gpu_num);
			add_row(GTK_GRID(gpu_grid), row++, buf,
			        gpu_name ? gpu_name : "(unknown)");

			if (vram_total > 0) {
				if (vram_total >= 1024)
					snprintf(buf, sizeof(buf), "%.1f GB",
					         vram_total / 1024.0);
				else
					snprintf(buf, sizeof(buf), "%u MB",
					         vram_total);
				add_row(GTK_GRID(gpu_grid), row++,
				        "VRAM Total:", buf);

				unsigned int vram_avail = vram_total - vram_used;
				if (vram_avail >= 1024)
					snprintf(buf, sizeof(buf), "%.1f GB",
					         vram_avail / 1024.0);
				else
					snprintf(buf, sizeof(buf), "%u MB",
					         vram_avail);
				add_row(GTK_GRID(gpu_grid), row++,
				        "VRAM Available:", buf);

				snprintf(buf, sizeof(buf), "%.1f GB (%.0f%%)",
				         vram_used / 1024.0,
				         (double)vram_used / vram_total * 100.0);
				add_row(GTK_GRID(gpu_grid), row++,
				        "VRAM Used:", buf);
			}

			gpu_num++;
		}
		g_variant_iter_free(gpu_iter);

		if (gpu_num == 0) {
			GtkWidget *none = gtk_label_new("No GPUs detected");
			gtk_widget_set_margin_start(none, 12);
			gtk_widget_set_halign(none, GTK_ALIGN_START);
			gtk_box_pack_start(box, none, FALSE, FALSE, 0);
		}

		g_variant_unref(sysinfo);
	} else {
		add_section(box, "Peer System Info");
		GtkWidget *na_label = gtk_label_new(
			"System info not available\n"
			"(peer may not be running OdinLink daemon)");
		gtk_widget_set_margin_start(na_label, 12);
		gtk_widget_set_halign(na_label, GTK_ALIGN_START);
		gtk_style_context_add_class(
			gtk_widget_get_style_context(na_label),
			"peer-field-value");
		gtk_box_pack_start(box, na_label, FALSE, FALSE, 0);
	}

	add_section(box, "Recent Logs");

	GVariant *logs_result = odl_tray_dbus_call_sync(
		"GetRecentLogs",
		g_variant_new("(u)", (guint32)50));

	GtkWidget *log_scroll = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(log_scroll),
	                               GTK_POLICY_AUTOMATIC,
	                               GTK_POLICY_AUTOMATIC);
	gtk_scrolled_window_set_min_content_height(
		GTK_SCROLLED_WINDOW(log_scroll), 150);
	gtk_widget_set_margin_start(log_scroll, 12);
	gtk_widget_set_margin_end(log_scroll, 4);
	gtk_box_pack_start(box, log_scroll, TRUE, TRUE, 0);

	GtkWidget *log_view = gtk_text_view_new();
	gtk_text_view_set_editable(GTK_TEXT_VIEW(log_view), FALSE);
	gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(log_view), FALSE);
	gtk_text_view_set_monospace(GTK_TEXT_VIEW(log_view), TRUE);
	gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(log_view),
	                            GTK_WRAP_WORD_CHAR);
	gtk_text_view_set_left_margin(GTK_TEXT_VIEW(log_view), 4);
	gtk_text_view_set_right_margin(GTK_TEXT_VIEW(log_view), 4);

	gtk_style_context_add_class(
		gtk_widget_get_style_context(log_view), "peer-log-view");

	GtkTextBuffer *log_buf = gtk_text_view_get_buffer(
		GTK_TEXT_VIEW(log_view));

	if (logs_result) {
		const gchar *log_text = NULL;
		g_variant_get(logs_result, "(s)", &log_text);
		if (log_text)
			gtk_text_buffer_set_text(log_buf, log_text, -1);
		g_variant_unref(logs_result);
	} else {
		gtk_text_buffer_set_text(log_buf,
			"(could not fetch logs from daemon)", -1);
	}

	GtkTextIter end_iter;
	gtk_text_buffer_get_end_iter(log_buf, &end_iter);
	GtkTextMark *end_mark = gtk_text_buffer_create_mark(
		log_buf, NULL, &end_iter, FALSE);
	gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(log_view),
	                             end_mark, 0.0, TRUE, 0.0, 1.0);

	gtk_container_add(GTK_CONTAINER(log_scroll), log_view);

	gtk_widget_show_all(ctx->content_box);
}

static void on_refresh_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct peer_window_ctx *ctx = user_data;
	peer_window_populate(ctx);
}

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
	(void)widget;
	(void)user_data;
	g_peers_window = NULL;
	g_free(pw_ctx);
	pw_ctx = NULL;
}

void odl_tray_peers_show(int device_index)
{
	if (g_peers_window && pw_ctx) {
		pw_ctx->device_index = device_index;
		peer_window_populate(pw_ctx);
		gtk_window_present(GTK_WINDOW(g_peers_window));
		return;
	}

	GtkCssProvider *css = gtk_css_provider_new();
	gtk_css_provider_load_from_data(css, peer_css, -1, NULL);
	gtk_style_context_add_provider_for_screen(
		gdk_screen_get_default(),
		GTK_STYLE_PROVIDER(css),
		GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	g_object_unref(css);

	pw_ctx = g_new0(struct peer_window_ctx, 1);
	pw_ctx->device_index = device_index;

	GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window), "OdinLink TB5 - Peer Details");
	gtk_window_set_default_size(GTK_WINDOW(window), 520, 600);
	gtk_window_set_resizable(GTK_WINDOW(window), TRUE);
	gtk_style_context_add_class(
		gtk_widget_get_style_context(window), "peer-window");
	g_signal_connect(window, "destroy", G_CALLBACK(on_window_destroy), NULL);

	pw_ctx->window = window;
	g_peers_window = window;

	GtkWidget *outer_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_container_add(GTK_CONTAINER(window), outer_box);

	GtkWidget *title = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(title),
	                     "<span size='x-large' weight='bold' "
	                     "foreground='#4CAF50'>"
	                     "Peer Device Details</span>");
	gtk_widget_set_margin_top(title, 12);
	gtk_widget_set_margin_bottom(title, 8);
	gtk_box_pack_start(GTK_BOX(outer_box), title, FALSE, FALSE, 0);

	GtkWidget *scrolled = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled),
	                               GTK_POLICY_NEVER,
	                               GTK_POLICY_AUTOMATIC);
	gtk_box_pack_start(GTK_BOX(outer_box), scrolled, TRUE, TRUE, 0);

	pw_ctx->content_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
	gtk_container_set_border_width(GTK_CONTAINER(pw_ctx->content_box), 12);
	gtk_container_add(GTK_CONTAINER(scrolled), pw_ctx->content_box);

	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
	gtk_widget_set_halign(hbox, GTK_ALIGN_END);
	gtk_widget_set_margin_top(hbox, 8);
	gtk_widget_set_margin_end(hbox, 12);
	gtk_widget_set_margin_bottom(hbox, 12);
	gtk_box_pack_end(GTK_BOX(outer_box), hbox, FALSE, FALSE, 0);

	GtkWidget *btn_refresh = gtk_button_new_with_label("Refresh");
	g_signal_connect(btn_refresh, "clicked",
	                 G_CALLBACK(on_refresh_clicked), pw_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), btn_refresh, FALSE, FALSE, 0);

	GtkWidget *btn_close = gtk_button_new_with_label("Close");
	g_signal_connect_swapped(btn_close, "clicked",
	                         G_CALLBACK(gtk_widget_destroy), window);
	gtk_box_pack_start(GTK_BOX(hbox), btn_close, FALSE, FALSE, 0);

	peer_window_populate(pw_ctx);
	gtk_widget_show_all(window);
}
