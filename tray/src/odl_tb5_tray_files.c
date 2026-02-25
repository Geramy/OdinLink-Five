/*
 * OdinLink TB5 Tray - File Browser Window
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>

GtkWidget *g_files_window = NULL;

enum {
	COL_NAME,
	COL_LOCATION,
	COL_LOC_COLOR,
	COL_SIZE,
	COL_MTIME,
	COL_IS_DIR,
	COL_REL_PATH,
	COL_LOC_RAW,
	NUM_COLS
};

struct files_window_ctx {
	GtkWidget      *window;
	GtkWidget      *tree_view;
	GtkListStore   *store;
	GtkWidget      *lbl_path;
	char           *current_dir;
	guint           refresh_timer;
};

static struct files_window_ctx *fw_ctx = NULL;

/* Format a byte count into a human-readable string */
static void format_file_size(uint64_t bytes, char *buf, size_t buflen)
{
	if (bytes == 0) {
		snprintf(buf, buflen, "-");
		return;
	}

	double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
	if (gb >= 1.0) {
		snprintf(buf, buflen, "%.2f GB", gb);
		return;
	}
	double mb = (double)bytes / (1024.0 * 1024.0);
	if (mb >= 1.0) {
		snprintf(buf, buflen, "%.2f MB", mb);
		return;
	}
	double kb = (double)bytes / 1024.0;
	if (kb >= 1.0) {
		snprintf(buf, buflen, "%.1f KB", kb);
		return;
	}
	snprintf(buf, buflen, "%" PRIu64 " bytes", bytes);
}

/* Format nanosecond timestamp to date string */
static void format_mtime_ns(uint64_t mtime_ns, char *buf, size_t buflen)
{
	if (mtime_ns == 0) {
		snprintf(buf, buflen, "-");
		return;
	}
	time_t secs = (time_t)(mtime_ns / 1000000000ULL);
	struct tm tm_buf;
	struct tm *tm = localtime_r(&secs, &tm_buf);
	if (!tm) {
		snprintf(buf, buflen, "-");
		return;
	}
	strftime(buf, buflen, "%Y-%m-%d %H:%M:%S", tm);
}

static const char *location_display(const char *loc)
{
	if (strcmp(loc, "local") == 0)  return "Local";
	if (strcmp(loc, "remote") == 0) return "Remote";
	if (strcmp(loc, "cached") == 0) return "Cached";
	if (strcmp(loc, "both") == 0)   return "Both";
	return loc;
}

static const char *location_color(const char *loc)
{
	if (strcmp(loc, "local") == 0)  return "green";
	if (strcmp(loc, "remote") == 0) return "blue";
	if (strcmp(loc, "cached") == 0) return "orange";
	if (strcmp(loc, "both") == 0)   return "purple";
	return "black";
}

/* Extract a JSON string value for a given key from a single object string */
static char *json_obj_get_str(const char *obj, const char *key,
                              char *buf, size_t buflen)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\":", key);
	const char *pos = strstr(obj, pattern);
	if (!pos)
		return NULL;
	pos += strlen(pattern);
	while (*pos == ' ' || *pos == '\t')
		pos++;
	if (*pos != '"')
		return NULL;
	pos++;

	size_t i = 0;
	while (*pos && *pos != '"' && i < buflen - 1) {
		if (*pos == '\\' && *(pos + 1)) {
			pos++;
		}
		buf[i++] = *pos++;
	}
	buf[i] = '\0';
	return buf;
}

/* Extract a JSON boolean value for a given key */
static gboolean json_obj_get_bool(const char *obj, const char *key)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\":", key);
	const char *pos = strstr(obj, pattern);
	if (!pos)
		return FALSE;
	pos += strlen(pattern);
	while (*pos == ' ' || *pos == '\t')
		pos++;
	return (strncmp(pos, "true", 4) == 0);
}

/* Extract a JSON uint64 value for a given key */
static uint64_t json_obj_get_uint64(const char *obj, const char *key)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\":", key);
	const char *pos = strstr(obj, pattern);
	if (!pos)
		return 0;
	pos += strlen(pattern);
	uint64_t val = 0;
	sscanf(pos, "%" SCNu64, &val);
	return val;
}

static char *build_rel_path(const char *dir, const char *name)
{
	if (!dir || strcmp(dir, "/") == 0)
		return g_strdup_printf("/%s", name);
	return g_strdup_printf("%s/%s", dir, name);
}

/* Populate tree from D-Bus ListFiles */
static void files_window_populate(struct files_window_ctx *ctx)
{
	gtk_list_store_clear(ctx->store);

	char *markup = g_strdup_printf("<b>%s</b>",
	                               ctx->current_dir ? ctx->current_dir : "/");
	gtk_label_set_markup(GTK_LABEL(ctx->lbl_path), markup);
	g_free(markup);

	GVariant *result = odl_tray_dbus_call_sync(
		"ListFiles",
		g_variant_new("(s)", ctx->current_dir ? ctx->current_dir : "/"));

	if (!result) {
		g_printerr("odl_tb5_tray_files: ListFiles failed for '%s'\n",
		           ctx->current_dir ? ctx->current_dir : "/");
		return;
	}

	const gchar *json = NULL;
	g_variant_get(result, "(s)", &json);

	if (!json || !json[0]) {
		g_variant_unref(result);
		return;
	}

	const char *cursor = json;

	while ((cursor = strchr(cursor, '{')) != NULL) {
		const char *obj_end = strchr(cursor, '}');
		if (!obj_end)
			break;

		size_t obj_len = (size_t)(obj_end - cursor + 1);
		char *obj = g_malloc(obj_len + 1);
		memcpy(obj, cursor, obj_len);
		obj[obj_len] = '\0';

		char path_buf[512];
		char loc_buf[32];
		char size_fmt[64];
		char mtime_fmt[64];

		char *path_str = json_obj_get_str(obj, "path", path_buf,
		                                  sizeof(path_buf));
		char *loc_str  = json_obj_get_str(obj, "location", loc_buf,
		                                  sizeof(loc_buf));
		gboolean is_dir = json_obj_get_bool(obj, "is_dir");
		uint64_t size   = json_obj_get_uint64(obj, "size");
		uint64_t mtime  = json_obj_get_uint64(obj, "mtime");

		if (path_str && loc_str) {
			const char *display_name = path_str;

			if (is_dir)
				snprintf(size_fmt, sizeof(size_fmt), "-");
			else
				format_file_size(size, size_fmt, sizeof(size_fmt));

			format_mtime_ns(mtime, mtime_fmt, sizeof(mtime_fmt));

			char *rel = build_rel_path(ctx->current_dir, path_str);

			GtkTreeIter iter;
			gtk_list_store_append(ctx->store, &iter);
			gtk_list_store_set(ctx->store, &iter,
			                   COL_NAME,      display_name,
			                   COL_LOCATION,  location_display(loc_str),
			                   COL_LOC_COLOR, location_color(loc_str),
			                   COL_SIZE,      size_fmt,
			                   COL_MTIME,     mtime_fmt,
			                   COL_IS_DIR,    is_dir,
			                   COL_REL_PATH,  rel,
			                   COL_LOC_RAW,   loc_str,
			                   -1);
			g_free(rel);
		}

		g_free(obj);
		cursor = obj_end + 1;
	}

	g_variant_unref(result);
}

static void on_action_fetch(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	char *rel_path = user_data;

	g_printerr("odl_tb5_tray_files: FetchFile '%s'\n", rel_path);
	GVariant *result = odl_tray_dbus_call_sync(
		"FetchFile", g_variant_new("(s)", rel_path));
	if (result)
		g_variant_unref(result);

	if (fw_ctx)
		files_window_populate(fw_ctx);
}

static void on_action_transfer(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	char *rel_path = user_data;

	g_printerr("odl_tb5_tray_files: TransferFile '%s'\n", rel_path);
	GVariant *result = odl_tray_dbus_call_sync(
		"TransferFile", g_variant_new("(s)", rel_path));
	if (result)
		g_variant_unref(result);

	if (fw_ctx)
		files_window_populate(fw_ctx);
}

static void on_action_remove_local(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	char *rel_path = user_data;

	g_printerr("odl_tb5_tray_files: RemoveLocalCopy '%s'\n", rel_path);
	GVariant *result = odl_tray_dbus_call_sync(
		"RemoveLocalCopy", g_variant_new("(s)", rel_path));
	if (result)
		g_variant_unref(result);

	if (fw_ctx)
		files_window_populate(fw_ctx);
}

static void on_action_remove_peer(GtkMenuItem *item, gpointer user_data)
{
	(void)item;
	char *rel_path = user_data;

	g_printerr("odl_tb5_tray_files: RemoveFromPeer '%s'\n", rel_path);
	GVariant *result = odl_tray_dbus_call_sync(
		"RemoveFromPeer", g_variant_new("(s)", rel_path));
	if (result)
		g_variant_unref(result);

	if (fw_ctx)
		files_window_populate(fw_ctx);
}

static void on_context_menu(struct files_window_ctx *ctx,
                            GdkEventButton *event)
{
	GtkTreeSelection *sel;
	GtkTreeModel *model;
	GtkTreeIter iter;

	sel = gtk_tree_view_get_selection(GTK_TREE_VIEW(ctx->tree_view));
	if (!gtk_tree_selection_get_selected(sel, &model, &iter))
		return;

	gchar *loc_raw  = NULL;
	gchar *rel_path = NULL;

	gtk_tree_model_get(model, &iter,
	                   COL_LOC_RAW,  &loc_raw,
	                   COL_REL_PATH, &rel_path,
	                   -1);

	if (!loc_raw || !rel_path) {
		g_free(loc_raw);
		g_free(rel_path);
		return;
	}

	GtkWidget *menu = gtk_menu_new();

	g_object_set_data_full(G_OBJECT(menu), "rel_path",
	                       g_strdup(rel_path), (GDestroyNotify)g_free);

	if (strcmp(loc_raw, "remote") == 0) {
		GtkWidget *mi = gtk_menu_item_new_with_label("Download Locally");
		g_signal_connect(mi, "activate",
		                 G_CALLBACK(on_action_fetch), rel_path);
		gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

	} else if (strcmp(loc_raw, "local") == 0) {
		GtkWidget *mi = gtk_menu_item_new_with_label("Transfer to Peer");
		g_signal_connect(mi, "activate",
		                 G_CALLBACK(on_action_transfer), rel_path);
		gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

	} else if (strcmp(loc_raw, "both") == 0) {
		GtkWidget *mi1 = gtk_menu_item_new_with_label("Remove Local Copy");
		g_signal_connect(mi1, "activate",
		                 G_CALLBACK(on_action_remove_local), rel_path);
		gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi1);

		GtkWidget *mi2 = gtk_menu_item_new_with_label("Remove from Peer");
		g_signal_connect(mi2, "activate",
		                 G_CALLBACK(on_action_remove_peer), rel_path);
		gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi2);

	} else if (strcmp(loc_raw, "cached") == 0) {
		GtkWidget *mi = gtk_menu_item_new_with_label("Remove Local Copy");
		g_signal_connect(mi, "activate",
		                 G_CALLBACK(on_action_remove_local), rel_path);
		gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
	}

	g_free(loc_raw);

	gtk_widget_show_all(menu);
	gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)event);
}

static gboolean on_auto_refresh(gpointer user_data)
{
	struct files_window_ctx *ctx = user_data;

	if (!ctx || !ctx->window)
		return G_SOURCE_REMOVE;

	if (!gtk_widget_get_visible(ctx->window))
		return G_SOURCE_CONTINUE;

	files_window_populate(ctx);
	return G_SOURCE_CONTINUE;
}

static void on_refresh_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct files_window_ctx *ctx = user_data;
	files_window_populate(ctx);
}

static void on_up_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct files_window_ctx *ctx = user_data;

	if (!ctx->current_dir || strcmp(ctx->current_dir, "/") == 0)
		return;

	char *last_slash = strrchr(ctx->current_dir, '/');
	if (!last_slash || last_slash == ctx->current_dir) {
		g_free(ctx->current_dir);
		ctx->current_dir = g_strdup("/");
	} else {
		*last_slash = '\0';
	}

	files_window_populate(ctx);
}

static void on_row_activated(GtkTreeView       *tree_view,
                             GtkTreePath        *path,
                             GtkTreeViewColumn  *column,
                             gpointer            user_data)
{
	(void)column;
	struct files_window_ctx *ctx = user_data;
	GtkTreeModel *model = gtk_tree_view_get_model(tree_view);
	GtkTreeIter iter;

	if (!gtk_tree_model_get_iter(model, &iter, path))
		return;

	gboolean is_dir = FALSE;
	gchar *rel_path = NULL;

	gtk_tree_model_get(model, &iter,
	                   COL_IS_DIR,   &is_dir,
	                   COL_REL_PATH, &rel_path,
	                   -1);

	if (is_dir && rel_path) {
		g_free(ctx->current_dir);
		ctx->current_dir = g_strdup(rel_path);
		files_window_populate(ctx);
	}

	g_free(rel_path);
}

static gboolean on_button_press(GtkWidget *widget, GdkEventButton *event,
                                gpointer user_data)
{
	struct files_window_ctx *ctx = user_data;

	if (event->type == GDK_BUTTON_PRESS && event->button == 3) {
		GtkTreePath *path = NULL;
		if (gtk_tree_view_get_path_at_pos(GTK_TREE_VIEW(widget),
		                                  (gint)event->x,
		                                  (gint)event->y,
		                                  &path, NULL, NULL, NULL)) {
			GtkTreeSelection *sel;
			sel = gtk_tree_view_get_selection(GTK_TREE_VIEW(widget));
			gtk_tree_selection_select_path(sel, path);
			gtk_tree_path_free(path);

			on_context_menu(ctx, event);
			return TRUE;
		}
	}
	return FALSE;
}

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
	(void)widget;
	struct files_window_ctx *ctx = user_data;

	if (ctx->refresh_timer > 0) {
		g_source_remove(ctx->refresh_timer);
		ctx->refresh_timer = 0;
	}

	g_free(ctx->current_dir);
	g_files_window = NULL;
	g_free(fw_ctx);
	fw_ctx = NULL;
}

/* Cell data function for Location column colouring */
static void location_cell_data_func(GtkTreeViewColumn *column,
                                    GtkCellRenderer   *cell,
                                    GtkTreeModel      *model,
                                    GtkTreeIter       *iter,
                                    gpointer           data)
{
	(void)column;
	(void)data;

	gchar *text  = NULL;
	gchar *color = NULL;

	gtk_tree_model_get(model, iter,
	                   COL_LOCATION,  &text,
	                   COL_LOC_COLOR, &color,
	                   -1);

	g_object_set(cell,
	             "text", text ? text : "",
	             "foreground", color ? color : "black",
	             NULL);

	g_free(text);
	g_free(color);
}

void odl_tray_files_show(void)
{
	if (g_files_window && fw_ctx) {
		files_window_populate(fw_ctx);
		gtk_window_present(GTK_WINDOW(g_files_window));
		return;
	}

	fw_ctx = g_new0(struct files_window_ctx, 1);
	fw_ctx->current_dir = g_strdup("/");

	GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window),
	                     "OdinLink TB5 - File Browser");
	gtk_window_set_default_size(GTK_WINDOW(window), 700, 500);
	gtk_window_set_resizable(GTK_WINDOW(window), TRUE);
	gtk_container_set_border_width(GTK_CONTAINER(window), 8);
	g_signal_connect(window, "destroy",
	                 G_CALLBACK(on_window_destroy), fw_ctx);

	fw_ctx->window = window;
	g_files_window = window;

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
	gtk_container_add(GTK_CONTAINER(window), vbox);

	GtkWidget *nav_bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
	gtk_box_pack_start(GTK_BOX(vbox), nav_bar, FALSE, FALSE, 0);

	GtkWidget *btn_up = gtk_button_new_with_label("Up");
	g_signal_connect(btn_up, "clicked",
	                 G_CALLBACK(on_up_clicked), fw_ctx);
	gtk_box_pack_start(GTK_BOX(nav_bar), btn_up, FALSE, FALSE, 0);

	GtkWidget *btn_refresh = gtk_button_new_with_label("Refresh");
	g_signal_connect(btn_refresh, "clicked",
	                 G_CALLBACK(on_refresh_clicked), fw_ctx);
	gtk_box_pack_start(GTK_BOX(nav_bar), btn_refresh, FALSE, FALSE, 0);

	fw_ctx->lbl_path = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(fw_ctx->lbl_path), "<b>/</b>");
	gtk_widget_set_halign(fw_ctx->lbl_path, GTK_ALIGN_START);
	gtk_label_set_ellipsize(GTK_LABEL(fw_ctx->lbl_path),
	                        PANGO_ELLIPSIZE_MIDDLE);
	gtk_box_pack_start(GTK_BOX(nav_bar), fw_ctx->lbl_path, TRUE, TRUE, 4);

	fw_ctx->store = gtk_list_store_new(
		NUM_COLS,
		G_TYPE_STRING,
		G_TYPE_STRING,
		G_TYPE_STRING,
		G_TYPE_STRING,
		G_TYPE_STRING,
		G_TYPE_BOOLEAN,
		G_TYPE_STRING,
		G_TYPE_STRING);

	GtkWidget *tree_view = gtk_tree_view_new_with_model(
		GTK_TREE_MODEL(fw_ctx->store));
	g_object_unref(fw_ctx->store);

	fw_ctx->tree_view = tree_view;

	gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(tree_view), TRUE);
	gtk_tree_view_set_enable_search(GTK_TREE_VIEW(tree_view), TRUE);

	GtkCellRenderer *rend_text;
	GtkTreeViewColumn *col;

	rend_text = gtk_cell_renderer_text_new();
	col = gtk_tree_view_column_new_with_attributes(
		"Name", rend_text, "text", COL_NAME, NULL);
	gtk_tree_view_column_set_expand(col, TRUE);
	gtk_tree_view_column_set_resizable(col, TRUE);
	gtk_tree_view_column_set_sort_column_id(col, COL_NAME);
	gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), col);

	rend_text = gtk_cell_renderer_text_new();
	col = gtk_tree_view_column_new();
	gtk_tree_view_column_set_title(col, "Location");
	gtk_tree_view_column_pack_start(col, rend_text, TRUE);
	gtk_tree_view_column_set_cell_data_func(
		col, rend_text, location_cell_data_func, NULL, NULL);
	gtk_tree_view_column_set_resizable(col, TRUE);
	gtk_tree_view_column_set_min_width(col, 80);
	gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), col);

	rend_text = gtk_cell_renderer_text_new();
	g_object_set(rend_text, "xalign", 1.0f, NULL);
	col = gtk_tree_view_column_new_with_attributes(
		"Size", rend_text, "text", COL_SIZE, NULL);
	gtk_tree_view_column_set_resizable(col, TRUE);
	gtk_tree_view_column_set_min_width(col, 80);
	gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), col);

	rend_text = gtk_cell_renderer_text_new();
	col = gtk_tree_view_column_new_with_attributes(
		"Modified", rend_text, "text", COL_MTIME, NULL);
	gtk_tree_view_column_set_resizable(col, TRUE);
	gtk_tree_view_column_set_min_width(col, 140);
	gtk_tree_view_append_column(GTK_TREE_VIEW(tree_view), col);

	g_signal_connect(tree_view, "row-activated",
	                 G_CALLBACK(on_row_activated), fw_ctx);
	g_signal_connect(tree_view, "button-press-event",
	                 G_CALLBACK(on_button_press), fw_ctx);

	GtkWidget *scrolled = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled),
	                               GTK_POLICY_AUTOMATIC,
	                               GTK_POLICY_AUTOMATIC);
	gtk_container_add(GTK_CONTAINER(scrolled), tree_view);
	gtk_box_pack_start(GTK_BOX(vbox), scrolled, TRUE, TRUE, 0);

	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
	gtk_widget_set_halign(hbox, GTK_ALIGN_END);
	gtk_box_pack_end(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

	GtkWidget *btn_close = gtk_button_new_with_label("Close");
	g_signal_connect_swapped(btn_close, "clicked",
	                         G_CALLBACK(gtk_widget_destroy), window);
	gtk_box_pack_start(GTK_BOX(hbox), btn_close, FALSE, FALSE, 0);

	files_window_populate(fw_ctx);
	gtk_widget_show_all(window);

	fw_ctx->refresh_timer = g_timeout_add(5000, on_auto_refresh, fw_ctx);
}
