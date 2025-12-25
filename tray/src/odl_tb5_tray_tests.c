/*
 * OdinLink TB5 Tray - Test Runner Dialog (Visual Charts Edition)
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include "odl_tb5_tray_test_parse.h"
#include "odl_tb5_tray_test_charts.h"
#include <stdio.h>
#include <string.h>

static const char *test_runner_css =
	".test-runner {"
	"  background-color: #282828;"
	"  color: #E6E6E6;"
	"}"
	".test-runner label {"
	"  color: #E6E6E6;"
	"}"
	".test-runner .title-label {"
	"  color: #4CAF50;"
	"}"
	".test-runner .info-label {"
	"  color: #9E9E9E;"
	"}"
	".test-runner progressbar trough {"
	"  background-color: #363636;"
	"  min-height: 20px;"
	"  border-radius: 4px;"
	"}"
	".test-runner progressbar progress {"
	"  background-color: #4CAF50;"
	"  min-height: 20px;"
	"  border-radius: 4px;"
	"}"
	".test-runner .subtest-label {"
	"  color: #9E9E9E;"
	"  font-style: italic;"
	"}"
	".test-runner notebook {"
	"  background-color: #282828;"
	"}"
	".test-runner notebook header {"
	"  background-color: #363636;"
	"}"
	".test-runner notebook header tab {"
	"  background-color: #363636;"
	"  color: #E6E6E6;"
	"  padding: 4px 12px;"
	"  border: none;"
	"}"
	".test-runner notebook header tab:checked {"
	"  background-color: #4CAF50;"
	"  color: #282828;"
	"}"
	".test-runner textview {"
	"  background-color: #1E1E1E;"
	"  color: #E6E6E6;"
	"}"
	".test-runner textview text {"
	"  background-color: #1E1E1E;"
	"  color: #E6E6E6;"
	"}"
	".test-runner scrolledwindow {"
	"  background-color: #282828;"
	"}"
	".test-runner button {"
	"  background-image: none;"
	"  background-color: #363636;"
	"  color: #E6E6E6;"
	"  border: 1px solid #555555;"
	"  border-radius: 4px;"
	"  padding: 4px 12px;"
	"  text-shadow: none;"
	"  box-shadow: none;"
	"}"
	".test-runner button label {"
	"  color: #E6E6E6;"
	"}"
	".test-runner button:hover {"
	"  background-image: none;"
	"  background-color: #4CAF50;"
	"  color: #282828;"
	"}"
	".test-runner button:hover label {"
	"  color: #282828;"
	"}"
	".test-runner separator {"
	"  background-color: #555555;"
	"  min-height: 1px;"
	"}";

struct test_runner_ctx {
	int          device_index;
	char        *test_type;
	char        *test_id;

	GtkWidget   *window;
	GtkWidget   *lbl_title;
	GtkWidget   *lbl_info;
	GtkWidget   *progress_bar;
	GtkWidget   *lbl_subtest;
	GtkWidget   *results_box;
	GtkWidget   *lbl_status;
	GtkWidget   *notebook;
	GtkWidget   *btn_cancel;
	GtkWidget   *btn_close;
	GtkWidget   *btn_run_again;

	struct odl_parsed_test_result *parsed;
	GString     *log_buffer;
	char        *raw_output;

	guint        poll_timer_id;
	gulong       signal_handler_id;
	gboolean     running;
};

static struct test_runner_ctx *s_ctx = NULL;

static void     test_start(struct test_runner_ctx *ctx);
static gboolean test_poll_timer(gpointer user_data);
static void     test_update_progress(struct test_runner_ctx *ctx,
                                     const char *state,
                                     guint32 progress_pct,
                                     const char *subtest);
static void     test_on_completed(struct test_runner_ctx *ctx,
                                  gboolean success,
                                  const char *summary);
static void     test_fetch_result(struct test_runner_ctx *ctx);
static void     test_set_running(struct test_runner_ctx *ctx,
                                 gboolean running);
static void     test_cleanup_timer(struct test_runner_ctx *ctx);
static void     build_result_tabs(struct test_runner_ctx *ctx);

static void test_log(struct test_runner_ctx *ctx, const char *text)
{
	if (!ctx || !text)
		return;
	g_string_append(ctx->log_buffer, text);
}

static void test_set_status(struct test_runner_ctx *ctx, const char *text)
{
	if (ctx && ctx->lbl_status)
		gtk_label_set_text(GTK_LABEL(ctx->lbl_status), text);
}

static void test_set_running(struct test_runner_ctx *ctx, gboolean running)
{
	ctx->running = running;
	gtk_widget_set_sensitive(ctx->btn_cancel, running);
	gtk_widget_set_sensitive(ctx->btn_run_again, !running);
}

static void test_cleanup_timer(struct test_runner_ctx *ctx)
{
	if (ctx->poll_timer_id > 0) {
		g_source_remove(ctx->poll_timer_id);
		ctx->poll_timer_id = 0;
	}
}

static void test_clear_results(struct test_runner_ctx *ctx)
{
	if (ctx->notebook) {
		gtk_widget_destroy(ctx->notebook);
		ctx->notebook = NULL;
	}
	g_free(ctx->parsed);
	ctx->parsed = NULL;
	g_free(ctx->raw_output);
	ctx->raw_output = NULL;
	g_string_truncate(ctx->log_buffer, 0);

	if (ctx->lbl_status)
		gtk_widget_show(ctx->lbl_status);
}

/* D-Bus signal callback for test progress/completion */
static void on_daemon_signal_for_tests(GDBusProxy  *proxy,
                                       const gchar *sender_name,
                                       const gchar *signal_name,
                                       GVariant    *parameters,
                                       gpointer     user_data)
{
	(void)proxy;
	(void)sender_name;
	struct test_runner_ctx *ctx = user_data;

	if (!ctx || !ctx->test_id)
		return;

	if (g_strcmp0(signal_name, "TestProgress") == 0) {
		const gchar *test_id = NULL;
		guint32 progress;
		const gchar *subtest = NULL;

		g_variant_get(parameters, "(sus)", &test_id, &progress,
		              &subtest);

		if (g_strcmp0(test_id, ctx->test_id) == 0) {
			gtk_progress_bar_set_fraction(
				GTK_PROGRESS_BAR(ctx->progress_bar),
				progress / 100.0);

			if (subtest && subtest[0]) {
				gtk_label_set_text(
					GTK_LABEL(ctx->lbl_subtest), subtest);
				test_set_status(ctx, subtest);
			}
		}

	} else if (g_strcmp0(signal_name, "TestCompleted") == 0) {
		const gchar *test_id = NULL;
		gboolean success;
		const gchar *summary = NULL;

		g_variant_get(parameters, "(sbs)", &test_id, &success,
		              &summary);

		if (g_strcmp0(test_id, ctx->test_id) == 0) {
			test_on_completed(ctx, success, summary);
		}
	}
}

static void test_start(struct test_runner_ctx *ctx)
{
	test_clear_results(ctx);

	gtk_progress_bar_set_fraction(
		GTK_PROGRESS_BAR(ctx->progress_bar), 0.0);
	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(ctx->progress_bar), "Starting...");
	gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest), "Initializing...");
	test_set_status(ctx, "Starting test...");
	test_set_running(ctx, TRUE);

	g_free(ctx->test_id);
	ctx->test_id = NULL;

	char info_buf[256];
	snprintf(info_buf, sizeof(info_buf),
	         "Device: %d  |  Test: %s",
	         ctx->device_index, ctx->test_type);
	gtk_label_set_text(GTK_LABEL(ctx->lbl_info), info_buf);

	GVariant *result = odl_tray_dbus_call_sync(
		"RunTest",
		g_variant_new("(is)", ctx->device_index, ctx->test_type));

	if (!result) {
		test_set_status(ctx,
			"Failed to start test. Is the OdinLink daemon running?");
		test_log(ctx,
			"[ERROR] Failed to start test.\n"
			"Is the OdinLink daemon running?\n"
			"Check that odl_tb5_daemon is active on the "
			"session bus.\n");
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Error");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   "Could not reach daemon");
		test_set_running(ctx, FALSE);
		return;
	}

	const gchar *test_id = NULL;
	g_variant_get(result, "(s)", &test_id);

	if (!test_id || !test_id[0]) {
		test_set_status(ctx, "Daemon returned empty test ID.");
		test_log(ctx, "[ERROR] Daemon returned empty test ID.\n");
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Error");
		test_set_running(ctx, FALSE);
		g_variant_unref(result);
		return;
	}

	ctx->test_id = g_strdup(test_id);
	g_variant_unref(result);

	test_set_status(ctx, "Test running...");
	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(ctx->progress_bar), "Running...");

	test_cleanup_timer(ctx);
	ctx->poll_timer_id = g_timeout_add(500, test_poll_timer, ctx);
}

static gboolean test_poll_timer(gpointer user_data)
{
	struct test_runner_ctx *ctx = user_data;

	if (!ctx || !ctx->test_id)
		return G_SOURCE_REMOVE;

	GVariant *result = odl_tray_dbus_call_sync(
		"GetTestStatus",
		g_variant_new("(s)", ctx->test_id));

	if (!result)
		return G_SOURCE_CONTINUE;

	const gchar *state = NULL;
	guint32 progress_pct = 0;
	const gchar *current_subtest = NULL;

	g_variant_get(result, "(sus)", &state, &progress_pct,
	              &current_subtest);

	test_update_progress(ctx, state, progress_pct, current_subtest);

	gboolean is_terminal = (g_strcmp0(state, "completed") == 0 ||
	                         g_strcmp0(state, "failed") == 0 ||
	                         g_strcmp0(state, "cancelled") == 0);

	g_variant_unref(result);

	if (is_terminal) {
		ctx->poll_timer_id = 0;
		return G_SOURCE_REMOVE;
	}

	return G_SOURCE_CONTINUE;
}

static void test_update_progress(struct test_runner_ctx *ctx,
                                 const char *state,
                                 guint32 progress_pct,
                                 const char *subtest)
{
	gtk_progress_bar_set_fraction(
		GTK_PROGRESS_BAR(ctx->progress_bar),
		progress_pct / 100.0);

	if (subtest && subtest[0]) {
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest), subtest);
		test_set_status(ctx, subtest);
	}

	if (g_strcmp0(state, "completed") == 0) {
		test_on_completed(ctx, TRUE, "Test completed successfully");
	} else if (g_strcmp0(state, "failed") == 0) {
		test_on_completed(ctx, FALSE, "Test failed");
	} else if (g_strcmp0(state, "cancelled") == 0) {
		test_cleanup_timer(ctx);
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Cancelled");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest), "Cancelled");
		test_set_status(ctx, "Test cancelled by user");
		test_set_running(ctx, FALSE);
	} else {
		char pct_buf[32];
		snprintf(pct_buf, sizeof(pct_buf), "Running... %u%%",
		         progress_pct);
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), pct_buf);
	}
}

static void test_on_completed(struct test_runner_ctx *ctx,
                              gboolean success,
                              const char *summary)
{
	test_cleanup_timer(ctx);

	if (success) {
		gtk_progress_bar_set_fraction(
			GTK_PROGRESS_BAR(ctx->progress_bar), 1.0);
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Complete");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   summary ? summary : "Complete");
		test_set_status(ctx, "Fetching results...");
	} else {
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Failed");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   summary ? summary : "Failed");
		test_set_status(ctx, summary ? summary : "Test failed");
	}

	test_fetch_result(ctx);
	test_set_running(ctx, FALSE);
}

static void test_fetch_result(struct test_runner_ctx *ctx)
{
	if (!ctx->test_id)
		return;

	GVariant *result = odl_tray_dbus_call_sync(
		"GetTestResult",
		g_variant_new("(s)", ctx->test_id));

	if (!result) {
		test_set_status(ctx, "Could not retrieve test results.");
		test_log(ctx,
			"[WARN] Could not retrieve detailed test results.\n");
		return;
	}

	const gchar *result_json = NULL;
	g_variant_get(result, "(s)", &result_json);

	if (!result_json || !result_json[0]) {
		test_set_status(ctx, "No result data returned.");
		test_log(ctx, "No result data returned.\n");
		g_variant_unref(result);
		return;
	}

	char *output = odl_parse_extract_json_output(result_json);
	if (output) {
		ctx->raw_output = output;
		test_log(ctx, output);
	} else {
		ctx->raw_output = g_strdup(result_json);
		test_log(ctx, result_json);
	}

	g_variant_unref(result);

	ctx->parsed = g_new0(struct odl_parsed_test_result, 1);
	odl_parse_test_output(ctx->raw_output, ctx->test_type, ctx->parsed);

	build_result_tabs(ctx);
}

/* Wrap content in a scrolled window */
static GtkWidget *create_chart_scroll(GtkWidget *content)
{
	GtkWidget *sw = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
	                               GTK_POLICY_NEVER,
	                               GTK_POLICY_AUTOMATIC);
	gtk_container_add(GTK_CONTAINER(sw), content);
	return sw;
}

static void add_bandwidth_tab(struct test_runner_ctx *ctx,
                               GtkWidget *notebook)
{
	struct odl_parsed_test_result *p = ctx->parsed;
	if (!p->bandwidth.valid)
		return;

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
	gtk_container_set_border_width(GTK_CONTAINER(vbox), 8);

	GtkWidget *tp = odl_chart_create_throughput(&p->bandwidth);
	gtk_box_pack_start(GTK_BOX(vbox), tp, FALSE, FALSE, 0);

	for (int i = 0; i < p->num_stats; i++) {
		if (strstr(p->stats[i].label, "Bandwidth") ||
		    strstr(p->stats[i].label, "Throughput")) {
			GtkWidget *sc =
				odl_chart_create_stats_card(&p->stats[i]);
			gtk_box_pack_start(GTK_BOX(vbox), sc,
			                   FALSE, FALSE, 0);
		}
	}

	GtkWidget *sw = create_chart_scroll(vbox);
	gtk_notebook_append_page(GTK_NOTEBOOK(notebook), sw,
	                         gtk_label_new("Bandwidth"));
}

static void add_stats_tab(struct test_runner_ctx *ctx,
                           GtkWidget *notebook,
                           int stats_idx, int hist_idx,
                           const char *tab_label)
{
	struct odl_parsed_test_result *p = ctx->parsed;

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
	gtk_container_set_border_width(GTK_CONTAINER(vbox), 8);

	if (stats_idx >= 0 && stats_idx < p->num_stats &&
	    p->stats[stats_idx].valid) {
		GtkWidget *sc =
			odl_chart_create_stats_card(&p->stats[stats_idx]);
		gtk_box_pack_start(GTK_BOX(vbox), sc, FALSE, FALSE, 0);
	}

	if (hist_idx >= 0 && hist_idx < p->num_histograms &&
	    p->histograms[hist_idx].valid) {
		GtkWidget *hc = odl_chart_create_histogram(
			&p->histograms[hist_idx], tab_label);
		gtk_box_pack_start(GTK_BOX(vbox), hc, FALSE, FALSE, 0);
	}

	GtkWidget *sw = create_chart_scroll(vbox);
	gtk_notebook_append_page(GTK_NOTEBOOK(notebook), sw,
	                         gtk_label_new(tab_label));
}

static void add_raw_output_tab(struct test_runner_ctx *ctx,
                                GtkWidget *notebook)
{
	GtkWidget *sw = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
	                               GTK_POLICY_AUTOMATIC,
	                               GTK_POLICY_AUTOMATIC);

	GtkWidget *tv = gtk_text_view_new();
	gtk_text_view_set_editable(GTK_TEXT_VIEW(tv), FALSE);
	gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(tv), FALSE);
	gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(tv), GTK_WRAP_WORD_CHAR);
	gtk_text_view_set_left_margin(GTK_TEXT_VIEW(tv), 6);
	gtk_text_view_set_right_margin(GTK_TEXT_VIEW(tv), 6);
	gtk_text_view_set_top_margin(GTK_TEXT_VIEW(tv), 4);
	gtk_text_view_set_bottom_margin(GTK_TEXT_VIEW(tv), 4);

	PangoFontDescription *fd =
		pango_font_description_from_string("Monospace 9");
	gtk_widget_override_font(tv, fd);
	pango_font_description_free(fd);

	const char *text = ctx->raw_output ? ctx->raw_output
	                                   : ctx->log_buffer->str;
	GtkTextBuffer *buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(tv));
	gtk_text_buffer_set_text(buf, text, -1);

	gtk_container_add(GTK_CONTAINER(sw), tv);
	gtk_notebook_append_page(GTK_NOTEBOOK(notebook), sw,
	                         gtk_label_new("Raw Output"));
}

static void build_result_tabs(struct test_runner_ctx *ctx)
{
	struct odl_parsed_test_result *p = ctx->parsed;
	if (!p)
		return;

	gboolean has_data = p->bandwidth.valid || p->num_stats > 0 ||
	                    p->num_histograms > 0 || p->load_cmp.valid ||
	                    p->jitter_summary.valid;

	ctx->notebook = gtk_notebook_new();
	gtk_notebook_set_scrollable(GTK_NOTEBOOK(ctx->notebook), TRUE);

	if (has_data) {
		if (p->bandwidth.valid)
			add_bandwidth_tab(ctx, ctx->notebook);

		int tab_count = p->num_stats > p->num_histograms
		              ? p->num_stats : p->num_histograms;
		for (int i = 0; i < tab_count; i++) {
			const char *label = "Results";
			if (i < p->num_stats && p->stats[i].label[0]) {
				label = p->stats[i].label;
				if (p->bandwidth.valid &&
				    (strstr(label, "Bandwidth") ||
				     strstr(label, "Throughput")))
					continue;
			}
			add_stats_tab(ctx, ctx->notebook, i, i, label);
		}

		if (p->load_cmp.valid) {
			GtkWidget *vbox =
				gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
			gtk_container_set_border_width(
				GTK_CONTAINER(vbox), 8);
			GtkWidget *lc = odl_chart_create_load_comparison(
				&p->load_cmp);
			gtk_box_pack_start(GTK_BOX(vbox), lc,
			                   FALSE, FALSE, 0);
			GtkWidget *sw = create_chart_scroll(vbox);
			gtk_notebook_append_page(
				GTK_NOTEBOOK(ctx->notebook), sw,
				gtk_label_new("Load Impact"));
		}

		if (p->jitter_summary.valid) {
			GtkWidget *vbox =
				gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
			gtk_container_set_border_width(
				GTK_CONTAINER(vbox), 8);
			GtkWidget *js = odl_chart_create_jitter_summary(
				&p->jitter_summary);
			gtk_box_pack_start(GTK_BOX(vbox), js,
			                   FALSE, FALSE, 0);
			GtkWidget *sw = create_chart_scroll(vbox);
			gtk_notebook_append_page(
				GTK_NOTEBOOK(ctx->notebook), sw,
				gtk_label_new("Jitter"));
		}
	}

	add_raw_output_tab(ctx, ctx->notebook);

	if (ctx->lbl_status)
		gtk_widget_hide(ctx->lbl_status);

	gtk_box_pack_start(GTK_BOX(ctx->results_box), ctx->notebook,
	                   TRUE, TRUE, 0);
	gtk_widget_show_all(ctx->notebook);
}

static void on_cancel_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct test_runner_ctx *ctx = user_data;

	if (!ctx || !ctx->test_id || !ctx->running)
		return;

	GVariant *result = odl_tray_dbus_call_sync(
		"CancelTest",
		g_variant_new("(s)", ctx->test_id));

	if (result) {
		gboolean success;
		g_variant_get(result, "(b)", &success);
		g_variant_unref(result);

		if (success) {
			test_set_status(ctx, "Cancel request sent...");
		} else {
			test_set_status(ctx,
				"Cancel was not accepted by daemon.");
		}
	} else {
		test_set_status(ctx, "Failed to send cancel request.");
	}
}

static void on_close_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct test_runner_ctx *ctx = user_data;

	if (ctx && ctx->window)
		gtk_widget_destroy(ctx->window);
}

static void on_run_again_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct test_runner_ctx *ctx = user_data;

	if (!ctx || ctx->running)
		return;

	test_start(ctx);
}

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
	(void)widget;
	(void)user_data;

	if (!s_ctx)
		return;

	test_cleanup_timer(s_ctx);

	if (s_ctx->signal_handler_id > 0 && g_daemon_proxy) {
		g_signal_handler_disconnect(g_daemon_proxy,
		                            s_ctx->signal_handler_id);
		s_ctx->signal_handler_id = 0;
	}

	g_free(s_ctx->test_type);
	g_free(s_ctx->test_id);
	g_free(s_ctx->parsed);
	g_free(s_ctx->raw_output);
	if (s_ctx->log_buffer)
		g_string_free(s_ctx->log_buffer, TRUE);

	g_test_window = NULL;
	g_free(s_ctx);
	s_ctx = NULL;
}

void odl_tray_tests_show(int device_index, const char *test_type)
{
	if (g_test_window && s_ctx) {
		if (s_ctx->running) {
			gtk_window_present(GTK_WINDOW(g_test_window));
			return;
		}

		s_ctx->device_index = device_index;
		g_free(s_ctx->test_type);
		s_ctx->test_type = g_strdup(test_type);

		gtk_window_present(GTK_WINDOW(g_test_window));
		test_start(s_ctx);
		return;
	}

	s_ctx = g_new0(struct test_runner_ctx, 1);
	s_ctx->device_index = device_index;
	s_ctx->test_type = g_strdup(test_type);
	s_ctx->log_buffer = g_string_new(NULL);

	GtkCssProvider *css = gtk_css_provider_new();
	gtk_css_provider_load_from_data(css, test_runner_css, -1, NULL);
	gtk_style_context_add_provider_for_screen(
		gdk_screen_get_default(),
		GTK_STYLE_PROVIDER(css),
		GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	g_object_unref(css);

	GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window),
	                     "OdinLink TB5 - Test Runner");
	gtk_window_set_default_size(GTK_WINDOW(window), 640, 580);
	gtk_window_set_resizable(GTK_WINDOW(window), TRUE);
	gtk_container_set_border_width(GTK_CONTAINER(window), 12);

	GtkStyleContext *wsc = gtk_widget_get_style_context(window);
	gtk_style_context_add_class(wsc, "test-runner");

	g_signal_connect(window, "destroy",
	                 G_CALLBACK(on_window_destroy), NULL);

	s_ctx->window = window;
	g_test_window = window;

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
	gtk_container_add(GTK_CONTAINER(window), vbox);

	s_ctx->lbl_title = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(s_ctx->lbl_title),
	                     "<span size='large' weight='bold' "
	                     "color='#4CAF50'>"
	                     "OdinLink TB5 - Test Runner</span>");
	gtk_widget_set_halign(s_ctx->lbl_title, GTK_ALIGN_START);
	gtk_box_pack_start(GTK_BOX(vbox), s_ctx->lbl_title,
	                   FALSE, FALSE, 4);

	char info_buf[256];
	snprintf(info_buf, sizeof(info_buf),
	         "Device: %d  |  Test: %s", device_index, test_type);
	s_ctx->lbl_info = gtk_label_new(info_buf);
	gtk_widget_set_halign(s_ctx->lbl_info, GTK_ALIGN_START);
	GtkStyleContext *isc =
		gtk_widget_get_style_context(s_ctx->lbl_info);
	gtk_style_context_add_class(isc, "info-label");
	gtk_box_pack_start(GTK_BOX(vbox), s_ctx->lbl_info,
	                   FALSE, FALSE, 0);

	s_ctx->progress_bar = gtk_progress_bar_new();
	gtk_progress_bar_set_show_text(
		GTK_PROGRESS_BAR(s_ctx->progress_bar), TRUE);
	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(s_ctx->progress_bar), "Idle");
	gtk_box_pack_start(GTK_BOX(vbox), s_ctx->progress_bar,
	                   FALSE, FALSE, 0);

	s_ctx->lbl_subtest = gtk_label_new("Waiting to start...");
	gtk_widget_set_halign(s_ctx->lbl_subtest, GTK_ALIGN_START);
	gtk_label_set_ellipsize(GTK_LABEL(s_ctx->lbl_subtest),
	                        PANGO_ELLIPSIZE_END);
	GtkStyleContext *ssc =
		gtk_widget_get_style_context(s_ctx->lbl_subtest);
	gtk_style_context_add_class(ssc, "subtest-label");
	gtk_box_pack_start(GTK_BOX(vbox), s_ctx->lbl_subtest,
	                   FALSE, FALSE, 0);

	gtk_box_pack_start(GTK_BOX(vbox),
	                   gtk_separator_new(GTK_ORIENTATION_HORIZONTAL),
	                   FALSE, FALSE, 4);

	s_ctx->results_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_box_pack_start(GTK_BOX(vbox), s_ctx->results_box,
	                   TRUE, TRUE, 0);

	s_ctx->lbl_status = gtk_label_new("Waiting to start...");
	gtk_widget_set_valign(s_ctx->lbl_status, GTK_ALIGN_CENTER);
	gtk_widget_set_halign(s_ctx->lbl_status, GTK_ALIGN_CENTER);
	gtk_box_pack_start(GTK_BOX(s_ctx->results_box), s_ctx->lbl_status,
	                   TRUE, FALSE, 0);

	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
	gtk_widget_set_halign(hbox, GTK_ALIGN_END);
	gtk_box_pack_end(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

	s_ctx->btn_cancel = gtk_button_new_with_label("Cancel");
	g_signal_connect(s_ctx->btn_cancel, "clicked",
	                 G_CALLBACK(on_cancel_clicked), s_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), s_ctx->btn_cancel,
	                   FALSE, FALSE, 0);

	s_ctx->btn_run_again = gtk_button_new_with_label("Run Again");
	g_signal_connect(s_ctx->btn_run_again, "clicked",
	                 G_CALLBACK(on_run_again_clicked), s_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), s_ctx->btn_run_again,
	                   FALSE, FALSE, 0);

	s_ctx->btn_close = gtk_button_new_with_label("Close");
	g_signal_connect(s_ctx->btn_close, "clicked",
	                 G_CALLBACK(on_close_clicked), s_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), s_ctx->btn_close,
	                   FALSE, FALSE, 0);

	if (g_daemon_proxy) {
		s_ctx->signal_handler_id = g_signal_connect(
			g_daemon_proxy, "g-signal",
			G_CALLBACK(on_daemon_signal_for_tests), s_ctx);
	}

	gtk_widget_show_all(window);
	test_start(s_ctx);
}
