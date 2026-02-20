/*
 * OdinLink TB5 Tray - Test Overview Dashboard
 *
 * Compact summary of all test results on one page.
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray.h"
#include "odl_tb5_tray_test_parse.h"
#include <stdio.h>
#include <string.h>

static const char *overview_css =
	".test-overview {"
	"  background-color: #282828;"
	"  color: #E6E6E6;"
	"}"
	".test-overview label {"
	"  color: #E6E6E6;"
	"}"
	".test-overview .title-label {"
	"  color: #4CAF50;"
	"}"
	".test-overview .info-label {"
	"  color: #9E9E9E;"
	"}"
	".test-overview .section-title {"
	"  color: #4CAF50;"
	"  font-weight: bold;"
	"}"
	".test-overview .metric-value {"
	"  color: #FFFFFF;"
	"  font-weight: bold;"
	"}"
	".test-overview .metric-label {"
	"  color: #9E9E9E;"
	"}"
	".test-overview .card {"
	"  background-color: #363636;"
	"  border-radius: 6px;"
	"  padding: 8px;"
	"}"
	".test-overview .card-pass {"
	"  border-left: 4px solid #4CAF50;"
	"}"
	".test-overview .card-fail {"
	"  border-left: 4px solid #F44336;"
	"}"
	".test-overview .card-pending {"
	"  border-left: 4px solid #555555;"
	"}"
	".test-overview progressbar trough {"
	"  background-color: #363636;"
	"  min-height: 16px;"
	"  border-radius: 4px;"
	"}"
	".test-overview progressbar progress {"
	"  background-color: #4CAF50;"
	"  min-height: 16px;"
	"  border-radius: 4px;"
	"}"
	".test-overview button {"
	"  background-image: none;"
	"  background-color: #363636;"
	"  color: #E6E6E6;"
	"  border: 1px solid #555555;"
	"  border-radius: 4px;"
	"  padding: 4px 12px;"
	"  text-shadow: none;"
	"  box-shadow: none;"
	"}"
	".test-overview button label {"
	"  color: #E6E6E6;"
	"}"
	".test-overview button:hover {"
	"  background-image: none;"
	"  background-color: #4CAF50;"
	"  color: #282828;"
	"}"
	".test-overview button:hover label {"
	"  color: #282828;"
	"}"
	".test-overview separator {"
	"  background-color: #555555;"
	"  min-height: 1px;"
	"}";

struct overview_ctx {
	int          device_index;
	char        *test_id;

	GtkWidget   *window;
	GtkWidget   *lbl_info;
	GtkWidget   *progress_bar;
	GtkWidget   *lbl_subtest;
	GtkWidget   *cards_box;
	GtkWidget   *btn_run;
	GtkWidget   *btn_close;

	/* Per-section cards (updated when results arrive) */
	GtkWidget   *card_bw;
	GtkWidget   *card_lat;
	GtkWidget   *card_jitter;
	GtkWidget   *card_load;
	GtkWidget   *card_mimo;

	guint        poll_timer_id;
	gulong       signal_handler_id;
	gboolean     running;
};

GtkWidget *g_overview_window = NULL;
static struct overview_ctx *s_ov_ctx = NULL;

static void     overview_run(struct overview_ctx *ctx);
static void     overview_cleanup_timer(struct overview_ctx *ctx);
static gboolean overview_poll_timer(gpointer user_data);
static void     overview_on_completed(struct overview_ctx *ctx,
                                      gboolean success);
static void     overview_build_cards(struct overview_ctx *ctx);

/* Create a styled metric line: "Label: Value" */
static GtkWidget *make_metric(const char *label, const char *value)
{
	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);

	char markup[256];
	snprintf(markup, sizeof(markup),
	         "<span color='#9E9E9E'>%s</span> "
	         "<span color='#FFFFFF' weight='bold'>%s</span>",
	         label, value);

	GtkWidget *lbl = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(lbl), markup);
	gtk_widget_set_halign(lbl, GTK_ALIGN_START);
	gtk_box_pack_start(GTK_BOX(hbox), lbl, FALSE, FALSE, 0);

	return hbox;
}

/* Create a section card with title and content lines */
static GtkWidget *make_card(const char *title, const char *css_class,
                             GtkWidget **content_box_out)
{
	GtkWidget *frame = gtk_frame_new(NULL);
	gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);

	GtkStyleContext *sc = gtk_widget_get_style_context(frame);
	gtk_style_context_add_class(sc, "card");
	gtk_style_context_add_class(sc, css_class);

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
	gtk_container_set_border_width(GTK_CONTAINER(vbox), 8);
	gtk_container_add(GTK_CONTAINER(frame), vbox);

	char title_markup[256];
	snprintf(title_markup, sizeof(title_markup),
	         "<span color='#4CAF50' weight='bold'>%s</span>", title);

	GtkWidget *lbl_title = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(lbl_title), title_markup);
	gtk_widget_set_halign(lbl_title, GTK_ALIGN_START);
	gtk_box_pack_start(GTK_BOX(vbox), lbl_title, FALSE, FALSE, 0);

	if (content_box_out)
		*content_box_out = vbox;

	return frame;
}

/* Build initial placeholder cards */
static void overview_build_placeholders(struct overview_ctx *ctx)
{
	GtkWidget *content;
	const char *pending = "card-pending";

	/* Bandwidth */
	ctx->card_bw = make_card("Bandwidth", pending, &content);
	gtk_box_pack_start(GTK_BOX(content),
	                   make_metric("Throughput:", "-- Gb/s"),
	                   FALSE, FALSE, 0);
	gtk_box_pack_start(GTK_BOX(ctx->cards_box), ctx->card_bw,
	                   FALSE, FALSE, 4);

	/* Latency */
	ctx->card_lat = make_card("Latency", pending, &content);
	gtk_box_pack_start(GTK_BOX(content),
	                   make_metric("Avg:", "-- us"),
	                   FALSE, FALSE, 0);
	gtk_box_pack_start(GTK_BOX(ctx->cards_box), ctx->card_lat,
	                   FALSE, FALSE, 4);

	/* Jitter */
	ctx->card_jitter = make_card("Jitter", pending, &content);
	gtk_box_pack_start(GTK_BOX(content),
	                   make_metric("Avg RTT:", "-- us"),
	                   FALSE, FALSE, 0);
	gtk_box_pack_start(GTK_BOX(ctx->cards_box), ctx->card_jitter,
	                   FALSE, FALSE, 4);

	/* Latency Under Load */
	ctx->card_load = make_card("Latency Under Load", pending, &content);
	gtk_box_pack_start(GTK_BOX(content),
	                   make_metric("Degradation:", "--%"),
	                   FALSE, FALSE, 0);
	gtk_box_pack_start(GTK_BOX(ctx->cards_box), ctx->card_load,
	                   FALSE, FALSE, 4);

	/* MIMO */
	ctx->card_mimo = make_card("MIMO", pending, &content);
	gtk_box_pack_start(GTK_BOX(content),
	                   make_metric("Aggregate:", "-- Gb/s"),
	                   FALSE, FALSE, 0);
	gtk_box_pack_start(GTK_BOX(ctx->cards_box), ctx->card_mimo,
	                   FALSE, FALSE, 4);

	gtk_widget_show_all(ctx->cards_box);
}

/* Replace a card with populated data */
static void replace_card(struct overview_ctx *ctx,
                          GtkWidget **old_card,
                          GtkWidget *new_card)
{
	if (*old_card) {
		gtk_container_remove(GTK_CONTAINER(ctx->cards_box), *old_card);
	}
	*old_card = new_card;
	gtk_box_pack_start(GTK_BOX(ctx->cards_box), new_card,
	                   FALSE, FALSE, 4);
	/* Reorder to maintain consistent order */
	int pos = 0;
	if (new_card == ctx->card_bw)     pos = 0;
	if (new_card == ctx->card_lat)    pos = 1;
	if (new_card == ctx->card_jitter) pos = 2;
	if (new_card == ctx->card_load)   pos = 3;
	if (new_card == ctx->card_mimo)   pos = 4;
	gtk_box_reorder_child(GTK_BOX(ctx->cards_box), new_card, pos);
	gtk_widget_show_all(new_card);
}

static void overview_build_cards(struct overview_ctx *ctx)
{
	if (!ctx->test_id)
		return;

	GVariant *result = odl_tray_dbus_call_sync(
		"GetTestResult",
		g_variant_new("(s)", ctx->test_id));

	if (!result)
		return;

	const gchar *result_json = NULL;
	g_variant_get(result, "(s)", &result_json);

	if (!result_json || !result_json[0]) {
		g_variant_unref(result);
		return;
	}

	char *output = odl_parse_extract_json_output(result_json);
	if (!output) {
		output = g_strdup(result_json);
	}
	g_variant_unref(result);

	struct odl_parsed_test_result parsed;
	odl_parse_test_output(output, "all", &parsed);

	char buf[256], buf2[128], buf3[128];
	GtkWidget *content, *card;

	/* Bandwidth card */
	if (parsed.bandwidth.valid) {
		card = make_card("Bandwidth", "card-pass", &content);
		snprintf(buf, sizeof(buf), "%.2f Gb/s (%.2f GB/s)",
		         parsed.bandwidth.gbps, parsed.bandwidth.gbytes_s);
		gtk_box_pack_start(GTK_BOX(content),
		                   make_metric("Throughput:", buf),
		                   FALSE, FALSE, 0);
		if (parsed.bandwidth.transferred[0]) {
			gtk_box_pack_start(GTK_BOX(content),
			                   make_metric("Transferred:",
			                              parsed.bandwidth.transferred),
			                   FALSE, FALSE, 0);
		}
		replace_card(ctx, &ctx->card_bw, card);
	}

	/* Latency card — use first stats block that looks like latency */
	for (int i = 0; i < parsed.num_stats; i++) {
		if (!parsed.stats[i].valid)
			continue;
		/* Skip stats with "Bandwidth"/"Throughput" in the name */
		if (strstr(parsed.stats[i].label, "Bandwidth") ||
		    strstr(parsed.stats[i].label, "Throughput") ||
		    strstr(parsed.stats[i].label, "MIMO"))
			continue;

		/* Use first matching latency stats for the latency card */
		if (strstr(parsed.stats[i].label, "Latency") ||
		    strstr(parsed.stats[i].label, "RTT") ||
		    i == 0) {
			card = make_card(parsed.stats[i].label, "card-pass",
			                 &content);

			odl_parse_format_latency(parsed.stats[i].avg_ns,
			                         buf2, sizeof(buf2));
			odl_parse_format_latency(parsed.stats[i].min_ns,
			                         buf3, sizeof(buf3));
			snprintf(buf, sizeof(buf), "%s  |  Min: %s",
			         buf2, buf3);
			gtk_box_pack_start(GTK_BOX(content),
			                   make_metric("Avg:", buf),
			                   FALSE, FALSE, 0);

			odl_parse_format_latency(parsed.stats[i].max_ns,
			                         buf2, sizeof(buf2));
			odl_parse_format_latency(parsed.stats[i].p99_ns,
			                         buf3, sizeof(buf3));
			snprintf(buf, sizeof(buf), "%s  |  p99: %s",
			         buf2, buf3);
			gtk_box_pack_start(GTK_BOX(content),
			                   make_metric("Max:", buf),
			                   FALSE, FALSE, 0);

			snprintf(buf, sizeof(buf), "%zu",
			         parsed.stats[i].sample_count);
			gtk_box_pack_start(GTK_BOX(content),
			                   make_metric("Samples:", buf),
			                   FALSE, FALSE, 0);

			replace_card(ctx, &ctx->card_lat, card);
			break;
		}
	}

	/* Jitter card */
	if (parsed.jitter_summary.valid) {
		card = make_card("Jitter", "card-pass", &content);

		odl_parse_format_latency(parsed.jitter_summary.avg_rtt_ns,
		                         buf2, sizeof(buf2));
		odl_parse_format_latency(parsed.jitter_summary.avg_jitter_ns,
		                         buf3, sizeof(buf3));
		snprintf(buf, sizeof(buf), "%s  |  Avg Jitter: %s",
		         buf2, buf3);
		gtk_box_pack_start(GTK_BOX(content),
		                   make_metric("Avg RTT:", buf),
		                   FALSE, FALSE, 0);

		odl_parse_format_latency(parsed.jitter_summary.max_jitter_ns,
		                         buf2, sizeof(buf2));
		snprintf(buf, sizeof(buf), "%s  |  Jitter/RTT: %.1f%%",
		         buf2, parsed.jitter_summary.jitter_rtt_ratio_pct);
		gtk_box_pack_start(GTK_BOX(content),
		                   make_metric("Max Jitter:", buf),
		                   FALSE, FALSE, 0);

		replace_card(ctx, &ctx->card_jitter, card);
	}

	/* Latency Under Load card */
	if (parsed.load_cmp.valid) {
		card = make_card("Latency Under Load", "card-pass", &content);

		snprintf(buf, sizeof(buf), "%.2f us  |  Under Load: %.2f us",
		         parsed.load_cmp.idle_us, parsed.load_cmp.load_us);
		gtk_box_pack_start(GTK_BOX(content),
		                   make_metric("Idle:", buf),
		                   FALSE, FALSE, 0);

		snprintf(buf, sizeof(buf), "%.1f%%",
		         parsed.load_cmp.degradation_pct);
		gtk_box_pack_start(GTK_BOX(content),
		                   make_metric("Degradation:", buf),
		                   FALSE, FALSE, 0);

		replace_card(ctx, &ctx->card_load, card);
	}

	/* MIMO — check for a second bandwidth measurement or MIMO stats */
	{
		const char *mimo_section = strstr(output, "MIMO");
		if (!mimo_section)
			mimo_section = strstr(output, "mimo");

		if (mimo_section) {
			const char *tp = strstr(mimo_section, "Aggregate:");
			if (!tp)
				tp = strstr(mimo_section, "Throughput:");

			if (tp) {
				double gbps = 0, gbytes = 0;
				tp += strcspn(tp, ":");
				tp++;
				if (sscanf(tp, " %lf Gb/s (%lf GB/s)",
				           &gbps, &gbytes) >= 1) {
					card = make_card("MIMO", "card-pass",
					                 &content);
					snprintf(buf, sizeof(buf),
					         "%.2f Gb/s (%.2f GB/s)",
					         gbps, gbytes);
					gtk_box_pack_start(GTK_BOX(content),
					                   make_metric("Aggregate:",
					                              buf),
					                   FALSE, FALSE, 0);
					replace_card(ctx, &ctx->card_mimo, card);
				}
			}
		}
	}

	g_free(output);
}

static void overview_cleanup_timer(struct overview_ctx *ctx)
{
	if (ctx->poll_timer_id > 0) {
		g_source_remove(ctx->poll_timer_id);
		ctx->poll_timer_id = 0;
	}
}

static void overview_set_running(struct overview_ctx *ctx, gboolean running)
{
	ctx->running = running;
	gtk_widget_set_sensitive(ctx->btn_run, !running);
}

/* D-Bus signal callback */
static void on_daemon_signal_for_overview(GDBusProxy  *proxy,
                                          const gchar *sender_name,
                                          const gchar *signal_name,
                                          GVariant    *parameters,
                                          gpointer     user_data)
{
	(void)proxy;
	(void)sender_name;
	struct overview_ctx *ctx = user_data;

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
			}

			char pct_buf[32];
			snprintf(pct_buf, sizeof(pct_buf), "Running... %u%%",
			         progress);
			gtk_progress_bar_set_text(
				GTK_PROGRESS_BAR(ctx->progress_bar), pct_buf);
		}

	} else if (g_strcmp0(signal_name, "TestCompleted") == 0) {
		const gchar *test_id = NULL;
		gboolean success;
		const gchar *summary = NULL;

		g_variant_get(parameters, "(sbs)", &test_id, &success,
		              &summary);

		if (g_strcmp0(test_id, ctx->test_id) == 0) {
			overview_on_completed(ctx, success);
		}
	}
}

static gboolean overview_poll_timer(gpointer user_data)
{
	struct overview_ctx *ctx = user_data;

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

	gtk_progress_bar_set_fraction(
		GTK_PROGRESS_BAR(ctx->progress_bar),
		progress_pct / 100.0);

	if (current_subtest && current_subtest[0]) {
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   current_subtest);
	}

	gboolean is_terminal = (g_strcmp0(state, "completed") == 0 ||
	                         g_strcmp0(state, "failed") == 0 ||
	                         g_strcmp0(state, "cancelled") == 0);

	if (is_terminal) {
		gboolean success = (g_strcmp0(state, "completed") == 0);
		overview_on_completed(ctx, success);
		g_variant_unref(result);
		ctx->poll_timer_id = 0;
		return G_SOURCE_REMOVE;
	}

	char pct_buf[32];
	snprintf(pct_buf, sizeof(pct_buf), "Running... %u%%", progress_pct);
	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(ctx->progress_bar), pct_buf);

	g_variant_unref(result);
	return G_SOURCE_CONTINUE;
}

static void overview_on_completed(struct overview_ctx *ctx,
                                  gboolean success)
{
	overview_cleanup_timer(ctx);

	if (success) {
		gtk_progress_bar_set_fraction(
			GTK_PROGRESS_BAR(ctx->progress_bar), 1.0);
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Complete");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   "All tests completed");
	} else {
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Failed");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   "Tests failed");
	}

	overview_build_cards(ctx);
	overview_set_running(ctx, FALSE);
}

static void overview_run(struct overview_ctx *ctx)
{
	/* Reset cards to pending */
	GList *children = gtk_container_get_children(
		GTK_CONTAINER(ctx->cards_box));
	for (GList *l = children; l; l = l->next)
		gtk_widget_destroy(GTK_WIDGET(l->data));
	g_list_free(children);
	ctx->card_bw = ctx->card_lat = ctx->card_jitter = NULL;
	ctx->card_load = ctx->card_mimo = NULL;
	overview_build_placeholders(ctx);

	gtk_progress_bar_set_fraction(
		GTK_PROGRESS_BAR(ctx->progress_bar), 0.0);
	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(ctx->progress_bar), "Starting...");
	gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest), "Initializing...");
	overview_set_running(ctx, TRUE);

	g_free(ctx->test_id);
	ctx->test_id = NULL;

	GVariant *result = odl_tray_dbus_call_sync(
		"RunTest",
		g_variant_new("(is)", ctx->device_index, "all"));

	if (!result) {
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Error");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   "Failed to start tests. Is daemon running?");
		overview_set_running(ctx, FALSE);
		return;
	}

	const gchar *test_id = NULL;
	g_variant_get(result, "(s)", &test_id);

	if (!test_id || !test_id[0]) {
		gtk_progress_bar_set_text(
			GTK_PROGRESS_BAR(ctx->progress_bar), "Error");
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   "Daemon returned empty test ID");
		overview_set_running(ctx, FALSE);
		g_variant_unref(result);
		return;
	}

	ctx->test_id = g_strdup(test_id);
	g_variant_unref(result);

	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(ctx->progress_bar), "Running...");

	overview_cleanup_timer(ctx);
	ctx->poll_timer_id = g_timeout_add(500, overview_poll_timer, ctx);
}

static void on_run_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct overview_ctx *ctx = user_data;

	if (!ctx || ctx->running)
		return;

	/* Find first available device */
	GVariant *devices = odl_tray_dbus_call_sync("GetDevices", NULL);
	if (devices) {
		GVariantIter *iter = NULL;
		g_variant_get(devices, "(a(iss))", &iter);
		gint32 dev_index = -1;
		const gchar *s = NULL, *n = NULL;
		if (g_variant_iter_next(iter, "(i&s&s)", &dev_index, &s, &n))
			ctx->device_index = (int)dev_index;
		g_variant_iter_free(iter);
		g_variant_unref(devices);

		if (dev_index < 0) {
			gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
			                   "No devices available");
			return;
		}
	} else {
		gtk_label_set_text(GTK_LABEL(ctx->lbl_subtest),
		                   "Cannot reach daemon");
		return;
	}

	char info_buf[128];
	snprintf(info_buf, sizeof(info_buf),
	         "Device: %d  |  Running all tests...", ctx->device_index);
	gtk_label_set_text(GTK_LABEL(ctx->lbl_info), info_buf);

	overview_run(ctx);
}

static void on_close_clicked(GtkButton *button, gpointer user_data)
{
	(void)button;
	struct overview_ctx *ctx = user_data;

	if (ctx && ctx->window)
		gtk_widget_destroy(ctx->window);
}

static void on_window_destroy(GtkWidget *widget, gpointer user_data)
{
	(void)widget;
	(void)user_data;

	if (!s_ov_ctx)
		return;

	overview_cleanup_timer(s_ov_ctx);

	if (s_ov_ctx->signal_handler_id > 0 && g_daemon_proxy) {
		g_signal_handler_disconnect(g_daemon_proxy,
		                            s_ov_ctx->signal_handler_id);
		s_ov_ctx->signal_handler_id = 0;
	}

	g_free(s_ov_ctx->test_id);
	g_overview_window = NULL;
	g_free(s_ov_ctx);
	s_ov_ctx = NULL;
}

void odl_tray_overview_show(void)
{
	if (g_overview_window && s_ov_ctx) {
		gtk_window_present(GTK_WINDOW(g_overview_window));
		return;
	}

	s_ov_ctx = g_new0(struct overview_ctx, 1);
	s_ov_ctx->device_index = -1;

	GtkCssProvider *css = gtk_css_provider_new();
	gtk_css_provider_load_from_data(css, overview_css, -1, NULL);
	gtk_style_context_add_provider_for_screen(
		gdk_screen_get_default(),
		GTK_STYLE_PROVIDER(css),
		GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	g_object_unref(css);

	GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window),
	                     "OdinLink TB5 - Test Overview");
	gtk_window_set_default_size(GTK_WINDOW(window), 560, 620);
	gtk_window_set_resizable(GTK_WINDOW(window), TRUE);
	gtk_container_set_border_width(GTK_CONTAINER(window), 12);

	GtkStyleContext *wsc = gtk_widget_get_style_context(window);
	gtk_style_context_add_class(wsc, "test-overview");

	g_signal_connect(window, "destroy",
	                 G_CALLBACK(on_window_destroy), NULL);

	s_ov_ctx->window = window;
	g_overview_window = window;

	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
	gtk_container_add(GTK_CONTAINER(window), vbox);

	/* Title */
	GtkWidget *lbl_title = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(lbl_title),
	                     "<span size='large' weight='bold' "
	                     "color='#4CAF50'>"
	                     "OdinLink TB5 - Test Overview</span>");
	gtk_widget_set_halign(lbl_title, GTK_ALIGN_START);
	gtk_box_pack_start(GTK_BOX(vbox), lbl_title, FALSE, FALSE, 4);

	/* Info line */
	s_ov_ctx->lbl_info = gtk_label_new("Click 'Run All Tests' to start");
	gtk_widget_set_halign(s_ov_ctx->lbl_info, GTK_ALIGN_START);
	GtkStyleContext *isc =
		gtk_widget_get_style_context(s_ov_ctx->lbl_info);
	gtk_style_context_add_class(isc, "info-label");
	gtk_box_pack_start(GTK_BOX(vbox), s_ov_ctx->lbl_info,
	                   FALSE, FALSE, 0);

	/* Progress bar */
	s_ov_ctx->progress_bar = gtk_progress_bar_new();
	gtk_progress_bar_set_show_text(
		GTK_PROGRESS_BAR(s_ov_ctx->progress_bar), TRUE);
	gtk_progress_bar_set_text(
		GTK_PROGRESS_BAR(s_ov_ctx->progress_bar), "Idle");
	gtk_box_pack_start(GTK_BOX(vbox), s_ov_ctx->progress_bar,
	                   FALSE, FALSE, 0);

	/* Subtest label */
	s_ov_ctx->lbl_subtest = gtk_label_new("Ready");
	gtk_widget_set_halign(s_ov_ctx->lbl_subtest, GTK_ALIGN_START);
	gtk_label_set_ellipsize(GTK_LABEL(s_ov_ctx->lbl_subtest),
	                        PANGO_ELLIPSIZE_END);
	gtk_box_pack_start(GTK_BOX(vbox), s_ov_ctx->lbl_subtest,
	                   FALSE, FALSE, 0);

	gtk_box_pack_start(GTK_BOX(vbox),
	                   gtk_separator_new(GTK_ORIENTATION_HORIZONTAL),
	                   FALSE, FALSE, 4);

	/* Scrollable cards area */
	GtkWidget *sw = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sw),
	                               GTK_POLICY_NEVER,
	                               GTK_POLICY_AUTOMATIC);
	gtk_box_pack_start(GTK_BOX(vbox), sw, TRUE, TRUE, 0);

	s_ov_ctx->cards_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_container_add(GTK_CONTAINER(sw), s_ov_ctx->cards_box);

	/* Build placeholder cards */
	overview_build_placeholders(s_ov_ctx);

	/* Buttons */
	gtk_box_pack_start(GTK_BOX(vbox),
	                   gtk_separator_new(GTK_ORIENTATION_HORIZONTAL),
	                   FALSE, FALSE, 4);

	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
	gtk_widget_set_halign(hbox, GTK_ALIGN_END);
	gtk_box_pack_end(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

	s_ov_ctx->btn_run = gtk_button_new_with_label("Run All Tests");
	g_signal_connect(s_ov_ctx->btn_run, "clicked",
	                 G_CALLBACK(on_run_clicked), s_ov_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), s_ov_ctx->btn_run,
	                   FALSE, FALSE, 0);

	s_ov_ctx->btn_close = gtk_button_new_with_label("Close");
	g_signal_connect(s_ov_ctx->btn_close, "clicked",
	                 G_CALLBACK(on_close_clicked), s_ov_ctx);
	gtk_box_pack_start(GTK_BOX(hbox), s_ov_ctx->btn_close,
	                   FALSE, FALSE, 0);

	/* Connect D-Bus signals */
	if (g_daemon_proxy) {
		s_ov_ctx->signal_handler_id = g_signal_connect(
			g_daemon_proxy, "g-signal",
			G_CALLBACK(on_daemon_signal_for_overview), s_ov_ctx);
	}

	gtk_widget_show_all(window);
}
