/*
 * OdinLink TB5 Tray - Test Result Charts (Cairo)
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray_test_charts.h"

#include <cairo.h>
#include <pango/pangocairo.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CLR_BG       0.157, 0.157, 0.157
#define CLR_CARD     0.212, 0.212, 0.212
#define CLR_TEXT     0.902, 0.902, 0.902
#define CLR_DIM      0.620, 0.620, 0.620
#define CLR_GREEN    0.298, 0.686, 0.314
#define CLR_BLUE     0.129, 0.588, 0.953
#define CLR_RED      0.957, 0.263, 0.212
#define CLR_AMBER    1.000, 0.757, 0.027

static void rounded_rect(cairo_t *cr, double x, double y,
                         double w, double h, double r)
{
	cairo_new_sub_path(cr);
	cairo_arc(cr, x + w - r, y + r, r, -M_PI / 2, 0);
	cairo_arc(cr, x + w - r, y + h - r, r, 0, M_PI / 2);
	cairo_arc(cr, x + r, y + h - r, r, M_PI / 2, M_PI);
	cairo_arc(cr, x + r, y + r, r, M_PI, 3 * M_PI / 2);
	cairo_close_path(cr);
}

static void draw_text(cairo_t *cr, double x, double y,
                      const char *text, const char *font_desc_str,
                      double r, double g, double b)
{
	PangoLayout *layout = pango_cairo_create_layout(cr);
	PangoFontDescription *fd = pango_font_description_from_string(font_desc_str);
	pango_layout_set_font_description(layout, fd);
	pango_layout_set_text(layout, text, -1);

	cairo_set_source_rgb(cr, r, g, b);
	cairo_move_to(cr, x, y);
	pango_cairo_show_layout(cr, layout);

	pango_font_description_free(fd);
	g_object_unref(layout);
}

static void draw_text_right(cairo_t *cr, double x, double y,
                            const char *text, const char *font_desc_str,
                            double r, double g, double b, double max_width)
{
	PangoLayout *layout = pango_cairo_create_layout(cr);
	PangoFontDescription *fd = pango_font_description_from_string(font_desc_str);
	pango_layout_set_font_description(layout, fd);
	pango_layout_set_text(layout, text, -1);

	int tw, th;
	pango_layout_get_pixel_size(layout, &tw, &th);

	cairo_set_source_rgb(cr, r, g, b);
	cairo_move_to(cr, x + max_width - tw, y);
	pango_cairo_show_layout(cr, layout);

	pango_font_description_free(fd);
	g_object_unref(layout);
}

/* Return bar color based on histogram bucket index */
static void bar_color_for_bucket(int index, int total,
                                 double *r, double *g, double *b)
{
	double t = (total > 1) ? (double)index / (total - 1) : 0.0;

	if (t < 0.33) {
		*r = 0.298; *g = 0.686; *b = 0.314;
	} else if (t < 0.66) {
		*r = 0.129; *g = 0.588; *b = 0.953;
	} else if (t < 0.85) {
		*r = 1.000; *g = 0.757; *b = 0.027;
	} else {
		*r = 0.957; *g = 0.263; *b = 0.212;
	}
}

struct hist_draw_ctx {
	struct odl_parsed_histogram *hist;
	char title[64];
};

static gboolean on_draw_histogram(GtkWidget *widget, cairo_t *cr,
                                  gpointer user_data)
{
	struct hist_draw_ctx *ctx = user_data;
	struct odl_parsed_histogram *hist = ctx->hist;

	int width = gtk_widget_get_allocated_width(widget);
	int height = gtk_widget_get_allocated_height(widget);

	cairo_set_source_rgb(cr, CLR_CARD);
	rounded_rect(cr, 0, 0, width, height, 8);
	cairo_fill(cr);

	if (!hist || !hist->valid || hist->num_buckets == 0)
		return FALSE;

	double left = 100;
	double right = width - 20;
	double top = 35;
	double bottom = height - 10;
	double bar_area = right - left;

	draw_text(cr, 12, 8, ctx->title, "Sans Bold 10", CLR_TEXT);

	uint64_t max_count = 0;
	for (int i = 0; i < hist->num_buckets; i++) {
		if (hist->count[i] > max_count)
			max_count = hist->count[i];
	}
	if (max_count == 0)
		return FALSE;

	double bar_height = (bottom - top) / hist->num_buckets;
	double gap = 3.0;

	for (int i = 0; i < hist->num_buckets; i++) {
		double y = top + i * bar_height;
		double bh = bar_height - gap;

		draw_text(cr, 8, y + bh / 2 - 7,
		          hist->label[i], "Monospace 8", CLR_DIM);

		double bar_w = bar_area * hist->count[i] / max_count;
		if (bar_w < 2 && hist->count[i] > 0)
			bar_w = 2;

		double r, g, b;
		bar_color_for_bucket(i, hist->num_buckets, &r, &g, &b);

		cairo_set_source_rgb(cr, r, g, b);
		rounded_rect(cr, left, y, bar_w, bh, 3);
		cairo_fill(cr);

		char info[64];
		snprintf(info, sizeof(info), "%lu (%.1f%%)",
		         (unsigned long)hist->count[i], hist->pct[i]);
		draw_text(cr, left + bar_w + 6, y + bh / 2 - 7,
		          info, "Monospace 8", CLR_TEXT);
	}

	return FALSE;
}

static void on_hist_ctx_destroy(gpointer data)
{
	g_free(data);
}

GtkWidget *odl_chart_create_histogram(struct odl_parsed_histogram *hist,
                                      const char *title)
{
	GtkWidget *da = gtk_drawing_area_new();
	gtk_widget_set_size_request(da, 500,
	                            hist ? 35 + hist->num_buckets * 24 + 10 : 100);

	struct hist_draw_ctx *ctx = g_new0(struct hist_draw_ctx, 1);
	ctx->hist = hist;
	g_strlcpy(ctx->title, title ? title : "Histogram", sizeof(ctx->title));

	g_signal_connect(da, "draw", G_CALLBACK(on_draw_histogram), ctx);
	g_object_set_data_full(G_OBJECT(da), "hist-ctx", ctx,
	                       on_hist_ctx_destroy);

	return da;
}

static gboolean on_draw_throughput(GtkWidget *widget, cairo_t *cr,
                                   gpointer user_data)
{
	struct odl_parsed_bandwidth *bw = user_data;
	int width = gtk_widget_get_allocated_width(widget);
	int height = gtk_widget_get_allocated_height(widget);

	cairo_set_source_rgb(cr, CLR_CARD);
	rounded_rect(cr, 0, 0, width, height, 8);
	cairo_fill(cr);

	draw_text(cr, 12, 8, "Throughput", "Sans Bold 10", CLR_TEXT);

	if (!bw || !bw->valid)
		return FALSE;

	double max_gbps = 80.0;
	double frac = bw->gbps / max_gbps;
	if (frac > 1.0)
		frac = 1.0;

	double bar_x = 12;
	double bar_y = 32;
	double bar_w = width - 24;
	double bar_h = 30;

	cairo_set_source_rgb(cr, 0.12, 0.12, 0.12);
	rounded_rect(cr, bar_x, bar_y, bar_w, bar_h, 6);
	cairo_fill(cr);

	double r, g, b;
	if (frac >= 0.6) {
		r = 0.298; g = 0.686; b = 0.314;
	} else if (frac >= 0.3) {
		r = 0.129; g = 0.588; b = 0.953;
	} else {
		r = 0.957; g = 0.263; b = 0.212;
	}

	cairo_set_source_rgb(cr, r, g, b);
	double fill_w = bar_w * frac;
	if (fill_w < 12)
		fill_w = 12;
	rounded_rect(cr, bar_x, bar_y, fill_w, bar_h, 6);
	cairo_fill(cr);

	char val[128];
	snprintf(val, sizeof(val), "%.2f Gb/s (%.2f GB/s)",
	         bw->gbps, bw->gbytes_s);
	draw_text(cr, bar_x + 10, bar_y + 6, val, "Sans Bold 11", CLR_TEXT);

	char maxlbl[32];
	snprintf(maxlbl, sizeof(maxlbl), "/ %.0f Gb/s", max_gbps);
	draw_text_right(cr, bar_x, bar_y + 6, maxlbl, "Sans 9",
	                CLR_DIM, bar_w);

	if (bw->transferred[0]) {
		char xfer[128];
		snprintf(xfer, sizeof(xfer), "Transferred: %s", bw->transferred);
		draw_text(cr, 12, bar_y + bar_h + 6, xfer, "Sans 9", CLR_DIM);
	}

	return FALSE;
}

GtkWidget *odl_chart_create_throughput(struct odl_parsed_bandwidth *bw)
{
	GtkWidget *da = gtk_drawing_area_new();
	gtk_widget_set_size_request(da, 500, 85);
	g_signal_connect(da, "draw", G_CALLBACK(on_draw_throughput), bw);
	return da;
}

static gboolean on_draw_stats_card(GtkWidget *widget, cairo_t *cr,
                                   gpointer user_data)
{
	struct odl_parsed_stats *stats = user_data;
	int width = gtk_widget_get_allocated_width(widget);
	int height = gtk_widget_get_allocated_height(widget);

	cairo_set_source_rgb(cr, CLR_CARD);
	rounded_rect(cr, 0, 0, width, height, 8);
	cairo_fill(cr);

	if (!stats || !stats->valid)
		return FALSE;

	draw_text(cr, 12, 8, stats->label, "Sans Bold 10", CLR_TEXT);

	char buf[64];
	double col1_x = 16;
	double col2_x = width / 2.0 + 8;
	double y = 34;
	double row_h = 22;
	const char *font_label = "Sans 9";
	const char *font_val = "Monospace Bold 10";

	draw_text(cr, col1_x, y, "Min", font_label, CLR_DIM);
	odl_parse_format_latency(stats->min_ns, buf, sizeof(buf));
	draw_text(cr, col1_x + 60, y, buf, font_val, CLR_GREEN);

	draw_text(cr, col2_x, y, "Max", font_label, CLR_DIM);
	odl_parse_format_latency(stats->max_ns, buf, sizeof(buf));
	draw_text(cr, col2_x + 60, y, buf, font_val, CLR_RED);

	y += row_h;

	draw_text(cr, col1_x, y, "Avg", font_label, CLR_DIM);
	odl_parse_format_latency(stats->avg_ns, buf, sizeof(buf));
	draw_text(cr, col1_x + 60, y, buf, font_val, CLR_BLUE);

	draw_text(cr, col2_x, y, "Median", font_label, CLR_DIM);
	odl_parse_format_latency(stats->median_ns, buf, sizeof(buf));
	draw_text(cr, col2_x + 60, y, buf, font_val, CLR_BLUE);

	y += row_h;

	draw_text(cr, col1_x, y, "p95", font_label, CLR_DIM);
	odl_parse_format_latency(stats->p95_ns, buf, sizeof(buf));
	draw_text(cr, col1_x + 60, y, buf, font_val, CLR_TEXT);

	draw_text(cr, col2_x, y, "p99", font_label, CLR_DIM);
	odl_parse_format_latency(stats->p99_ns, buf, sizeof(buf));
	draw_text(cr, col2_x + 60, y, buf, font_val, CLR_TEXT);

	y += row_h;

	draw_text(cr, col1_x, y, "p99.9", font_label, CLR_DIM);
	odl_parse_format_latency(stats->p999_ns, buf, sizeof(buf));
	draw_text(cr, col1_x + 60, y, buf, font_val, CLR_AMBER);

	draw_text(cr, col2_x, y, "Jitter", font_label, CLR_DIM);
	odl_parse_format_latency(stats->stddev_ns, buf, sizeof(buf));
	draw_text(cr, col2_x + 60, y, buf, font_val, CLR_AMBER);

	y += row_h;
	char samples[64];
	snprintf(samples, sizeof(samples), "%zu samples", stats->sample_count);
	draw_text(cr, col1_x, y, samples, "Sans 8", CLR_DIM);

	return FALSE;
}

GtkWidget *odl_chart_create_stats_card(struct odl_parsed_stats *stats)
{
	GtkWidget *da = gtk_drawing_area_new();
	gtk_widget_set_size_request(da, 500, 150);
	g_signal_connect(da, "draw", G_CALLBACK(on_draw_stats_card), stats);
	return da;
}

static gboolean on_draw_load_cmp(GtkWidget *widget, cairo_t *cr,
                                 gpointer user_data)
{
	struct odl_parsed_load_comparison *cmp = user_data;
	int width = gtk_widget_get_allocated_width(widget);
	int height = gtk_widget_get_allocated_height(widget);

	cairo_set_source_rgb(cr, CLR_CARD);
	rounded_rect(cr, 0, 0, width, height, 8);
	cairo_fill(cr);

	if (!cmp || !cmp->valid)
		return FALSE;

	draw_text(cr, 12, 8, "Latency Comparison", "Sans Bold 10", CLR_TEXT);

	char buf[64];
	double y = 34;

	draw_text(cr, 16, y, "Idle:", "Sans 9", CLR_DIM);
	snprintf(buf, sizeof(buf), "%.2f us", cmp->idle_us);
	draw_text(cr, 120, y, buf, "Monospace Bold 10", CLR_GREEN);

	y += 22;
	draw_text(cr, 16, y, "Under Load:", "Sans 9", CLR_DIM);
	snprintf(buf, sizeof(buf), "%.2f us", cmp->load_us);
	double dr = cmp->degradation_pct > 50 ? 0.957 : 0.129;
	double dg = cmp->degradation_pct > 50 ? 0.263 : 0.588;
	double db = cmp->degradation_pct > 50 ? 0.212 : 0.953;
	draw_text(cr, 120, y, buf, "Monospace Bold 10", dr, dg, db);

	y += 22;
	draw_text(cr, 16, y, "Degradation:", "Sans 9", CLR_DIM);
	snprintf(buf, sizeof(buf), "%.1f%%", cmp->degradation_pct);
	draw_text(cr, 120, y, buf, "Monospace Bold 10", CLR_AMBER);

	return FALSE;
}

GtkWidget *odl_chart_create_load_comparison(struct odl_parsed_load_comparison *cmp)
{
	GtkWidget *da = gtk_drawing_area_new();
	gtk_widget_set_size_request(da, 500, 105);
	g_signal_connect(da, "draw", G_CALLBACK(on_draw_load_cmp), cmp);
	return da;
}

static gboolean on_draw_jitter_summary(GtkWidget *widget, cairo_t *cr,
                                       gpointer user_data)
{
	struct odl_parsed_jitter_summary *js = user_data;
	int width = gtk_widget_get_allocated_width(widget);
	int height = gtk_widget_get_allocated_height(widget);

	cairo_set_source_rgb(cr, CLR_CARD);
	rounded_rect(cr, 0, 0, width, height, 8);
	cairo_fill(cr);

	if (!js || !js->valid)
		return FALSE;

	draw_text(cr, 12, 8, "Jitter Summary", "Sans Bold 10", CLR_TEXT);

	char buf[64];
	double y = 34;

	draw_text(cr, 16, y, "Avg RTT:", "Sans 9", CLR_DIM);
	odl_parse_format_latency(js->avg_rtt_ns, buf, sizeof(buf));
	draw_text(cr, 130, y, buf, "Monospace Bold 10", CLR_BLUE);

	y += 22;
	draw_text(cr, 16, y, "Avg Jitter:", "Sans 9", CLR_DIM);
	odl_parse_format_latency(js->avg_jitter_ns, buf, sizeof(buf));
	draw_text(cr, 130, y, buf, "Monospace Bold 10", CLR_AMBER);

	y += 22;
	draw_text(cr, 16, y, "Max Jitter:", "Sans 9", CLR_DIM);
	odl_parse_format_latency(js->max_jitter_ns, buf, sizeof(buf));
	draw_text(cr, 130, y, buf, "Monospace Bold 10", CLR_RED);

	y += 22;
	draw_text(cr, 16, y, "Jitter/RTT:", "Sans 9", CLR_DIM);
	snprintf(buf, sizeof(buf), "%.2f%%", js->jitter_rtt_ratio_pct);
	draw_text(cr, 130, y, buf, "Monospace Bold 10", CLR_TEXT);

	return FALSE;
}

GtkWidget *odl_chart_create_jitter_summary(struct odl_parsed_jitter_summary *js)
{
	GtkWidget *da = gtk_drawing_area_new();
	gtk_widget_set_size_request(da, 500, 125);
	g_signal_connect(da, "draw", G_CALLBACK(on_draw_jitter_summary), js);
	return da;
}
