/*
 * OdinLink TB5 Tray - Test Result Charts
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_TRAY_TEST_CHARTS_H
#define ODL_TB5_TRAY_TEST_CHARTS_H

#include <gtk/gtk.h>
#include "odl_tb5_tray_test_parse.h"

/* Create a drawing area for a histogram chart (horizontal bars) */
GtkWidget *odl_chart_create_histogram(struct odl_parsed_histogram *hist,
                                      const char *title);

/* Create a drawing area for a throughput gauge bar */
GtkWidget *odl_chart_create_throughput(struct odl_parsed_bandwidth *bw);

/* Create a drawing area for a stats summary card */
GtkWidget *odl_chart_create_stats_card(struct odl_parsed_stats *stats);

/* Create a comparison card for latency-under-load results */
GtkWidget *odl_chart_create_load_comparison(struct odl_parsed_load_comparison *cmp);

/* Create a jitter summary card */
GtkWidget *odl_chart_create_jitter_summary(struct odl_parsed_jitter_summary *js);

#endif
