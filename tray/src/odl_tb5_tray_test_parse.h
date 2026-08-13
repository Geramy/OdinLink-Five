/*
 * OdinLink TB5 Tray - Test Output Parser
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_TRAY_TEST_PARSE_H
#define ODL_TB5_TRAY_TEST_PARSE_H

#include <glib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define ODL_PARSE_HIST_BUCKETS 13
#define ODL_PARSE_MAX_STATS    4

struct odl_parsed_stats {
	char   label[64];
	size_t sample_count;
	double min_ns;
	double max_ns;
	double avg_ns;
	double median_ns;
	double p50_ns;
	double p95_ns;
	double p99_ns;
	double p999_ns;
	double stddev_ns;
	bool   valid;
};

struct odl_parsed_histogram {
	char     label[ODL_PARSE_HIST_BUCKETS][32];
	uint64_t count[ODL_PARSE_HIST_BUCKETS];
	double   pct[ODL_PARSE_HIST_BUCKETS];
	int      num_buckets;
	bool     valid;
};

struct odl_parsed_bandwidth {
	double gbps;
	double gbytes_s;
	char   transferred[64];
	char   elapsed[64];
	bool   valid;
};

struct odl_parsed_load_comparison {
	double idle_us;
	double load_us;
	double degradation_pct;
	bool   valid;
};

struct odl_parsed_jitter_summary {
	double avg_rtt_ns;
	double avg_jitter_ns;
	double max_jitter_ns;
	double jitter_rtt_ratio_pct;
	bool   valid;
};

#define ODL_PARSE_MAX_COMPRESS 4

struct odl_parsed_compress_row {
	char   name[32];
	double in_bytes;
	double wire_bytes;
	double ratio; /* 0 = incompressible */
};

struct odl_parsed_compress {
	char backend[64];
	struct odl_parsed_compress_row rows[ODL_PARSE_MAX_COMPRESS];
	int  nrows;
	bool valid;
};

struct odl_parsed_test_result {
	char test_type[32];

	struct odl_parsed_bandwidth    bandwidth;
	struct odl_parsed_stats        stats[ODL_PARSE_MAX_STATS];
	int                            num_stats;
	struct odl_parsed_histogram    histograms[ODL_PARSE_MAX_STATS];
	int                            num_histograms;
	struct odl_parsed_load_comparison load_cmp;
	struct odl_parsed_jitter_summary  jitter_summary;
	struct odl_parsed_compress     compress;
};

/* Parse output text from daemon result JSON into structured metrics */
int odl_parse_test_output(const char *output_text,
                          const char *test_type,
                          struct odl_parsed_test_result *result);

/* Parse a formatted latency value like "1.23 us" into nanoseconds */
double odl_parse_latency_ns(const char *str);

/* Extract and unescape the "output" field from the daemon result JSON */
char *odl_parse_extract_json_output(const char *json);

/* Format nanoseconds into a human-readable string (writes into buf) */
const char *odl_parse_format_latency(double ns, char *buf, size_t bufsz);

#endif
