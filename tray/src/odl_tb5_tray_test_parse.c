/*
 * OdinLink TB5 Tray - Test Output Parser
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_tray_test_parse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Parse a formatted latency value like "1.23 us" into nanoseconds */
double odl_parse_latency_ns(const char *str)
{
	double val;
	char unit[8];

	if (!str)
		return 0.0;

	while (*str && isspace((unsigned char)*str))
		str++;

	if (sscanf(str, "%lf %7s", &val, unit) != 2)
		return 0.0;

	if (strcmp(unit, "ns") == 0)       return val;
	if (strcmp(unit, "us") == 0)       return val * 1000.0;
	if (strcmp(unit, "ms") == 0)       return val * 1000000.0;
	if (strcmp(unit, "s") == 0)        return val * 1000000000.0;
	return 0.0;
}

/* Format nanoseconds into a human-readable string */
const char *odl_parse_format_latency(double ns, char *buf, size_t bufsz)
{
	if (ns < 1000.0)
		snprintf(buf, bufsz, "%.0f ns", ns);
	else if (ns < 1000000.0)
		snprintf(buf, bufsz, "%.2f us", ns / 1000.0);
	else if (ns < 1000000000.0)
		snprintf(buf, bufsz, "%.2f ms", ns / 1000000.0);
	else
		snprintf(buf, bufsz, "%.2f s", ns / 1000000000.0);
	return buf;
}

/* Extract and unescape the "output" field from daemon result JSON */
char *odl_parse_extract_json_output(const char *json)
{
	if (!json)
		return NULL;

	const char *key = strstr(json, "\"output\":");
	if (!key)
		return NULL;

	key += strlen("\"output\":");
	while (*key && isspace((unsigned char)*key))
		key++;

	if (strncmp(key, "null", 4) == 0)
		return NULL;

	if (*key != '"')
		return NULL;
	key++;

	GString *out = g_string_new(NULL);
	for (const char *p = key; *p && *p != '"'; p++) {
		if (*p == '\\' && *(p + 1)) {
			p++;
			switch (*p) {
			case 'n':  g_string_append_c(out, '\n'); break;
			case 'r':  g_string_append_c(out, '\r'); break;
			case 't':  g_string_append_c(out, '\t'); break;
			case '"':  g_string_append_c(out, '"');  break;
			case '\\': g_string_append_c(out, '\\'); break;
			default:   g_string_append_c(out, *p);   break;
			}
		} else {
			g_string_append_c(out, *p);
		}
	}

	return g_string_free(out, FALSE);
}

static bool parse_stat_line(const char *line, const char *prefix, double *out)
{
	const char *p = strstr(line, prefix);
	if (!p)
		return false;
	p += strlen(prefix);
	while (*p && isspace((unsigned char)*p))
		p++;
	*out = odl_parse_latency_ns(p);
	return true;
}

/* Parse a stats block starting with "=== Label (N samples) ===" */
static int parse_stats_block(const char *text, int offset,
                             struct odl_parsed_stats *stats)
{
	const char *p = text + offset;
	const char *header = strstr(p, "=== ");
	if (!header)
		return -1;

	char label_buf[64] = {0};
	size_t count = 0;
	if (sscanf(header, "=== %63[^(] (%zu samples)", label_buf, &count) < 1)
		return -1;

	char *end = label_buf + strlen(label_buf) - 1;
	while (end > label_buf && isspace((unsigned char)*end))
		*end-- = '\0';

	g_strlcpy(stats->label, label_buf, sizeof(stats->label));
	stats->sample_count = count;

	const char *block_end = strstr(header + 4, "\n===");
	if (!block_end)
		block_end = text + strlen(text);

	const char *line = header;
	while (line < block_end) {
		const char *nl = strchr(line, '\n');
		if (!nl)
			nl = text + strlen(text);

		char linebuf[256];
		size_t len = (size_t)(nl - line);
		if (len >= sizeof(linebuf))
			len = sizeof(linebuf) - 1;
		memcpy(linebuf, line, len);
		linebuf[len] = '\0';

		parse_stat_line(linebuf, "Min:", &stats->min_ns);
		parse_stat_line(linebuf, "Max:", &stats->max_ns);
		parse_stat_line(linebuf, "Avg:", &stats->avg_ns);
		parse_stat_line(linebuf, "Median:", &stats->median_ns);
		parse_stat_line(linebuf, "p50:", &stats->p50_ns);
		parse_stat_line(linebuf, "p95:", &stats->p95_ns);
		parse_stat_line(linebuf, "p99:", &stats->p99_ns);
		parse_stat_line(linebuf, "p99.9:", &stats->p999_ns);
		parse_stat_line(linebuf, "Jitter (stddev):", &stats->stddev_ns);

		line = nl + 1;
	}

	stats->valid = (stats->sample_count > 0);
	return (int)(block_end - text);
}

/* Parse a histogram block starting with "Histogram:" */
static int parse_histogram(const char *text, int offset,
                           struct odl_parsed_histogram *hist)
{
	const char *p = text + offset;
	const char *header = strstr(p, "Histogram:");
	if (!header)
		return -1;

	const char *line = strchr(header, '\n');
	if (!line)
		return -1;
	line++;

	hist->num_buckets = 0;
	memset(hist->count, 0, sizeof(hist->count));
	memset(hist->pct, 0, sizeof(hist->pct));

	while (*line && hist->num_buckets < ODL_PARSE_HIST_BUCKETS) {
		const char *bar = strchr(line, '|');
		if (!bar)
			break;

		size_t label_len = (size_t)(bar - line);
		if (label_len >= 32)
			label_len = 31;
		memcpy(hist->label[hist->num_buckets], line, label_len);
		hist->label[hist->num_buckets][label_len] = '\0';
		char *lp = hist->label[hist->num_buckets];
		while (*lp && isspace((unsigned char)*lp))
			lp++;
		memmove(hist->label[hist->num_buckets], lp,
		        strlen(lp) + 1);
		char *te = hist->label[hist->num_buckets] +
		           strlen(hist->label[hist->num_buckets]) - 1;
		while (te > hist->label[hist->num_buckets] &&
		       isspace((unsigned char)*te))
			*te-- = '\0';

		const char *cp = bar + 1;
		while (*cp == '#')
			cp++;
		while (*cp && isspace((unsigned char)*cp))
			cp++;

		unsigned long cnt = 0;
		double pct = 0.0;
		if (sscanf(cp, "%lu (%lf%%)", &cnt, &pct) >= 1) {
			hist->count[hist->num_buckets] = cnt;
			hist->pct[hist->num_buckets] = pct;
			hist->num_buckets++;
		}

		const char *nl = strchr(line, '\n');
		if (!nl)
			break;
		line = nl + 1;

		if (!strchr(line, '|') ||
		    (line[0] != ' ' && line[0] != '\t'))
			break;
	}

	hist->valid = (hist->num_buckets > 0);
	return (int)((line - text));
}

/* Parse bandwidth/throughput from output text */
static void parse_bandwidth(const char *text,
                            struct odl_parsed_bandwidth *bw)
{
	const char *tp = strstr(text, "Throughput:");
	if (!tp)
		return;

	tp += strlen("Throughput:");
	while (*tp && isspace((unsigned char)*tp))
		tp++;

	if (sscanf(tp, "%lf Gb/s (%lf GB/s)", &bw->gbps, &bw->gbytes_s) >= 1)
		bw->valid = true;

	const char *tr = strstr(text, "Transferred:");
	if (tr) {
		tr += strlen("Transferred:");
		while (*tr && isspace((unsigned char)*tr))
			tr++;
		const char *nl = strchr(tr, '\n');
		if (nl) {
			size_t len = (size_t)(nl - tr);
			if (len >= sizeof(bw->transferred))
				len = sizeof(bw->transferred) - 1;
			memcpy(bw->transferred, tr, len);
			bw->transferred[len] = '\0';
		}
	}
}

/* Parse latency-under-load comparison section */
static void parse_load_comparison(const char *text,
                                  struct odl_parsed_load_comparison *cmp)
{
	const char *section = strstr(text, "=== Latency Comparison ===");
	if (!section)
		return;

	const char *p;

	p = strstr(section, "Idle latency:");
	if (p) {
		p += strlen("Idle latency:");
		cmp->idle_us = odl_parse_latency_ns(p) / 1000.0;
	}

	p = strstr(section, "Under load:");
	if (p) {
		p += strlen("Under load:");
		cmp->load_us = odl_parse_latency_ns(p) / 1000.0;
	}

	p = strstr(section, "Degradation:");
	if (p) {
		p += strlen("Degradation:");
		while (*p && isspace((unsigned char)*p))
			p++;
		sscanf(p, "%lf%%", &cmp->degradation_pct);
	}

	cmp->valid = (cmp->idle_us > 0 || cmp->load_us > 0);
}

/* Parse jitter summary section */
static void parse_jitter_summary(const char *text,
                                 struct odl_parsed_jitter_summary *js)
{
	const char *section = strstr(text, "Jitter Summary:");
	if (!section)
		return;

	const char *p;

	p = strstr(section, "Average RTT:");
	if (p) {
		p += strlen("Average RTT:");
		js->avg_rtt_ns = odl_parse_latency_ns(p);
	}

	p = strstr(section, "Average Jitter:");
	if (p) {
		p += strlen("Average Jitter:");
		js->avg_jitter_ns = odl_parse_latency_ns(p);
	}

	p = strstr(section, "Max Jitter:");
	if (p) {
		p += strlen("Max Jitter:");
		js->max_jitter_ns = odl_parse_latency_ns(p);
	}

	p = strstr(section, "Jitter/RTT ratio:");
	if (p) {
		p += strlen("Jitter/RTT ratio:");
		while (*p && isspace((unsigned char)*p))
			p++;
		sscanf(p, "%lf%%", &js->jitter_rtt_ratio_pct);
	}

	js->valid = (js->avg_rtt_ns > 0);
}

static void parse_compress(const char *text, struct odl_parsed_compress *c)
{
	const char *sec = strstr(text, "=== Compression (measured) ===");
	const char *p;
	const char *line;

	if (!sec)
		return;
	p = strstr(sec, "Backend:");
	if (p) {
		p += strlen("Backend:");
		while (*p && isspace((unsigned char)*p))
			p++;
		const char *nl = strchr(p, '\n');
		size_t n = nl ? (size_t)(nl - p) : strlen(p);
		if (n >= sizeof(c->backend))
			n = sizeof(c->backend) - 1;
		memcpy(c->backend, p, n);
		c->backend[n] = '\0';
	}
	line = sec;
	while (c->nrows < ODL_PARSE_MAX_COMPRESS) {
		const char *row = strstr(line, "Payload ");
		char name[32];
		double in_b = 0, wire_b = 0, ratio = 0;

		if (!row)
			break;
		if (sscanf(row, "Payload %31s in=%lf wire=%lf ratio=%lf",
			   name, &in_b, &wire_b, &ratio) >= 3) {
			g_strlcpy(c->rows[c->nrows].name, name,
				  sizeof(c->rows[c->nrows].name));
			c->rows[c->nrows].in_bytes = in_b;
			c->rows[c->nrows].wire_bytes = wire_b;
			c->rows[c->nrows].ratio = ratio;
			c->nrows++;
			c->valid = true;
		}
		line = row + 8;
	}
}

/* Top-level parser: dispatch to sub-parsers based on output content */
int odl_parse_test_output(const char *output_text,
                          const char *test_type,
                          struct odl_parsed_test_result *result)
{
	if (!output_text || !test_type || !result)
		return -1;

	memset(result, 0, sizeof(*result));
	g_strlcpy(result->test_type, test_type, sizeof(result->test_type));

	parse_bandwidth(output_text, &result->bandwidth);
	parse_compress(output_text, &result->compress);

	const char *search = output_text;
	while (result->num_stats < ODL_PARSE_MAX_STATS) {
		const char *found = strstr(search, "=== ");
		if (!found)
			break;
		if (strstr(found, "=== Latency Comparison") == found ||
		    strstr(found, "=== Compression") == found) {
			search = found + 4;
			continue;
		}

		int end_offset = parse_stats_block(output_text,
		                                   (int)(found - output_text),
		                                   &result->stats[result->num_stats]);
		if (end_offset < 0) {
			search = found + 4;
			continue;
		}
		if (result->stats[result->num_stats].valid)
			result->num_stats++;
		search = output_text + end_offset;
	}

	search = output_text;
	while (result->num_histograms < ODL_PARSE_MAX_STATS) {
		const char *found = strstr(search, "Histogram:");
		if (!found)
			break;

		int end_offset = parse_histogram(output_text,
		                                 (int)(found - output_text - 20 > 0 ?
		                                       found - output_text - 20 : 0),
		                                 &result->histograms[result->num_histograms]);
		if (end_offset < 0) {
			search = found + 10;
			continue;
		}
		if (result->histograms[result->num_histograms].valid)
			result->num_histograms++;
		search = output_text + end_offset;
	}

	parse_load_comparison(output_text, &result->load_cmp);

	parse_jitter_summary(output_text, &result->jitter_summary);

	return 0;
}
