/*
 * OdinLink Thunderbolt 5 - Statistics Engine
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Histogram bucket boundaries in nanoseconds */
static const uint64_t hist_bounds[ODL_HIST_BUCKETS] = {
	100,         /* [0]  < 100ns */
	200,         /* [1]  100-200ns */
	500,         /* [2]  200-500ns */
	1000,        /* [3]  500ns-1us */
	2000,        /* [4]  1-2us */
	5000,        /* [5]  2-5us */
	10000,       /* [6]  5-10us */
	20000,       /* [7]  10-20us */
	50000,       /* [8]  20-50us */
	100000,      /* [9]  50-100us */
	500000,      /* [10] 100-500us */
	1000000,     /* [11] 500us-1ms */
	UINT64_MAX,  /* [12] > 1ms */
};

static const char *hist_labels[ODL_HIST_BUCKETS] = {
	"    < 100ns",
	"  100-200ns",
	"  200-500ns",
	" 500ns-1us ",
	"    1-2us  ",
	"    2-5us  ",
	"   5-10us  ",
	"  10-20us  ",
	"  20-50us  ",
	" 50-100us  ",
	"100-500us  ",
	"500us-1ms  ",
	"    > 1ms  ",
};

/* Initialize a statistics collector with the given sample capacity. */
int odl_stats_init(struct odl_stats *stats, size_t capacity)
{
	memset(stats, 0, sizeof(*stats));

	if (capacity > ODL_STATS_MAX_SAMPLES)
		capacity = ODL_STATS_MAX_SAMPLES;

	stats->samples = malloc(capacity * sizeof(uint64_t));
	if (!stats->samples)
		return -1;

	stats->capacity = capacity;
	stats->min_ns = UINT64_MAX;
	return 0;
}

/* Free all resources held by a statistics collector. */
void odl_stats_free(struct odl_stats *stats)
{
	free(stats->samples);
	memset(stats, 0, sizeof(*stats));
}

/* Add a single sample to the statistics collector. */
void odl_stats_add(struct odl_stats *stats, uint64_t sample_ns)
{
	if (stats->count < stats->capacity)
		stats->samples[stats->count] = sample_ns;

	stats->count++;
	stats->sum_ns += sample_ns;

	if (sample_ns < stats->min_ns)
		stats->min_ns = sample_ns;
	if (sample_ns > stats->max_ns)
		stats->max_ns = sample_ns;

	for (int i = 0; i < ODL_HIST_BUCKETS; i++) {
		if (sample_ns < hist_bounds[i]) {
			stats->hist[i]++;
			break;
		}
	}
}

/* Compare two uint64_t values for qsort. */
static int cmp_u64(const void *a, const void *b)
{
	uint64_t va = *(const uint64_t *)a;
	uint64_t vb = *(const uint64_t *)b;
	return (va > vb) - (va < vb);
}

/* Compute percentiles, average, and standard deviation from collected samples. */
void odl_stats_finalize(struct odl_stats *stats)
{
	size_t n;

	if (stats->count == 0)
		return;

	n = (stats->count < stats->capacity) ? stats->count : stats->capacity;
	stats->avg_ns = (double)stats->sum_ns / stats->count;

	qsort(stats->samples, n, sizeof(uint64_t), cmp_u64);

	stats->median_ns = stats->samples[n / 2];
	stats->p50_ns = stats->samples[(size_t)(n * 0.50)];
	stats->p95_ns = stats->samples[(size_t)(n * 0.95)];
	stats->p99_ns = stats->samples[(size_t)(n * 0.99)];
	stats->p999_ns = stats->samples[n > 1000 ? (size_t)(n * 0.999) : n - 1];

	double sum_sq = 0;
	for (size_t i = 0; i < n; i++) {
		double diff = (double)stats->samples[i] - stats->avg_ns;
		sum_sq += diff * diff;
	}
	stats->stddev_ns = sqrt(sum_sq / n);
}

/* Print a summary of collected statistics. */
void odl_stats_print(const struct odl_stats *stats, const char *label)
{
	char buf[64];

	printf("\n=== %s Results (%zu samples) ===\n", label, stats->count);
	printf("  Min:     %s\n", odl_format_latency(stats->min_ns, buf, sizeof(buf)));
	printf("  Max:     %s\n", odl_format_latency(stats->max_ns, buf, sizeof(buf)));
	printf("  Avg:     %s\n", odl_format_latency((uint64_t)stats->avg_ns, buf, sizeof(buf)));
	printf("  Median:  %s\n", odl_format_latency(stats->median_ns, buf, sizeof(buf)));
	printf("  p50:     %s\n", odl_format_latency(stats->p50_ns, buf, sizeof(buf)));
	printf("  p95:     %s\n", odl_format_latency(stats->p95_ns, buf, sizeof(buf)));
	printf("  p99:     %s\n", odl_format_latency(stats->p99_ns, buf, sizeof(buf)));
	printf("  p99.9:   %s\n", odl_format_latency(stats->p999_ns, buf, sizeof(buf)));
	printf("  Jitter (stddev): %s\n",
	       odl_format_latency((uint64_t)stats->stddev_ns, buf, sizeof(buf)));
}

/* Print a histogram of the latency distribution. */
void odl_stats_print_histogram(const struct odl_stats *stats)
{
	uint64_t max_count = 0;

	for (int i = 0; i < ODL_HIST_BUCKETS; i++) {
		if (stats->hist[i] > max_count)
			max_count = stats->hist[i];
	}

	if (max_count == 0)
		return;

	printf("\n  Latency Histogram:\n");
	for (int i = 0; i < ODL_HIST_BUCKETS; i++) {
		if (stats->hist[i] == 0)
			continue;

		int bar_len = (int)(40.0 * stats->hist[i] / max_count);
		double pct = 100.0 * stats->hist[i] / stats->count;

		printf("  %s |", hist_labels[i]);
		for (int j = 0; j < bar_len; j++)
			putchar('#');
		printf(" %lu (%.1f%%)\n",
		       (unsigned long)stats->hist[i], pct);
	}
}

/* Write all samples and summary statistics to a CSV file. */
void odl_stats_write_csv(const struct odl_stats *stats, const char *path)
{
	FILE *fp = fopen(path, "w");
	if (!fp) {
		fprintf(stderr, "Failed to open %s for writing\n", path);
		return;
	}

	fprintf(fp, "sample_index,latency_ns\n");
	size_t n = (stats->count < stats->capacity) ? stats->count : stats->capacity;
	for (size_t i = 0; i < n; i++)
		fprintf(fp, "%zu,%lu\n", i, (unsigned long)stats->samples[i]);

	fprintf(fp, "\n# Summary\n");
	fprintf(fp, "# min_ns,%lu\n", (unsigned long)stats->min_ns);
	fprintf(fp, "# max_ns,%lu\n", (unsigned long)stats->max_ns);
	fprintf(fp, "# avg_ns,%.2f\n", stats->avg_ns);
	fprintf(fp, "# median_ns,%lu\n", (unsigned long)stats->median_ns);
	fprintf(fp, "# p50_ns,%lu\n", (unsigned long)stats->p50_ns);
	fprintf(fp, "# p95_ns,%lu\n", (unsigned long)stats->p95_ns);
	fprintf(fp, "# p99_ns,%lu\n", (unsigned long)stats->p99_ns);
	fprintf(fp, "# p999_ns,%lu\n", (unsigned long)stats->p999_ns);
	fprintf(fp, "# stddev_ns,%.2f\n", stats->stddev_ns);

	fclose(fp);
	printf("Results written to %s\n", path);
}
