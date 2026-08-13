/*
 * Local compression report. No cable. Runs on Linux and Mac.
 *
 * Prints measured in/wire/ratio for labeled patterns. Does not invent
 * a multiplier. nvCOMP is not used.
 */
#include "odl_tb5_cli.h"

#include <odl_tb5/odl_compress.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ODL_COMPRESS_BENCH_BYTES (16u << 20) /* 16 MiB — enough for a stable ratio */

static void fill_zeros(uint8_t *p, size_t n)
{
	memset(p, 0, n);
}

static void fill_aa(uint8_t *p, size_t n)
{
	/* Same byte as the bandwidth test. */
	memset(p, 0xAA, n);
}

static void fill_random(uint8_t *p, size_t n)
{
	uint32_t s = 0xC0FFEEu;
	size_t i;

	for (i = 0; i < n; i++) {
		s ^= s << 13;
		s ^= s >> 17;
		s ^= s << 5;
		p[i] = (uint8_t)s;
	}
}

static void one_pattern(const char *name, void (*fill)(uint8_t *, size_t),
			size_t n)
{
	uint8_t *in, *wire;
	size_t cap, wire_n = 0;
	int rc;
	double ratio;

	in = malloc(n);
	cap = odl_compress_host_max_wire_bytes(n);
	wire = malloc(cap);
	if (!in || !wire) {
		free(in);
		free(wire);
		printf("Payload %-12s  (oom)\n", name);
		return;
	}
	fill(in, n);
	rc = odl_compress_host(in, n, wire, cap, &wire_n);
	if (rc != 0 || wire_n == 0) {
		printf("Payload %-12s  in=%zu wire=0 ratio=0.00  (incompressible)\n",
		       name, n);
		free(in);
		free(wire);
		return;
	}
	ratio = (double)n / (double)wire_n;
	printf("Payload %-12s  in=%zu wire=%zu ratio=%.2f\n",
	       name, n, wire_n, ratio);
	/* 1 GiB of this payload → how much wire; 1 GiB of wire → how much payload */
	printf("  1 GiB payload → %.3f GiB wire    1 GiB wire → %.2f GiB payload\n",
	       1.0 / ratio, ratio);
	free(in);
	free(wire);
}

void odl_cli_compress_report(void)
{
	const char *be = "lz4_block host";

	printf("\n=== Compression (measured) ===\n");
	printf("Backend: %s\n", be);
	printf("Size: %u bytes per pattern\n", ODL_COMPRESS_BENCH_BYTES);
	one_pattern("zeros", fill_zeros, ODL_COMPRESS_BENCH_BYTES);
	one_pattern("fill-0xAA", fill_aa, ODL_COMPRESS_BENCH_BYTES);
	one_pattern("random", fill_random, ODL_COMPRESS_BENCH_BYTES);
	printf("Note: ratio is this pattern on this machine. nvCOMP is not used.\n");
}
