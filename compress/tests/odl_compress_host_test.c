/* SPDX-License-Identifier: MIT */
/* Portable ODLC + LZ4 host round-trip. No CUDA. */
#include <odl_tb5/odl_compress.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *msg)
{
	fprintf(stderr, "FAIL: %s\n", msg);
	return 1;
}

static int roundtrip(const uint8_t *src, size_t n, const char *tag)
{
	size_t cap = odl_compress_host_max_wire_bytes(n);
	uint8_t *wire = malloc(cap);
	uint8_t *out = malloc(n);
	size_t wire_n = 0, out_n = 0;
	struct odl_compress_header hdr;
	int rc;

	if (!wire || !out)
		return fail("oom");

	rc = odl_compress_host(src, n, wire, cap, &wire_n);
	if (rc != 0) {
		free(wire);
		free(out);
		fprintf(stderr, "FAIL: %s compress rc=%d\n", tag, rc);
		return 1;
	}
	if (wire_n >= n)
		return fail("wire not smaller");
	if (!odl_compress_looks_compressed(wire, wire_n))
		return fail("magic missing");
	memcpy(&hdr, wire, sizeof(hdr));
	if (hdr.algo != ODL_ALGO_LZ4_BLOCK)
		return fail("algo is not lz4_block");
	if (hdr.original_bytes != n)
		return fail("original_bytes");

	if (odl_decompress_host(wire, wire_n, out, n, &out_n) != 0)
		return fail("decompress");
	if (out_n != n)
		return fail("out size");
	if (memcmp(src, out, n) != 0)
		return fail("mismatch");

	printf("ok %s in=%zu wire=%zu ratio=%.3f chunks=%u\n",
	       tag, n, wire_n, (double)n / (double)wire_n, hdr.num_chunks);
	free(wire);
	free(out);
	return 0;
}

int main(void)
{
	size_t n = 3 * 65536 + 100;
	uint8_t *rep = malloc(n);
	uint8_t *mix = malloc(n);
	size_t i;
	int rc = 0;

	if (!rep || !mix)
		return fail("oom");

	for (i = 0; i < n; i++)
		rep[i] = (uint8_t)(i & 0x3f);
	for (i = 0; i < n; i++)
		mix[i] = (uint8_t)((i * 1103515245u + 12345u) >> 16);

	/* Header + table is 40 bytes; need enough input to win. */
	if (roundtrip(rep, 4096, "4k") != 0)
		rc = 1;
	if (roundtrip(rep, 65536, "one-chunk") != 0)
		rc = 1;
	if (roundtrip(rep, n, "multi-chunk") != 0)
		rc = 1;

	/* Random data must either fail (not worth it) or round-trip. */
	{
		size_t cap = odl_compress_host_max_wire_bytes(n);
		uint8_t *wire = malloc(cap);
		size_t wire_n = 0;
		int c = odl_compress_host(mix, n, wire, cap, &wire_n);
		if (c == 0) {
			uint8_t *out = malloc(n);
			size_t out_n = 0;
			if (odl_decompress_host(wire, wire_n, out, n, &out_n) != 0 ||
			    out_n != n || memcmp(mix, out, n) != 0)
				rc = fail("random round-trip");
			else
				printf("ok random in=%zu wire=%zu\n", n, wire_n);
			free(out);
		} else {
			printf("ok random incompressible (fallback raw)\n");
		}
		free(wire);
	}

	/* Reject nvCOMP-looking header (algo=1). */
	{
		struct odl_compress_header fake = {
			.magic = ODL_COMPRESS_MAGIC,
			.version = ODL_COMPRESS_VERSION,
			.algo = ODL_ALGO_GDEFLATE,
			.original_bytes = 8,
			.compressed_bytes = 8,
			.num_chunks = 1,
		};
		uint8_t buf[sizeof(fake) + 8];
		uint8_t out[8];
		size_t out_n = 0;
		memcpy(buf, &fake, sizeof(fake));
		if (odl_decompress_host(buf, sizeof(buf), out, 8, &out_n) == 0)
			rc = fail("accepted gdeflate");
		else
			printf("ok rejected nvCOMP gdeflate\n");
	}

	free(rep);
	free(mix);
	return rc;
}
