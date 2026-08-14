/* SPDX-License-Identifier: MIT */
/* Env parsing + helpers — no nvCOMP dependency. Always built. */

#include <odl_tb5/odl_compress.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Filled by odl_compress_backend_available() in stub or nvcomp TU. */
extern int odl_compress_backend_available(void);

static int g_inited;
static int g_want; /* ODL_COMPRESS=1 */
static int g_algo = ODL_ALGO_GDEFLATE;
static size_t g_threshold = 262144;

static int env_truthy(const char *v)
{
	if (!v || !*v)
		return 0;
	if (v[0] == '0' && v[1] == '\0')
		return 0;
	if (strcasecmp(v, "false") == 0 || strcasecmp(v, "off") == 0 ||
	    strcasecmp(v, "no") == 0)
		return 0;
	return 1;
}

int odl_compress_init(void)
{
	if (g_inited)
		return odl_compress_enabled();
	g_inited = 1;

	const char *en = getenv("ODL_COMPRESS");
	g_want = env_truthy(en);

	const char *th = getenv("ODL_COMPRESS_THRESHOLD");
	if (th && *th) {
		char *end = NULL;
		unsigned long long v = strtoull(th, &end, 10);
		if (end != th && v > 0)
			g_threshold = (size_t)v;
	}

	const char *algo = getenv("ODL_COMPRESS_ALGO");
	if (algo && *algo) {
		if (strcasecmp(algo, "gdeflate") == 0)
			g_algo = ODL_ALGO_GDEFLATE;
		else if (strcasecmp(algo, "lz4") == 0)
			g_algo = ODL_ALGO_LZ4;
		else if (strcasecmp(algo, "snappy") == 0)
			g_algo = ODL_ALGO_SNAPPY;
		else if (strcasecmp(algo, "none") == 0)
			g_algo = ODL_ALGO_NONE;
	}

	return odl_compress_enabled();
}

int odl_compress_enabled(void)
{
	if (!g_inited)
		odl_compress_init();
	return g_want && g_algo != ODL_ALGO_NONE && odl_compress_backend_available();
}

int odl_compress_algo(void)
{
	if (!g_inited)
		odl_compress_init();
	return g_algo;
}

size_t odl_compress_threshold(void)
{
	if (!g_inited)
		odl_compress_init();
	return g_threshold;
}

int odl_compress_header_ok(const struct odl_compress_header *hdr)
{
	if (!hdr)
		return 0;
	return hdr->magic == ODL_COMPRESS_MAGIC &&
	       hdr->version == ODL_COMPRESS_VERSION &&
	       hdr->original_bytes > 0 &&
	       hdr->compressed_bytes > 0;
}

/* odl_compress_max_wire_bytes / device_alloc provided by stub or nvcomp backend TU. */

int odl_compress_should(size_t size, int type_is_cuda)
{
	if (!type_is_cuda)
		return 0;
	if (!odl_compress_enabled())
		return 0;
	if (size < odl_compress_threshold())
		return 0;
	return 1;
}