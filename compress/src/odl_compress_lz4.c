/* SPDX-License-Identifier: MIT */
/*
 * Portable ODLC + LZ4 raw-block codec. No CUDA. Builds on Linux and Mac.
 * This is the format the Mac (bridge, Compression.framework, this file)
 * can actually decode. nvCOMP GDeflate/batched-LZ4 stays Linux-GPU-only.
 */
#include <odl_tb5/odl_compress.h>

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "lz4.h"

#ifdef __APPLE__
#include <compression.h>
#endif

extern int odl_compress_init(void);

int odl_compress_host_available(void)
{
	return 1;
}

static int host_wanted(void)
{
	const char *en = getenv("ODL_COMPRESS");

	/* On unless explicitly off. NCCL still requires ODL_COMPRESS=1
	 * plus nvCOMP — see odl_compress_enabled(). */
	if (!en || !*en)
		return 1;
	if (en[0] == '0' && en[1] == '\0')
		return 0;
	if (strcasecmp(en, "false") == 0 || strcasecmp(en, "off") == 0 ||
	    strcasecmp(en, "no") == 0)
		return 0;
	return 1;
}

int odl_compress_host_enabled(void)
{
	return host_wanted();
}

int odl_compress_should_host(size_t size)
{
	if (!host_wanted())
		return 0;
	if (size < odl_compress_threshold())
		return 0;
	return 1;
}

size_t odl_compress_host_max_wire_bytes(size_t original_bytes)
{
	size_t n, i, raw, bound;

	if (original_bytes == 0)
		return sizeof(struct odl_compress_header);
	n = (original_bytes + ODL_COMPRESS_CHUNK - 1) / ODL_COMPRESS_CHUNK;
	bound = sizeof(struct odl_compress_header) +
		n * sizeof(struct odl_lz4_chunk);
	for (i = 0; i < n; i++) {
		raw = original_bytes - i * ODL_COMPRESS_CHUNK;
		if (raw > ODL_COMPRESS_CHUNK)
			raw = ODL_COMPRESS_CHUNK;
		bound += (size_t)LZ4_compressBound((int)raw);
	}
	return bound;
}

int odl_compress_looks_compressed(const void *buf, size_t n)
{
	struct odl_compress_header hdr;

	if (!buf || n < sizeof(hdr))
		return 0;
	memcpy(&hdr, buf, sizeof(hdr));
	return odl_compress_header_ok(&hdr);
}

#ifdef __APPLE__
static int apple_lz4_compress(const void *in, int in_n, void *out, int out_cap)
{
	size_t n;

	if (in_n <= 0 || out_cap <= 0)
		return 0;
	n = compression_encode_buffer(out, (size_t)out_cap, in, (size_t)in_n,
				      NULL, COMPRESSION_LZ4_RAW);
	if (n == 0)
		return 0;
	return (int)n;
}

static int apple_lz4_decompress(const void *in, int in_n, void *out, int out_cap)
{
	size_t n;

	if (in_n <= 0 || out_cap <= 0)
		return 0;
	n = compression_decode_buffer(out, (size_t)out_cap, in, (size_t)in_n,
				      NULL, COMPRESSION_LZ4_RAW);
	if (n == 0)
		return 0;
	return (int)n;
}
#endif

static int block_compress(const void *in, int in_n, void *out, int out_cap)
{
#ifdef __APPLE__
	int n = apple_lz4_compress(in, in_n, out, out_cap);
	if (n > 0)
		return n;
#endif
	return LZ4_compress_default(in, out, in_n, out_cap);
}

static int block_decompress(const void *in, int in_n, void *out, int out_cap)
{
#ifdef __APPLE__
	int n = apple_lz4_decompress(in, in_n, out, out_cap);
	if (n > 0)
		return n;
#endif
	return LZ4_decompress_safe(in, out, in_n, out_cap);
}

int odl_compress_host(const void *in, size_t in_bytes,
		      void *out, size_t out_capacity,
		      size_t *out_wire_bytes)
{
	const uint8_t *src = in;
	uint8_t *dst = out;
	struct odl_compress_header hdr;
	struct odl_lz4_chunk *table;
	size_t nchunks, i, off, table_bytes, payload;
	uint8_t *blocks;

	if (!in || !out || !out_wire_bytes || in_bytes == 0)
		return -1;
	if (out_capacity < odl_compress_host_max_wire_bytes(in_bytes) &&
	    out_capacity < sizeof(hdr) + sizeof(struct odl_lz4_chunk) + in_bytes)
		return -1;

	nchunks = (in_bytes + ODL_COMPRESS_CHUNK - 1) / ODL_COMPRESS_CHUNK;
	table_bytes = nchunks * sizeof(struct odl_lz4_chunk);
	if (sizeof(hdr) + table_bytes > out_capacity)
		return -1;

	table = (struct odl_lz4_chunk *)(dst + sizeof(hdr));
	blocks = dst + sizeof(hdr) + table_bytes;
	off = 0;
	payload = 0;

	for (i = 0; i < nchunks; i++) {
		size_t raw = in_bytes - i * ODL_COMPRESS_CHUNK;
		int c;
		int cap;

		if (raw > ODL_COMPRESS_CHUNK)
			raw = ODL_COMPRESS_CHUNK;
		if (sizeof(hdr) + table_bytes + payload >= out_capacity)
			return -1;
		cap = (int)(out_capacity - sizeof(hdr) - table_bytes - payload);
		c = block_compress(src + off, (int)raw, blocks + payload, cap);
		if (c <= 0)
			return -1;
		table[i].raw_bytes = (uint32_t)raw;
		table[i].comp_bytes = (uint32_t)c;
		payload += (size_t)c;
		off += raw;
	}

	if (sizeof(hdr) + table_bytes + payload >= in_bytes)
		return -1; /* not worth it */

	memset(&hdr, 0, sizeof(hdr));
	hdr.magic = ODL_COMPRESS_MAGIC;
	hdr.version = ODL_COMPRESS_VERSION;
	hdr.algo = ODL_ALGO_LZ4_BLOCK;
	hdr.original_bytes = in_bytes;
	hdr.compressed_bytes = table_bytes + payload;
	hdr.num_chunks = (uint32_t)nchunks;
	memcpy(dst, &hdr, sizeof(hdr));

	*out_wire_bytes = sizeof(hdr) + table_bytes + payload;
	return 0;
}

int odl_decompress_host(const void *wire, size_t wire_bytes,
			void *out, size_t out_capacity,
			size_t *out_original_bytes)
{
	const uint8_t *src = wire;
	uint8_t *dst = out;
	struct odl_compress_header hdr;
	const struct odl_lz4_chunk *table;
	size_t table_bytes, i, raw_off, comp_off;
	const uint8_t *blocks;

	if (!wire || !out || wire_bytes < sizeof(hdr))
		return -1;
	memcpy(&hdr, src, sizeof(hdr));
	if (!odl_compress_header_ok(&hdr))
		return -1;
	if (hdr.algo != ODL_ALGO_LZ4_BLOCK)
		return -1; /* nvCOMP blob — Mac cannot decode */
	if (hdr.original_bytes > out_capacity)
		return -1;
	if (hdr.num_chunks == 0)
		return -1;
	table_bytes = (size_t)hdr.num_chunks * sizeof(struct odl_lz4_chunk);
	if (sizeof(hdr) + hdr.compressed_bytes > wire_bytes)
		return -1;
	if (hdr.compressed_bytes < table_bytes)
		return -1;

	table = (const struct odl_lz4_chunk *)(src + sizeof(hdr));
	blocks = src + sizeof(hdr) + table_bytes;
	raw_off = 0;
	comp_off = 0;

	for (i = 0; i < hdr.num_chunks; i++) {
		int n;
		uint32_t raw = table[i].raw_bytes;
		uint32_t comp = table[i].comp_bytes;

		if (raw == 0 || raw > ODL_COMPRESS_CHUNK)
			return -1;
		if (raw_off + raw > hdr.original_bytes)
			return -1;
		if (comp_off + comp > hdr.compressed_bytes - table_bytes)
			return -1;
		n = block_decompress(blocks + comp_off, (int)comp,
				     dst + raw_off, (int)raw);
		if (n != (int)raw)
			return -1;
		raw_off += raw;
		comp_off += comp;
	}
	if (raw_off != hdr.original_bytes)
		return -1;
	if (out_original_bytes)
		*out_original_bytes = hdr.original_bytes;
	return 0;
}
