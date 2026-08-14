/* SPDX-License-Identifier: MIT */
/**
 * OdinLink optional GPU compression (nvCOMP) for the TB link path.
 *
 * Wire format when compression is used:
 *   [odl_compress_header | compressed payload | optional padding to max]
 *
 * Env (both peers should match):
 *   ODL_COMPRESS=1
 *   ODL_COMPRESS_ALGO=gdeflate|lz4|snappy
 *   ODL_COMPRESS_THRESHOLD=262144
 *   ODL_COMPRESS_LEVEL=1
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ODL_COMPRESS_MAGIC   0x4F444C43u /* 'ODLC' */
#define ODL_COMPRESS_VERSION 1

enum odl_compress_algo {
	ODL_ALGO_NONE     = 0,
	ODL_ALGO_GDEFLATE = 1,
	ODL_ALGO_LZ4      = 2,
	ODL_ALGO_SNAPPY   = 3,
};

struct odl_compress_header {
	uint32_t magic;
	uint16_t version;
	uint16_t algo;
	uint64_t original_bytes;
	uint64_t compressed_bytes;
	uint32_t num_chunks;
	uint32_t reserved;
} __attribute__((packed));

/* ── config ──────────────────────────────────────────────────────── */

/** Parse env once; safe to call repeatedly. Returns 1 if ODL_COMPRESS enabled. */
int odl_compress_init(void);

/** 1 if compression is enabled and nvCOMP is available. */
int odl_compress_enabled(void);

/** Algo currently selected (after init). */
int odl_compress_algo(void);

/** Minimum message size (bytes) before compression is attempted. */
size_t odl_compress_threshold(void);

/**
 * Upper bound on wire size for a payload of `original_bytes`
 * (header + max compressed). Use this for staging allocation.
 */
size_t odl_compress_max_wire_bytes(size_t original_bytes);

/* ── GPU compress / decompress ───────────────────────────────────── */

/**
 * Compress device buffer into wire format starting at d_out.
 * On success: *out_wire_bytes = header + compressed (no padding).
 * Returns 0 on success, -1 to fall back to raw (or error).
 *
 * stream may be 0 (default stream).
 */
int odl_compress_gpu(const void *d_in, size_t in_bytes,
		     void *d_out, size_t out_capacity,
		     size_t *out_wire_bytes, void *stream /* cudaStream_t */);

/**
 * Decompress wire buffer (must start with valid header) into d_out.
 * d_out must hold header.original_bytes.
 * Returns 0 on success, -1 on error.
 */
int odl_decompress_gpu(const void *d_wire, size_t wire_bytes,
		       void *d_out, size_t out_capacity,
		       size_t *out_original_bytes, void *stream);

/** Host-side header probe (does not touch device). 1 if magic matches. */
int odl_compress_header_ok(const struct odl_compress_header *hdr);

/**
 * Device memory helpers (cudaMalloc/Free when nvCOMP backend is present).
 * Stub returns NULL / no-op so callers need no #ifdef.
 */
void *odl_compress_device_alloc(size_t bytes);
void odl_compress_device_free(void *ptr);

/**
 * Should this message try compression? (enabled + size >= threshold + backend).
 * type_is_cuda: 1 for GPU pointers, 0 for host (v1 host compress not implemented).
 */
int odl_compress_should(size_t size, int type_is_cuda);

#ifdef __cplusplus
}
#endif
