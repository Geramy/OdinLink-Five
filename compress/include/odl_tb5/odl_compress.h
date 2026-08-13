/* SPDX-License-Identifier: MIT */
/**
 * OdinLink optional compression for the TB link path.
 *
 * Wire (little-endian, packed):
 *   [odl_compress_header | payload]
 *
 * Two payload families share this header:
 *
 *   1) nvCOMP native (Linux CUDA only) — algo 1/2/3
 *      GDeflate / batched LZ4 / Snappy from nvCOMP managers.
 *      A Mac cannot decode these. Linux↔Linux NCCL only.
 *
 *   2) Portable LZ4 blocks — algo 4 (ODL_ALGO_LZ4_BLOCK)
 *      This is what the Mac and the TB-bridge speak.
 *
 *      payload:
 *        num_chunks × { u32 raw_bytes, u32 comp_bytes }   (LE)
 *        then concatenated standard LZ4 raw blocks
 *        (LZ4_compress_default / Apple COMPRESSION_LZ4_RAW)
 *
 *      compressed_bytes = table + all blocks. Chunks are 64 KiB
 *      except the last. Empty input is never wrapped.
 *
 * Env:
 *   ODL_COMPRESS=1
 *   ODL_COMPRESS_ALGO=gdeflate|lz4|snappy|lz4_block
 *   ODL_COMPRESS_THRESHOLD=262144
 *   ODL_COMPRESS_LEVEL=1
 *
 * NCCL (CUDA) uses gdeflate/lz4/snappy when nvCOMP is linked.
 * The Mac / bridge path always writes and reads lz4_block.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ODL_COMPRESS_MAGIC   0x4F444C43u /* 'ODLC' as LE u32 */
#define ODL_COMPRESS_VERSION 1
#define ODL_COMPRESS_CHUNK   65536u

enum odl_compress_algo {
	ODL_ALGO_NONE      = 0,
	ODL_ALGO_GDEFLATE  = 1, /* nvCOMP only */
	ODL_ALGO_LZ4       = 2, /* nvCOMP batched LZ4 — not Mac-readable */
	ODL_ALGO_SNAPPY    = 3, /* nvCOMP only */
	ODL_ALGO_LZ4_BLOCK = 4, /* portable, Mac + host */
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

struct odl_lz4_chunk {
	uint32_t raw_bytes;
	uint32_t comp_bytes;
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
 * Should this message try GPU compression? (enabled + size + nvCOMP).
 * type_is_cuda: 1 for GPU pointers. Host always returns 0 here —
 * use odl_compress_should_host() for the Mac/bridge path.
 */
int odl_compress_should(size_t size, int type_is_cuda);

/* ── Host / Mac portable LZ4 (algo 4) ────────────────────────────── */

/** Always 1 — vendored LZ4, no nvCOMP required. */
int odl_compress_host_available(void);

/** 1 if ODL_COMPRESS is on and host LZ4 is available. */
int odl_compress_host_enabled(void);

/** 1 if size >= threshold and host compression is enabled. */
int odl_compress_should_host(size_t size);

/** Upper bound on an lz4_block wire blob for `original_bytes`. */
size_t odl_compress_host_max_wire_bytes(size_t original_bytes);

/**
 * Compress host buffer into ODLC + lz4_block.
 * Returns 0 on success (and *out_wire_bytes < in_bytes).
 * Returns -1 to keep the raw payload (incompressible / error).
 */
int odl_compress_host(const void *in, size_t in_bytes,
		      void *out, size_t out_capacity,
		      size_t *out_wire_bytes);

/**
 * Decompress an ODLC lz4_block blob. Rejects nvCOMP algos (1/2/3).
 * d_out must hold header.original_bytes.
 */
int odl_decompress_host(const void *wire, size_t wire_bytes,
			void *out, size_t out_capacity,
			size_t *out_original_bytes);

/** 1 if buf starts with a valid ODLC header. */
int odl_compress_looks_compressed(const void *buf, size_t n);

#ifdef __cplusplus
}
#endif
