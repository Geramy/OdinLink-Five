/* SPDX-License-Identifier: MIT */
/**
 * nvCOMP-backed GPU compress/decompress for OdinLink.
 * Built only when ODL_HAS_NVCOMP is defined (CMake found nvCOMP).
 */
#include <odl_tb5/odl_compress.h>

#include <cuda_runtime.h>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/nvcompManager.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

extern "C" int odl_compress_algo(void);
extern "C" int odl_compress_init(void);

namespace {

constexpr size_t kChunk = 1 << 16; /* 64 KiB chunks — good GPU occupancy */

std::unique_ptr<nvcomp::nvcompManagerBase> make_manager(int algo, cudaStream_t stream)
{
	using namespace nvcomp;
	switch (algo) {
	case ODL_ALGO_LZ4:
		return std::unique_ptr<nvcompManagerBase>(
			new LZ4Manager(kChunk,
				       nvcompBatchedLZ4CompressDefaultOpts,
				       nvcompBatchedLZ4DecompressDefaultOpts,
				       stream));
	case ODL_ALGO_SNAPPY:
		return std::unique_ptr<nvcompManagerBase>(
			new SnappyManager(kChunk,
					  nvcompBatchedSnappyCompressDefaultOpts,
					  nvcompBatchedSnappyDecompressDefaultOpts,
					  stream));
	case ODL_ALGO_GDEFLATE:
	default:
		return std::unique_ptr<nvcompManagerBase>(
			new GdeflateManager(kChunk,
					    nvcompBatchedGdeflateCompressDefaultOpts,
					    nvcompBatchedGdeflateDecompressDefaultOpts,
					    stream));
	}
}

} // namespace

extern "C" int odl_compress_backend_available(void)
{
	return 1;
}

extern "C" void *odl_compress_device_alloc(size_t bytes)
{
	void *p = nullptr;
	if (cudaMalloc(&p, bytes) != cudaSuccess)
		return nullptr;
	return p;
}

extern "C" void odl_compress_device_free(void *ptr)
{
	if (ptr)
		cudaFree(ptr);
}

extern "C" size_t odl_compress_max_wire_bytes(size_t original_bytes)
{
	/* nvCOMP max_compressed is often > original (format overhead on incompressible data). */
	try {
		auto mgr = make_manager(odl_compress_algo(), nullptr);
		auto cfg = mgr->configure_compression(original_bytes);
		return sizeof(struct odl_compress_header) + cfg.max_compressed_buffer_size;
	} catch (...) {
		return sizeof(struct odl_compress_header) + original_bytes * 2 + (1 << 20);
	}
}

extern "C" int odl_compress_gpu(const void *d_in, size_t in_bytes,
				void *d_out, size_t out_capacity,
				size_t *out_wire_bytes, void *stream_v)
{
	if (!d_in || !d_out || !out_wire_bytes || in_bytes == 0)
		return -1;
	odl_compress_init();
	int algo = odl_compress_algo();
	if (algo == ODL_ALGO_NONE)
		return -1;

	cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
	try {
		auto mgr = make_manager(algo, stream);
		auto cfg = mgr->configure_compression(in_bytes);
		size_t max_comp = cfg.max_compressed_buffer_size;
		if (sizeof(struct odl_compress_header) + max_comp > out_capacity)
			return -1;

		uint8_t *payload = static_cast<uint8_t *>(d_out) +
				   sizeof(struct odl_compress_header);
		/* Device size_t for compressed length */
		size_t *d_comp_size = nullptr;
		if (cudaMalloc(reinterpret_cast<void **>(&d_comp_size), sizeof(size_t)) !=
		    cudaSuccess)
			return -1;

		mgr->compress(static_cast<const uint8_t *>(d_in), payload, cfg, d_comp_size);
		size_t comp_size = 0;
		cudaMemcpyAsync(&comp_size, d_comp_size, sizeof(size_t),
				cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		cudaFree(d_comp_size);

		if (comp_size == 0 ||
		    comp_size >= in_bytes) {
			/* Not worth it — caller falls back to raw */
			return -1;
		}

		struct odl_compress_header hdr = {};
		hdr.magic = ODL_COMPRESS_MAGIC;
		hdr.version = ODL_COMPRESS_VERSION;
		hdr.algo = static_cast<uint16_t>(algo);
		hdr.original_bytes = in_bytes;
		hdr.compressed_bytes = comp_size;
		hdr.num_chunks = static_cast<uint32_t>(cfg.num_chunks);
		cudaMemcpyAsync(d_out, &hdr, sizeof(hdr), cudaMemcpyHostToDevice, stream);
		cudaStreamSynchronize(stream);

		*out_wire_bytes = sizeof(hdr) + comp_size;
		return 0;
	} catch (const std::exception &ex) {
		fprintf(stderr, "ODL_COMPRESS: compress failed: %s\n", ex.what());
		return -1;
	}
}

extern "C" int odl_decompress_gpu(const void *d_wire, size_t wire_bytes,
				  void *d_out, size_t out_capacity,
				  size_t *out_original_bytes, void *stream_v)
{
	if (!d_wire || !d_out || wire_bytes < sizeof(struct odl_compress_header))
		return -1;

	cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
	struct odl_compress_header hdr = {};
	cudaMemcpyAsync(&hdr, d_wire, sizeof(hdr), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	if (!odl_compress_header_ok(&hdr))
		return -1;
	if (hdr.original_bytes > out_capacity)
		return -1;
	if (sizeof(hdr) + hdr.compressed_bytes > wire_bytes)
		return -1;

	try {
		auto mgr = make_manager(static_cast<int>(hdr.algo), stream);
		/* API: decompress(decomp_out, comp_in, config) */
		uint8_t *payload =
			const_cast<uint8_t *>(static_cast<const uint8_t *>(d_wire) +
					     sizeof(hdr));
		auto dcfg = mgr->configure_decompression(payload);
		mgr->decompress(static_cast<uint8_t *>(d_out), payload, dcfg);
		cudaStreamSynchronize(stream);
		if (out_original_bytes)
			*out_original_bytes = hdr.original_bytes;
		return 0;
	} catch (const std::exception &ex) {
		fprintf(stderr, "ODL_COMPRESS: decompress failed: %s\n", ex.what());
		return -1;
	}
}
