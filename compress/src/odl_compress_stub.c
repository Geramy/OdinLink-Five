/* SPDX-License-Identifier: MIT */
/**
 * Stub backend when nvCOMP is not linked (ODL_HAS_NVCOMP undefined).
 * Compression is always "unavailable"; NCCL falls back to raw sends.
 */
#include <odl_tb5/odl_compress.h>

#include <stddef.h>

int odl_compress_backend_available(void)
{
	return 0;
}

size_t odl_compress_max_wire_bytes(size_t original_bytes)
{
	/* No nvCOMP. Host LZ4 bound is still a valid staging size. */
	return odl_compress_host_max_wire_bytes(original_bytes);
}

int odl_compress_gpu(const void *d_in, size_t in_bytes,
		     void *d_out, size_t out_capacity,
		     size_t *out_wire_bytes, void *stream)
{
	(void)d_in;
	(void)in_bytes;
	(void)d_out;
	(void)out_capacity;
	(void)out_wire_bytes;
	(void)stream;
	return -1;
}

int odl_decompress_gpu(const void *d_wire, size_t wire_bytes,
		       void *d_out, size_t out_capacity,
		       size_t *out_original_bytes, void *stream)
{
	(void)d_wire;
	(void)wire_bytes;
	(void)d_out;
	(void)out_capacity;
	(void)out_original_bytes;
	(void)stream;
	return -1;
}

void *odl_compress_device_alloc(size_t bytes)
{
	(void)bytes;
	return NULL;
}

void odl_compress_device_free(void *ptr)
{
	(void)ptr;
}
