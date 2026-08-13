/* SPDX-License-Identifier: MIT */
/* Micro-bench: raw memcpy timing vs compress+decompress (same GPU). */
#include <odl_tb5/odl_compress.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double ms_now()
{
	cudaDeviceSynchronize();
	return 0; /* use events below */
}

int main(int argc, char **argv)
{
	size_t nbytes = 4 << 20; /* 4 MiB */
	int iters = 20;
	if (argc > 1)
		nbytes = strtoull(argv[1], nullptr, 10);
	if (argc > 2)
		iters = atoi(argv[2]);

	setenv("ODL_COMPRESS", "1", 0);
	odl_compress_init();
	if (!odl_compress_enabled()) {
		fprintf(stderr,
			"compression not enabled (need ODL_COMPRESS=1 and nvCOMP build)\n");
		return 2;
	}

	uint8_t *d_in = nullptr, *d_wire = nullptr, *d_out = nullptr;
	cudaMalloc(&d_in, nbytes);
	/* compressible pattern */
	std::vector<uint8_t> h(nbytes);
	for (size_t i = 0; i < nbytes; i++)
		h[i] = static_cast<uint8_t>(i & 0x3f);
	cudaMemcpy(d_in, h.data(), nbytes, cudaMemcpyHostToDevice);

	size_t wire_cap = odl_compress_max_wire_bytes(nbytes);
	cudaMalloc(&d_wire, wire_cap);
	cudaMalloc(&d_out, nbytes);

	size_t wire_bytes = 0;
	if (odl_compress_gpu(d_in, nbytes, d_wire, wire_cap, &wire_bytes, nullptr) != 0) {
		fprintf(stderr, "compress failed or not beneficial\n");
		return 1;
	}
	printf("in=%zu  wire=%zu  ratio=%.3f\n", nbytes, wire_bytes,
	       double(nbytes) / double(wire_bytes));

	size_t out_bytes = 0;
	if (odl_decompress_gpu(d_wire, wire_bytes, d_out, nbytes, &out_bytes, nullptr) != 0) {
		fprintf(stderr, "decompress failed\n");
		return 1;
	}

	std::vector<uint8_t> hout(nbytes);
	cudaMemcpy(hout.data(), d_out, nbytes, cudaMemcpyDeviceToHost);
	if (memcmp(h.data(), hout.data(), nbytes) != 0) {
		fprintf(stderr, "MISMATCH after round-trip\n");
		return 1;
	}
	printf("round-trip OK\n");

	cudaEvent_t a, b;
	cudaEventCreate(&a);
	cudaEventCreate(&b);
	float ms = 0;
	cudaEventRecord(a);
	for (int i = 0; i < iters; i++) {
		size_t w = 0;
		odl_compress_gpu(d_in, nbytes, d_wire, wire_cap, &w, nullptr);
		size_t o = 0;
		odl_decompress_gpu(d_wire, w, d_out, nbytes, &o, nullptr);
	}
	cudaEventRecord(b);
	cudaEventSynchronize(b);
	cudaEventElapsedTime(&ms, a, b);
	printf("avg compress+decompress: %.3f ms (n=%d)\n", ms / iters, iters);
	printf("effective GB/s (in+out)/time: %.2f\n",
	       (2.0 * nbytes / 1e9) / (ms / iters / 1e3));

	cudaFree(d_in);
	cudaFree(d_wire);
	cudaFree(d_out);
	return 0;
}
