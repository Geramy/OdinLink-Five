/*
 * OdinLink — CLI: MIMO Test (Multiple Streams at Once)
 *
 * Opens several streams in parallel and blasts data through all of them
 * simultaneously. Tests how well the multiplexed I/O path handles
 * concurrent traffic — relevant for NCCL collective operations where
 * multiple GPUs are sending at the same time.
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

struct mimo_stream_ctx {
	odl_tb5_t    handle;
	uint8_t      sid;
	uint8_t      dst;
	int          stream_id;
	uint32_t     block_size;
	uint32_t     duration_sec;
	volatile bool stop;
	uint64_t     bytes_transferred;
	uint64_t     elapsed_ns;
};

/* Worker thread for a single MIMO stream. */
static void *mimo_stream_thread(void *arg)
{
	struct mimo_stream_ctx *ctx = (struct mimo_stream_ctx *)arg;
	uint8_t *data;
	uint64_t t_start, t_end;
	uint64_t deadline_ns;
	int ret;

	data = malloc(ctx->block_size);
	if (!data)
		return NULL;
	memset(data, 0xAA, ctx->block_size);

	t_start = odl_time_ns();
	deadline_ns = t_start + (uint64_t)ctx->duration_sec * 1000000000ULL;

	while (!ctx->stop) {
		if (odl_time_ns() >= deadline_ns)
			break;

		ret = odl_tb5_stream_send(ctx->handle, ctx->sid, ctx->dst,
					  data, ctx->block_size);
		if (ret < 0)
			break;

		ctx->bytes_transferred += ctx->block_size;
	}

	t_end = odl_time_ns();
	ctx->elapsed_ns = t_end - t_start;

	free(data);
	return NULL;
}

/* Client (initiator) for MIMO test. */
int odl_cli_mimo_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			 const struct odl_cli_params *params)
{
	struct mimo_stream_ctx *streams = NULL;
	pthread_t *threads = NULL;
	uint32_t num_streams;
	uint32_t block_size;
	uint32_t duration_sec;
	uint64_t total_bytes = 0;
	uint64_t max_elapsed_ns = 0;
	char msg_buf[4096];
	uint32_t type, seq;
	int ret;

	num_streams = params->num_streams ? params->num_streams
					  : ODL_DEFAULT_STREAMS;
	block_size = params->block_sizes[0] ? params->block_sizes[0]
					    : ODL_DEFAULT_BLOCK_SIZE;
	duration_sec = params->duration_sec ? params->duration_sec
					    : ODL_DEFAULT_DURATION;

	char size_buf[32];
	printf("  Streams: %u\n", num_streams);
	printf("  Block size: %s\n",
	       odl_format_size(block_size, size_buf, sizeof(size_buf)));
	printf("  Duration: %u seconds\n", duration_sec);

	ret = odl_cli_send_msg(handle, sid, dst,
			       ODL_CLI_MSG_TEST_START, 0, NULL, 0);
	if (ret < 0)
		return ret;

	streams = calloc(num_streams, sizeof(*streams));
	if (!streams) {
		ret = -ENOMEM;
		goto out_stop;
	}

	threads = calloc(num_streams, sizeof(*threads));
	if (!threads) {
		ret = -ENOMEM;
		goto out_free;
	}

	for (uint32_t i = 0; i < num_streams; i++) {
		streams[i].handle = handle;
		streams[i].sid = sid;
		streams[i].dst = dst;
		streams[i].stream_id = (int)i;
		streams[i].block_size = block_size;
		streams[i].duration_sec = duration_sec;
		streams[i].stop = false;
		streams[i].bytes_transferred = 0;
		streams[i].elapsed_ns = 0;
	}

	printf("  Launching %u streams...\n", num_streams);
	for (uint32_t i = 0; i < num_streams; i++) {
		ret = pthread_create(&threads[i], NULL,
				     mimo_stream_thread, &streams[i]);
		if (ret != 0) {
			fprintf(stderr, "  Failed to create thread %u: %s\n",
				i, strerror(ret));
			for (uint32_t j = 0; j < i; j++)
				streams[j].stop = true;
			for (uint32_t j = 0; j < i; j++)
				pthread_join(threads[j], NULL);
			ret = -ret;
			goto out_free;
		}
	}

	{
		struct timespec ts;
		ts.tv_sec = duration_sec;
		ts.tv_nsec = 0;
		nanosleep(&ts, NULL);
	}

	for (uint32_t i = 0; i < num_streams; i++)
		streams[i].stop = true;

	for (uint32_t i = 0; i < num_streams; i++)
		pthread_join(threads[i], NULL);

	printf("\n  === Per-Stream Results ===\n");
	for (uint32_t i = 0; i < num_streams; i++) {
		uint64_t bytes = streams[i].bytes_transferred;
		uint64_t ns = streams[i].elapsed_ns;

		total_bytes += bytes;
		if (ns > max_elapsed_ns)
			max_elapsed_ns = ns;

		double gbytes_s = 0.0;
		if (ns > 0)
			gbytes_s = (double)bytes / (double)ns;

		printf("  Stream %d: %.2f GB/s (%lu bytes in ",
		       streams[i].stream_id, gbytes_s,
		       (unsigned long)bytes);

		char lat_buf[64];
		printf("%s)\n", odl_format_latency(ns, lat_buf, sizeof(lat_buf)));
	}

	{
		char tp_buf[64];
		double agg_gbytes_s = 0.0;

		if (max_elapsed_ns > 0)
			agg_gbytes_s = (double)total_bytes / (double)max_elapsed_ns;

		printf("\n  === Aggregate ===\n");
		printf("  Total: %.2f GB/s across %u streams\n",
		       agg_gbytes_s, num_streams);
		printf("  Total data: %s\n",
		       odl_format_size(total_bytes, tp_buf, sizeof(tp_buf)));
		printf("  Throughput: %s\n",
		       odl_format_throughput(total_bytes, max_elapsed_ns,
					    tp_buf, sizeof(tp_buf)));
	}

	ret = 0;

out_free:
	free(threads);
	free(streams);

out_stop:
	odl_cli_send_msg(handle, sid, dst,
			 ODL_CLI_MSG_TEST_STOP, 0, NULL, 0);

	{
		struct odl_cli_result result;

		memset(&result, 0, sizeof(result));
		result.bytes_transferred = total_bytes;
		result.elapsed_ns = max_elapsed_ns;

		odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_RESULT, 0,
				 &result.bytes_transferred,
				 sizeof(result) - sizeof(result.hdr));

		odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				 &type, &seq, NULL);
	}

	return ret;
}

/* Server (responder) for MIMO test. */
int odl_cli_mimo_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			 const struct odl_cli_test_req *req)
{
	uint8_t *recv_buf = NULL;
	uint32_t recv_buf_size;
	char msg_buf[4096];
	uint32_t type, seq;
	uint64_t bytes_received = 0;
	uint64_t t_start, t_end = 0;
	int ret;

	recv_buf_size = ODL_DEFAULT_BLOCK_SIZE > 4096 ?
			ODL_DEFAULT_BLOCK_SIZE : 4096;

	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &type, &seq, NULL);
	if (ret < 0)
		return ret;

	if (type != ODL_CLI_MSG_TEST_START)
		return -EPROTO;

	printf("  [Server] MIMO test started (%u streams)\n",
	       req->num_streams);

	recv_buf = malloc(recv_buf_size);
	if (!recv_buf)
		return -ENOMEM;

	t_start = odl_time_ns();

	for (;;) {
		uint8_t src_id;
		uint32_t actual_len;

		ret = odl_tb5_stream_wait_rx(handle, sid, 2000);
		if (ret == -ETIMEDOUT)
			continue;
		if (ret < 0)
			break;

		ret = odl_tb5_stream_recv(handle, sid, recv_buf, recv_buf_size,
					  &src_id, &actual_len);
		if (ret < 0)
			break;

		/* Check for CLI control messages */
		if (actual_len >= sizeof(struct odl_cli_header)) {
			struct odl_cli_header *hdr =
				(struct odl_cli_header *)recv_buf;
			if (hdr->magic == ODL_CLI_MAGIC &&
			    hdr->type == ODL_CLI_MSG_TEST_STOP) {
				t_end = odl_time_ns();
				break;
			}
		}

		bytes_received += actual_len;
	}

	if (t_end == 0)
		t_end = odl_time_ns();

	free(recv_buf);

	{
		char size_buf2[64], tp_buf[64];
		uint64_t elapsed = t_end - t_start;

		printf("  [Server] MIMO test stopped\n");
		printf("  [Server] Received: %s in ",
		       odl_format_size(bytes_received, size_buf2,
				       sizeof(size_buf2)));

		char lat_buf[64];
		printf("%s\n", odl_format_latency(elapsed, lat_buf,
						   sizeof(lat_buf)));
		printf("  [Server] Throughput: %s\n",
		       odl_format_throughput(bytes_received, elapsed,
					    tp_buf, sizeof(tp_buf)));
	}

	/* Receive client's RESULT, send server's RESULT */
	{
		struct odl_cli_result result;

		memset(&result, 0, sizeof(result));
		result.bytes_transferred = bytes_received;
		result.elapsed_ns = t_end - t_start;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
		if (ret < 0)
			return ret;

		odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_RESULT, 0,
				 &result.bytes_transferred,
				 sizeof(result) - sizeof(result.hdr));
	}

	return 0;
}
