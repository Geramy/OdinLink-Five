/*
 * OdinLink Thunderbolt 5 - Latency Under Load Test
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Client (initiator) for latency-under-load test. */
int odl_cli_latency_load_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
				const struct odl_cli_params *params)
{
	struct odl_stats load_stats;
	struct odl_stats idle_stats;
	char msg_buf[4096];
	uint32_t type, seq;
	uint64_t t_start, t_end, rtt;
	uint32_t bg_block_size;
	uint32_t iterations;
	uint32_t warmup;
	uint8_t *bg_data = NULL;
	int ret;

	bg_block_size = params->bg_block_size ? params->bg_block_size
					      : ODL_DEFAULT_BG_BLOCK_SIZE;
	iterations = params->iterations ? params->iterations
					: ODL_DEFAULT_ITERATIONS;
	warmup = params->warmup_iters ? params->warmup_iters
				      : ODL_DEFAULT_WARMUP;

	char size_buf[32];
	printf("  Background block size: %s\n",
	       odl_format_size(bg_block_size, size_buf, sizeof(size_buf)));
	printf("  Iterations: %u\n", iterations);

	ret = odl_cli_send_msg(handle, sid, dst,
			       ODL_CLI_MSG_TEST_START, 0, NULL, 0);
	if (ret < 0)
		return ret;

	ret = odl_stats_init(&idle_stats, iterations);
	if (ret < 0)
		return -ENOMEM;

	printf("  Measuring idle latency...\n");

	for (uint32_t i = 0; i < warmup; i++) {
		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PING, i, NULL, 0);
		if (ret < 0)
			goto out_idle;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
		if (ret < 0)
			goto out_idle;
	}

	for (uint32_t i = 0; i < iterations; i++) {
		t_start = odl_time_ns();

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PING, i, NULL, 0);
		if (ret < 0)
			goto out_idle;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
		if (ret < 0)
			goto out_idle;

		if (type != ODL_CLI_MSG_PONG) {
			ret = -EPROTO;
			goto out_idle;
		}

		t_end = odl_time_ns();
		rtt = t_end - t_start;
		odl_stats_add(&idle_stats, rtt);
	}

	odl_stats_finalize(&idle_stats);
	odl_stats_print(&idle_stats, "Idle Latency");

	ret = odl_stats_init(&load_stats, iterations);
	if (ret < 0)
		goto out_idle;

	printf("  Measuring latency under load...\n");

	bg_data = malloc(bg_block_size);
	if (!bg_data) {
		ret = -ENOMEM;
		goto out_load;
	}
	memset(bg_data, 0xBB, bg_block_size);

	for (uint32_t i = 0; i < warmup; i++) {
		ret = odl_tb5_stream_send(handle, sid, dst,
					  bg_data, bg_block_size);
		if (ret < 0)
			goto out_load;

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PING, i, NULL, 0);
		if (ret < 0)
			goto out_load;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
		if (ret < 0)
			goto out_load;
	}

	for (uint32_t i = 0; i < iterations; i++) {
		ret = odl_tb5_stream_send(handle, sid, dst,
					  bg_data, bg_block_size);
		if (ret < 0)
			goto out_load;

		t_start = odl_time_ns();

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PING, i, NULL, 0);
		if (ret < 0)
			goto out_load;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
		if (ret < 0)
			goto out_load;

		if (type != ODL_CLI_MSG_PONG) {
			ret = -EPROTO;
			goto out_load;
		}

		t_end = odl_time_ns();
		rtt = t_end - t_start;
		odl_stats_add(&load_stats, rtt);

		if (!params->quiet && (i + 1) % (iterations / 10 + 1) == 0) {
			printf("  [%u/%u] current RTT: ",
			       i + 1, iterations);
			char lat_buf[64];
			printf("%s\n",
			       odl_format_latency(rtt, lat_buf, sizeof(lat_buf)));
		}
	}

	odl_stats_finalize(&load_stats);
	odl_stats_print(&load_stats, "Latency Under Load");
	odl_stats_print_histogram(&load_stats);

	{
		char idle_buf[64], load_buf[64];
		double idle_us = idle_stats.avg_ns / 1000.0;
		double load_us = load_stats.avg_ns / 1000.0;
		double degradation = 0.0;

		if (idle_us > 0.0)
			degradation = ((load_us - idle_us) / idle_us) * 100.0;

		printf("\n  === Latency Comparison ===\n");
		printf("  Idle latency:       %s\n",
		       odl_format_latency((uint64_t)idle_stats.avg_ns,
					  idle_buf, sizeof(idle_buf)));
		printf("  Under load:         %s\n",
		       odl_format_latency((uint64_t)load_stats.avg_ns,
					  load_buf, sizeof(load_buf)));
		printf("  Degradation:        %.1f%%\n", degradation);
	}

	if (params->output_file)
		odl_stats_write_csv(&load_stats, params->output_file);

	ret = 0;

out_load:
	free(bg_data);
	odl_stats_free(&load_stats);
out_idle:
	odl_stats_free(&idle_stats);

	odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_TEST_STOP, 0, NULL, 0);

	{
		struct odl_cli_result result;

		memset(&result, 0, sizeof(result));
		result.bytes_transferred = (uint64_t)iterations * bg_block_size;
		result.min_latency_ns = load_stats.min_ns;
		result.max_latency_ns = load_stats.max_ns;
		result.avg_latency_ns = (uint64_t)load_stats.avg_ns;
		result.p50_latency_ns = load_stats.p50_ns;
		result.p99_latency_ns = load_stats.p99_ns;
		result.p999_latency_ns = load_stats.p999_ns;
		result.jitter_ns = (uint64_t)load_stats.stddev_ns;

		odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_RESULT, 0,
				 &result.bytes_transferred,
				 sizeof(result) - sizeof(result.hdr));
	}

	return ret;
}

/* Server (responder) for latency-under-load test. */
int odl_cli_latency_load_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
				const struct odl_cli_test_req *req)
{
	uint8_t *recv_buf = NULL;
	uint32_t recv_buf_size;
	uint64_t bytes_received = 0;
	int ret;

	(void)req;

	recv_buf_size = ODL_DEFAULT_BG_BLOCK_SIZE > 4096 ?
			ODL_DEFAULT_BG_BLOCK_SIZE : 4096;
	recv_buf = malloc(recv_buf_size);
	if (!recv_buf)
		return -ENOMEM;

	/* Wait for TEST_START */
	{
		char msg_buf[4096];
		uint32_t type, seq;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
		if (ret < 0)
			goto out;

		if (type != ODL_CLI_MSG_TEST_START) {
			ret = -EPROTO;
			goto out;
		}
	}

	printf("  [Server] Latency-under-load test started\n");

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

		/* Check if this is a CLI control message */
		if (actual_len >= sizeof(struct odl_cli_header)) {
			struct odl_cli_header *hdr =
				(struct odl_cli_header *)recv_buf;

			if (hdr->magic == ODL_CLI_MAGIC) {
				switch (hdr->type) {
				case ODL_CLI_MSG_PING:
					ret = odl_cli_send_msg(handle, sid, dst,
							       ODL_CLI_MSG_PONG,
							       hdr->sequence,
							       NULL, 0);
					if (ret < 0)
						goto out;
					break;

				case ODL_CLI_MSG_TEST_STOP:
					printf("  [Server] Test stopped "
					       "(received %lu bytes of BW data)\n",
					       (unsigned long)bytes_received);

					/* Receive client's RESULT */
					{
						char msg_buf2[4096];
						uint32_t type2, seq2;

						ret = odl_cli_recv_msg(handle, sid,
								       msg_buf2,
								       sizeof(msg_buf2),
								       &type2, &seq2,
								       NULL);
					}
					goto out;

				default:
					break;
				}
				continue;
			}
		}

		/* Raw bandwidth data */
		bytes_received += actual_len;
	}

out:
	free(recv_buf);
	return (ret < 0) ? ret : 0;
}
