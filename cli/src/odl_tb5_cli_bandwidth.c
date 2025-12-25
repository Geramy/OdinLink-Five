/*
 * OdinLink Thunderbolt 5 - Bandwidth Test Implementation
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Client (initiator) for bandwidth test. */
int odl_cli_bandwidth_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			     const struct odl_cli_params *params)
{
	uint8_t *data = NULL;
	uint64_t bytes_sent = 0;
	uint64_t t_start, t_now, elapsed_ns;
	uint64_t deadline_ns;
	char fmt_buf[64];
	char size_buf[32];
	int ret;
	int bs_idx;

	for (bs_idx = 0; bs_idx < params->num_block_sizes; bs_idx++) {
		uint32_t block_size = params->block_sizes[bs_idx];

		bytes_sent = 0;

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_TEST_START, 0,
				       &block_size, sizeof(block_size));
		if (ret < 0) {
			fprintf(stderr, "Failed to send TEST_START: %s\n",
				strerror(-ret));
			return ret;
		}

		data = malloc(block_size);
		if (!data)
			return -ENOMEM;
		memset(data, 0xAA, block_size);

		if (params->verbose) {
			printf("  Block size: %s\n",
			       odl_format_size(block_size, size_buf,
					       sizeof(size_buf)));
		}

		t_start = odl_time_ns();
		deadline_ns = (uint64_t)params->duration_sec * 1000000000ULL;

		for (;;) {
			ret = odl_tb5_stream_send(handle, sid, dst,
						  data, block_size);
			if (ret < 0) {
				fprintf(stderr, "Send failed: %s\n",
					strerror(-ret));
				break;
			}

			bytes_sent += block_size;

			t_now = odl_time_ns();
			elapsed_ns = t_now - t_start;
			if (elapsed_ns >= deadline_ns)
				break;
		}

		free(data);
		data = NULL;

		elapsed_ns = odl_time_ns() - t_start;

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_TEST_STOP, 0, NULL, 0);
		if (ret < 0) {
			fprintf(stderr, "Failed to send TEST_STOP: %s\n",
				strerror(-ret));
			return ret;
		}

		printf("  Transferred: %s in %s\n",
		       odl_format_size(bytes_sent, size_buf, sizeof(size_buf)),
		       odl_format_latency(elapsed_ns, fmt_buf, sizeof(fmt_buf)));
		printf("  Throughput:  %s\n",
		       odl_format_throughput(bytes_sent, elapsed_ns,
					    fmt_buf, sizeof(fmt_buf)));

		{
			struct odl_cli_result result;
			char msg_buf[4096];
			uint32_t msg_type, msg_seq;

			memset(&result, 0, sizeof(result));
			result.bytes_transferred = bytes_sent;
			result.elapsed_ns = elapsed_ns;

			ret = odl_cli_send_msg(handle, sid, dst,
					       ODL_CLI_MSG_RESULT, 0,
					       &result.bytes_transferred,
					       sizeof(result) - sizeof(result.hdr));
			if (ret < 0)
				return ret;

			ret = odl_cli_recv_msg(handle, sid, msg_buf,
					       sizeof(msg_buf),
					       &msg_type, &msg_seq, NULL);
			if (ret < 0)
				return ret;

			if (msg_type == ODL_CLI_MSG_RESULT) {
				struct odl_cli_result *srv =
					(struct odl_cli_result *)msg_buf;

				if (params->verbose) {
					printf("  Server received: %s in %s\n",
					       odl_format_size(
						       srv->bytes_transferred,
						       size_buf,
						       sizeof(size_buf)),
					       odl_format_latency(
						       srv->elapsed_ns,
						       fmt_buf,
						       sizeof(fmt_buf)));
				}
			}
		}
	}

	return 0;
}

/* Server (responder) for bandwidth test. */
int odl_cli_bandwidth_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			     const struct odl_cli_test_req *req)
{
	uint8_t *recv_buf = NULL;
	uint64_t bytes_received = 0;
	uint64_t t_start = 0;
	uint64_t elapsed_ns;
	uint64_t deadline_ns;
	char msg_buf[4096];
	uint32_t msg_type, msg_seq;
	int ret;
	bool running = false;

	uint32_t block_size = req->block_size;
	uint32_t recv_buf_size = block_size > 4096 ? block_size : 4096;

	deadline_ns = (uint64_t)req->duration_sec * 1000000000ULL;
	deadline_ns += 2000000000ULL;

	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &msg_type, &msg_seq, NULL);
	if (ret < 0)
		return ret;

	if (msg_type != ODL_CLI_MSG_TEST_START)
		return -EPROTO;

	recv_buf = malloc(recv_buf_size);
	if (!recv_buf)
		return -ENOMEM;

	running = true;
	t_start = odl_time_ns();

	while (running) {
		uint8_t src_id;
		uint32_t actual_len;

		ret = odl_tb5_stream_wait_rx(handle, sid, 2000);
		if (ret == -ETIMEDOUT) {
			elapsed_ns = odl_time_ns() - t_start;
			if (elapsed_ns >= deadline_ns) {
				running = false;
				break;
			}
			continue;
		}
		if (ret < 0)
			break;

		ret = odl_tb5_stream_recv(handle, sid, recv_buf, recv_buf_size,
					  &src_id, &actual_len);
		if (ret < 0)
			break;

		/* Check if this is a CLI control message (TEST_STOP) */
		if (actual_len >= sizeof(struct odl_cli_header)) {
			struct odl_cli_header *hdr =
				(struct odl_cli_header *)recv_buf;
			if (hdr->magic == ODL_CLI_MAGIC &&
			    hdr->type == ODL_CLI_MSG_TEST_STOP) {
				running = false;
				break;
			}
		}

		bytes_received += actual_len;

		elapsed_ns = odl_time_ns() - t_start;
		if (elapsed_ns >= deadline_ns) {
			running = false;
			break;
		}
	}

	free(recv_buf);

	elapsed_ns = odl_time_ns() - t_start;

	/* Receive client's RESULT */
	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &msg_type, &msg_seq, NULL);
	if (ret < 0)
		return ret;

	/* Send server's RESULT */
	{
		struct odl_cli_result result;

		memset(&result, 0, sizeof(result));
		result.bytes_transferred = bytes_received;
		result.elapsed_ns = elapsed_ns;

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_RESULT, 0,
				       &result.bytes_transferred,
				       sizeof(result) - sizeof(result.hdr));
		if (ret < 0)
			return ret;
	}

	return 0;
}
