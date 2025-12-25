/*
 * OdinLink Thunderbolt 5 - Latency Ping-Pong Test
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Client (initiator) for latency ping-pong test. */
int odl_cli_latency_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			    const struct odl_cli_params *params)
{
	struct odl_stats stats;
	uint32_t total_iters;
	char fmt_buf[64];
	char msg_buf[4096];
	uint32_t msg_type, msg_seq;
	int ret;

	total_iters = params->warmup_iters + params->iterations;

	ret = odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_TEST_START, 0,
			       &params->iterations,
			       sizeof(params->iterations));
	if (ret < 0) {
		fprintf(stderr, "Failed to send TEST_START: %s\n",
			strerror(-ret));
		return ret;
	}

	ret = odl_stats_init(&stats, params->iterations);
	if (ret < 0) {
		fprintf(stderr, "Failed to allocate stats buffer\n");
		return ret;
	}

	if (params->verbose) {
		printf("  Warmup iterations: %u\n", params->warmup_iters);
		printf("  Measurement iterations: %u\n", params->iterations);
	}

	for (uint32_t i = 0; i < total_iters; i++) {
		bool is_warmup = (i < params->warmup_iters);
		uint64_t t0, t1, rtt;
		uint32_t type;

		t0 = odl_time_ns();

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PING, i, NULL, 0);
		if (ret < 0) {
			fprintf(stderr, "Send PING failed: %s\n",
				strerror(-ret));
			goto out_free;
		}

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, NULL, NULL);
		if (ret < 0)
			goto out_free;

		if (type != ODL_CLI_MSG_PONG) {
			fprintf(stderr, "Expected PONG, got type 0x%x\n", type);
			ret = -EPROTO;
			goto out_free;
		}

		t1 = odl_time_ns();
		rtt = t1 - t0;

		if (!is_warmup)
			odl_stats_add(&stats, rtt);

		if (params->verbose && !is_warmup && (i % 1000 == 0)) {
			printf("  [%u/%u] RTT: %s\n",
			       i - params->warmup_iters + 1,
			       params->iterations,
			       odl_format_latency(rtt, fmt_buf,
						  sizeof(fmt_buf)));
		}
	}

	odl_stats_finalize(&stats);

	odl_stats_print(&stats, "Latency");
	odl_stats_print_histogram(&stats);

	if (params->output_file)
		odl_stats_write_csv(&stats, params->output_file);

	ret = odl_cli_send_msg(handle, sid, dst,
			       ODL_CLI_MSG_TEST_STOP, 0, NULL, 0);
	if (ret < 0) {
		fprintf(stderr, "Failed to send TEST_STOP: %s\n",
			strerror(-ret));
		goto out_free;
	}

	{
		struct odl_cli_result result;

		memset(&result, 0, sizeof(result));
		result.bytes_transferred =
			(uint64_t)params->iterations *
			sizeof(struct odl_cli_header) * 2;
		result.elapsed_ns = stats.sum_ns;
		result.min_latency_ns = stats.min_ns;
		result.max_latency_ns = stats.max_ns;
		result.avg_latency_ns = (uint64_t)stats.avg_ns;
		result.p50_latency_ns = stats.p50_ns;
		result.p99_latency_ns = stats.p99_ns;
		result.p999_latency_ns = stats.p999_ns;
		result.jitter_ns = (uint64_t)stats.stddev_ns;

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_RESULT, 0,
				       &result.bytes_transferred,
				       sizeof(result) - sizeof(result.hdr));
		if (ret < 0)
			goto out_free;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &msg_type, &msg_seq, NULL);
		if (ret < 0)
			goto out_free;

		if (msg_type != ODL_CLI_MSG_RESULT && params->verbose) {
			fprintf(stderr, "Expected RESULT from server, got 0x%x\n",
				msg_type);
		}
	}

	ret = 0;

out_free:
	odl_stats_free(&stats);
	return ret;
}

/* Server (responder) for latency ping-pong test. */
int odl_cli_latency_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			    const struct odl_cli_test_req *req)
{
	char msg_buf[4096];
	uint32_t msg_type, msg_seq;
	uint64_t bytes_reflected = 0;
	uint64_t t_start;
	int ret;

	uint64_t timeout_ns = ((uint64_t)req->duration_sec * 2 + 30) *
			       1000000000ULL;

	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &msg_type, &msg_seq, NULL);
	if (ret < 0)
		return ret;

	if (msg_type != ODL_CLI_MSG_TEST_START)
		return -EPROTO;

	t_start = odl_time_ns();

	for (;;) {
		uint32_t type, seq;
		uint8_t src_id;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, &src_id);
		if (ret == -ETIMEDOUT)
			continue;
		if (ret < 0)
			return ret;

		if (type == ODL_CLI_MSG_TEST_STOP)
			break;

		if (type != ODL_CLI_MSG_PING)
			continue;

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PONG, seq, NULL, 0);
		if (ret < 0)
			return ret;

		bytes_reflected += sizeof(struct odl_cli_header) * 2;

		if ((odl_time_ns() - t_start) >= timeout_ns)
			break;
	}

	/* Receive client's RESULT */
	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &msg_type, &msg_seq, NULL);
	if (ret < 0)
		return ret;

	/* Send server's RESULT */
	{
		struct odl_cli_result result;
		uint64_t elapsed_ns = odl_time_ns() - t_start;

		memset(&result, 0, sizeof(result));
		result.bytes_transferred = bytes_reflected;
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
