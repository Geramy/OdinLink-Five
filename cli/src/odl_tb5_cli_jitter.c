/*
 * OdinLink — CLI: Jitter Test
 *
 * Measures the variation in round-trip latency over time (jitter).
 * Sends periodic ping-pong messages and records the min/avg/max
 * and standard deviation of latency. Important for real-time
 * and isochronous workloads.
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>

/* Client (initiator) for jitter test. */
int odl_cli_jitter_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			   const struct odl_cli_params *params)
{
	struct odl_stats rtt_stats;
	struct odl_stats jitter_stats;
	char msg_buf[4096];
	uint64_t prev_rtt = 0;
	uint32_t total_iters;
	int ret;

	total_iters = params->iterations + params->warmup_iters;

	ret = odl_stats_init(&rtt_stats, params->iterations);
	if (ret < 0)
		return ret;

	ret = odl_stats_init(&jitter_stats, params->iterations);
	if (ret < 0) {
		odl_stats_free(&rtt_stats);
		return ret;
	}

	ret = odl_cli_send_msg(handle, sid, dst,
			       ODL_CLI_MSG_TEST_START, 0, NULL, 0);
	if (ret < 0)
		goto out;

	printf("  Running %u iterations (+ %u warmup)...\n",
	       params->iterations, params->warmup_iters);

	for (uint32_t i = 0; i < total_iters; i++) {
		bool measuring = (i >= params->warmup_iters);
		uint32_t type;

		uint64_t t_send = odl_time_ns();

		ret = odl_cli_send_msg(handle, sid, dst,
				       ODL_CLI_MSG_PING, i, NULL, 0);
		if (ret < 0)
			goto out;

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, NULL, NULL);
		if (ret < 0)
			goto out;

		uint64_t t_recv = odl_time_ns();
		uint64_t rtt = t_recv - t_send;

		if (measuring) {
			odl_stats_add(&rtt_stats, rtt);

			if (prev_rtt > 0) {
				uint64_t jitter = (rtt > prev_rtt) ?
					(rtt - prev_rtt) : (prev_rtt - rtt);
				odl_stats_add(&jitter_stats, jitter);
			}
			prev_rtt = rtt;
		} else {
			prev_rtt = rtt;
		}
	}

	ret = odl_cli_send_msg(handle, sid, dst,
			       ODL_CLI_MSG_TEST_STOP, 0, NULL, 0);

	odl_stats_finalize(&rtt_stats);
	odl_stats_finalize(&jitter_stats);

	odl_stats_print(&rtt_stats, "Jitter Test - RTT");
	odl_stats_print_histogram(&rtt_stats);

	printf("\n");
	odl_stats_print(&jitter_stats, "Jitter Test - Inter-packet Jitter");
	odl_stats_print_histogram(&jitter_stats);

	char buf1[64], buf2[64], buf3[64];
	printf("\n  Jitter Summary:\n");
	printf("    Average RTT:       %s\n",
	       odl_format_latency((uint64_t)rtt_stats.avg_ns, buf1, sizeof(buf1)));
	printf("    Average Jitter:    %s\n",
	       odl_format_latency((uint64_t)jitter_stats.avg_ns, buf2, sizeof(buf2)));
	printf("    Max Jitter:        %s\n",
	       odl_format_latency(jitter_stats.max_ns, buf3, sizeof(buf3)));
	printf("    Jitter/RTT ratio:  %.2f%%\n",
	       rtt_stats.avg_ns > 0 ?
	       (jitter_stats.avg_ns / rtt_stats.avg_ns * 100.0) : 0);

	if (params->output_file) {
		char path[256];
		snprintf(path, sizeof(path), "%s_rtt.csv", params->output_file);
		odl_stats_write_csv(&rtt_stats, path);
		snprintf(path, sizeof(path), "%s_jitter.csv", params->output_file);
		odl_stats_write_csv(&jitter_stats, path);
	}

	ret = 0;

out:
	odl_stats_free(&rtt_stats);
	odl_stats_free(&jitter_stats);
	return ret;
}

/* Server (responder) for jitter test. */
int odl_cli_jitter_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			   const struct odl_cli_test_req *req)
{
	char msg_buf[4096];
	uint32_t type, seq;
	int ret;

	(void)req;

	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &type, &seq, NULL);
	if (ret < 0 || type != ODL_CLI_MSG_TEST_START)
		return (ret < 0) ? ret : -EPROTO;

	for (;;) {
		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, NULL);
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
	}

	return 0;
}
