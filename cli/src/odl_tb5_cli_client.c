#include "odl_tb5_cli.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

static int send_test_request(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			     const struct odl_cli_params *params,
			     enum odl_cli_test_type test_type, uint32_t block_size)
{
	struct odl_cli_test_req req;
	char msg_buf[4096];
	uint32_t type, seq;
	int ret;

	memset(&req, 0, sizeof(req));
	req.test_type = test_type;
	req.block_size = block_size;
	req.iterations = params->iterations;
	req.duration_sec = params->duration_sec;
	req.num_streams = params->num_streams;
	req.bg_block_size = params->bg_block_size;
	req.flags = 0;
	if (params->bidir)
		req.flags |= ODL_TEST_FLAG_BIDIR;
	if (params->warmup_iters > 0)
		req.flags |= ODL_TEST_FLAG_WARMUP;

	ret = odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_TEST_REQ, 0,
			       &req.test_type,
			       sizeof(req) - sizeof(req.hdr));
	if (ret < 0)
		return ret;

	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &type, &seq, NULL);
	if (ret < 0)
		return ret;

	if (type != ODL_CLI_MSG_TEST_ACK)
		return -EPROTO;

	return 0;
}

static int run_single_test(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			   const struct odl_cli_params *params,
			   enum odl_cli_test_type test_type)
{
	int ret;

	switch (test_type) {
	case ODL_TEST_BANDWIDTH:
		ret = send_test_request(handle, sid, dst, params, test_type,
					params->block_sizes[0]);
		if (ret < 0)
			return ret;

		ret = odl_cli_bandwidth_client(handle, sid, dst, params);
		break;

	case ODL_TEST_LATENCY:
		printf("\n--- Latency Test (%u iterations) ---\n",
		       params->iterations);
		ret = send_test_request(handle, sid, dst, params, test_type,
					params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_latency_client(handle, sid, dst, params);
		break;

	case ODL_TEST_LATENCY_LOAD:
		printf("\n--- Latency Under Load Test ---\n");
		ret = send_test_request(handle, sid, dst, params, test_type,
					params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_latency_load_client(handle, sid, dst, params);
		break;

	case ODL_TEST_MIMO:
		printf("\n--- MIMO Test (%u streams) ---\n",
		       params->num_streams);
		ret = send_test_request(handle, sid, dst, params, test_type,
					params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_mimo_client(handle, sid, dst, params);
		break;

	case ODL_TEST_JITTER:
		printf("\n--- Jitter Test (%u iterations) ---\n",
		       params->iterations);
		ret = send_test_request(handle, sid, dst, params, test_type,
					params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_jitter_client(handle, sid, dst, params);
		break;

	default:
		fprintf(stderr, "Unknown test type: %d\n", test_type);
		return -EINVAL;
	}

	return ret;
}

int odl_cli_run_client(const struct odl_cli_params *params)
{
	odl_tb5_t handle = NULL;
	uint8_t sid;
	int ret;

	printf("OdinLink TB5 Test Client\n");
	printf("========================\n");
	printf("Opening device %d...\n", params->device_index);

	ret = odl_tb5_open(&handle, params->device_index);
	if (ret < 0) {
		fprintf(stderr, "Failed to open device %d: %s\n",
			params->device_index, strerror(-ret));
		return 1;
	}

	printf("Waiting for peer connection...\n");
	ret = odl_tb5_wait_peer(handle, 30000);
	if (ret < 0) {
		fprintf(stderr, "Peer connection timed out: %s\n", strerror(-ret));
		goto out;
	}

	struct odl_tb5_peer_info peer;
	odl_tb5_get_peer(handle, &peer);
	printf("Connected to peer: %s (%s)\n",
	       peer.device_name, peer.vendor_name);
	printf("Link speed: %u Gb/s (x%u lanes)\n\n",
	       peer.link_speed, peer.link_width);

	ret = odl_tb5_stream_open(handle, ODL_STREAM_CLI, &sid);
	if (ret < 0) {
		fprintf(stderr, "Failed to open stream: %s\n", strerror(-ret));
		goto out;
	}

	ret = odl_cli_send_hello(handle, sid, ODL_STREAM_TEST);
	if (ret < 0) {
		fprintf(stderr, "Failed to send HELLO: %s\n", strerror(-ret));
		goto out_stream;
	}

	ret = odl_cli_recv_hello(handle, sid);
	if (ret < 0) {
		fprintf(stderr, "Handshake failed: %s\n", strerror(-ret));
		goto out_stream;
	}

	printf("Handshake complete.\n");

	if (params->test_type == ODL_TEST_ALL) {
		static const enum odl_cli_test_type all_tests[] = {
			ODL_TEST_BANDWIDTH,
			ODL_TEST_LATENCY,
			ODL_TEST_JITTER,
			ODL_TEST_LATENCY_LOAD,
			ODL_TEST_MIMO,
		};

		for (int i = 0; i < (int)(sizeof(all_tests) / sizeof(all_tests[0])); i++) {
			ret = run_single_test(handle, sid, ODL_STREAM_TEST,
					      params, all_tests[i]);
			if (ret < 0) {
				fprintf(stderr, "Test failed: %s\n", strerror(-ret));
				break;
			}
		}
	} else {
		ret = run_single_test(handle, sid, ODL_STREAM_TEST,
				      params, params->test_type);
	}

	odl_cli_send_msg(handle, sid, ODL_STREAM_TEST,
			 ODL_CLI_MSG_DONE, 0, NULL, 0);

	if (ret == 0)
		printf("\nAll tests completed successfully.\n");

out_stream:
	odl_tb5_stream_close(handle, sid);
out:
	odl_tb5_close(handle);
	return (ret < 0) ? 1 : 0;
}
