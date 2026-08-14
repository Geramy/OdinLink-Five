/*
 * OdinLink — CLI: Server Mode
 *
 * Listens for a client connection, then accepts and runs whatever test
 * the client requests (bandwidth, latency, jitter, MIMO). Runs until
 * the client disconnects.
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

int odl_cli_run_server(const struct odl_cli_params *params)
{
	odl_tb5_t handle = NULL;
	char msg_buf[4096];
	uint32_t type, seq;
	uint8_t src_id;
	uint8_t sid;
	int ret;

	printf("OdinLink TB5 Test Server\n");
	printf("========================\n");
	printf("Opening device %d...\n", params->device_index);

	ret = odl_tb5_open(&handle, params->device_index);
	if (ret < 0) {
		odl_cli_print_open_error(params->device_index, ret);
		return 1;
	}

	printf("Waiting for peer connection...\n");
	ret = odl_tb5_wait_peer(handle, 0);
	if (ret < 0) {
		odl_cli_print_wait_error(handle, ret);
		goto out;
	}

	struct odl_tb5_peer_info peer;
	odl_tb5_get_peer(handle, &peer);
	printf("Connected to peer: %s (%s)\n",
	       peer.device_name, peer.vendor_name);
	printf("Link speed: %u Gb/s (x%u lanes)\n\n",
	       peer.link_speed, peer.link_width);

	ret = odl_tb5_stream_open(handle, ODL_STREAM_TEST, &sid);
	if (ret < 0) {
		fprintf(stderr, "Failed to open stream: %s\n", strerror(-ret));
		goto out;
	}

	printf("Waiting for client handshake...\n");
	for (;;) {
		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, &src_id);
		if (ret == -ETIMEDOUT)
			continue;
		if (ret < 0) {
			fprintf(stderr, "Handshake failed: %s\n", strerror(-ret));
			goto out_stream;
		}
		if (type == ODL_CLI_MSG_HELLO || type == ODL_CLI_MSG_HELLO_ACK)
			break;
		fprintf(stderr, "Expected HELLO, got type 0x%x\n", type);
		ret = -EPROTO;
		goto out_stream;
	}

	ret = odl_cli_send_msg(handle, sid, src_id,
			       ODL_CLI_MSG_HELLO_ACK, 0, NULL, 0);
	if (ret < 0) {
		fprintf(stderr, "Failed to send HELLO_ACK: %s\n", strerror(-ret));
		goto out_stream;
	}

	printf("Handshake complete. Waiting for test commands...\n\n");

	for (;;) {
		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, &src_id);
		if (ret < 0) {
			if (ret == -ETIMEDOUT)
				continue;
			fprintf(stderr, "Receive error: %s\n", strerror(-ret));
			break;
		}

		if (type == ODL_CLI_MSG_DONE) {
			printf("Client finished. Shutting down.\n");
			break;
		}

		if (type != ODL_CLI_MSG_TEST_REQ) {
			if (params->verbose)
				fprintf(stderr, "Unexpected message type: 0x%x\n", type);
			continue;
		}

		struct odl_cli_test_req *req = (struct odl_cli_test_req *)msg_buf;

		ret = odl_cli_send_msg(handle, sid, src_id,
				       ODL_CLI_MSG_TEST_ACK, seq, NULL, 0);
		if (ret < 0) {
			fprintf(stderr, "Failed to send TEST_ACK\n");
			continue;
		}

		switch (req->test_type) {
		case ODL_TEST_BANDWIDTH:
			printf("[Server] Running bandwidth test (block=%u, dur=%us)...\n",
			       req->block_size, req->duration_sec);
			odl_cli_bandwidth_server(handle, sid, src_id, req);
			break;

		case ODL_TEST_LATENCY:
			printf("[Server] Running latency test (iters=%u)...\n",
			       req->iterations);
			odl_cli_latency_server(handle, sid, src_id, req);
			break;

		case ODL_TEST_LATENCY_LOAD:
			printf("[Server] Running latency-under-load test...\n");
			odl_cli_latency_load_server(handle, sid, src_id, req);
			break;

		case ODL_TEST_MIMO:
			printf("[Server] Running MIMO test (streams=%u)...\n",
			       req->num_streams);
			odl_cli_mimo_server(handle, sid, src_id, req);
			break;

		case ODL_TEST_JITTER:
			printf("[Server] Running jitter test (iters=%u)...\n",
			       req->iterations);
			odl_cli_jitter_server(handle, sid, src_id, req);
			break;

		default:
			fprintf(stderr, "[Server] Unknown test type: %u\n",
				req->test_type);
			break;
		}

		printf("[Server] Test complete.\n\n");
	}

out_stream:
	odl_tb5_stream_close(handle, sid);
out:
	odl_tb5_close(handle);
	return (ret < 0) ? 1 : 0;
}
