/*
 * OdinLink — CLI: Wire Protocol for Client/Server Communication
 *
 * Defines how the CLI client and server talk to each other over
 * OdinLink streams: message types (HELLO, TEST_REQ, PING, DATA, etc.)
 * and serialization/deserialization of test parameters and results.
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>

static uint32_t g_sequence;

void odl_cli_print_open_error(int index, int err)
{
	fprintf(stderr, "Failed to open device %d (/dev/odl_tb5_%d): %s\n",
		index, index, strerror(-err));
	if (-err == ENOENT)
		fprintf(stderr,
			"  /dev/odl_tb5_* appears only after probe — both "
			"machines must load odl_tb5. A Thunderbolt host "
			"(or thunderbolt-net / en05) is not enough.\n"
			"  Check: dmesg | grep -E "
			"'odl_tb5: (probed device|still no peer)'\n");
	else if (-err == EACCES)
		fprintf(stderr,
			"  Permission denied. Install driver/71-odl-tb5.rules "
			"or: sudo chmod 660 /dev/odl_tb5_%d\n", index);
}

void odl_cli_print_wait_error(odl_tb5_t handle, int err)
{
	struct odl_tb5_peer_info peer;
	const char *st = "unknown";

	memset(&peer, 0, sizeof(peer));
	if (handle && odl_tb5_get_peer(handle, &peer) == 0)
		st = odl_tb5_state_str(peer.state);

	fprintf(stderr, "Peer connection failed: %s (state=%s)\n",
		strerror(-err), st);
	fprintf(stderr,
		"  The link is usable only in READY. Look for "
		"\"odl_tb5: entering READY state\" in dmesg "
		"(DMA-ping can take tens of seconds after probe).\n");
}

int odl_cli_send_msg(odl_tb5_t handle, uint8_t stream_id, uint8_t dst_id,
		     uint32_t type, uint32_t seq,
		     const void *payload, size_t payload_len)
{
	size_t total_len = sizeof(struct odl_cli_header) + payload_len;
	uint8_t *buf;
	struct odl_cli_header *hdr;
	int ret;

	buf = malloc(total_len);
	if (!buf)
		return -ENOMEM;

	hdr = (struct odl_cli_header *)buf;
	hdr->magic = ODL_CLI_MAGIC;
	hdr->type = type;
	hdr->sequence = seq;
	hdr->payload_len = (uint32_t)payload_len;
	hdr->timestamp_ns = odl_time_ns();

	if (payload && payload_len > 0)
		memcpy(buf + sizeof(*hdr), payload, payload_len);

	ret = odl_tb5_stream_send(handle, stream_id, dst_id,
				  buf, (uint32_t)total_len);
	free(buf);
	return ret;
}

int odl_cli_recv_msg(odl_tb5_t handle, uint8_t stream_id,
		     void *buf, size_t buf_size,
		     uint32_t *type, uint32_t *seq, uint8_t *src_id)
{
	uint8_t local_src_id;
	uint32_t actual_len;
	struct odl_cli_header *hdr;
	int ret;

	ret = odl_tb5_stream_wait_rx(handle, stream_id, 1000);
	if (ret < 0)
		return ret;

	ret = odl_tb5_stream_recv(handle, stream_id,
				  buf, (uint32_t)buf_size,
				  &local_src_id, &actual_len);
	if (ret < 0)
		return ret;

	if (actual_len < sizeof(struct odl_cli_header))
		return -EPROTO;

	hdr = (struct odl_cli_header *)buf;
	if (hdr->magic != ODL_CLI_MAGIC)
		return -EPROTO;

	if (type)
		*type = hdr->type;
	if (seq)
		*seq = hdr->sequence;
	if (src_id)
		*src_id = local_src_id;

	return (int)actual_len;
}

int odl_cli_send_hello(odl_tb5_t handle, uint8_t stream_id, uint8_t dst_id)
{
	struct odl_cli_hello hello;

	memset(&hello, 0, sizeof(hello));
	gethostname(hello.hostname, sizeof(hello.hostname) - 1);
	hello.version = 1;
	hello.capabilities = 0;

	return odl_cli_send_msg(handle, stream_id, dst_id,
				ODL_CLI_MSG_HELLO, g_sequence++,
				&hello.hostname, sizeof(hello) - sizeof(hello.hdr));
}

int odl_cli_recv_hello(odl_tb5_t handle, uint8_t stream_id)
{
	char buf[512];
	uint32_t type, seq;
	int ret;

	ret = odl_cli_recv_msg(handle, stream_id, buf, sizeof(buf),
			       &type, &seq, NULL);
	if (ret < 0)
		return ret;

	if (type != ODL_CLI_MSG_HELLO && type != ODL_CLI_MSG_HELLO_ACK)
		return -EPROTO;

	return 0;
}
