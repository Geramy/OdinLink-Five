/*
 * OdinLink — Linux → Mac tensor / frame sender
 *
 * Uses the OdinLink stream API (no libibverbs). The Mac kext posts
 * 4 KB RX descriptors; we send the same size stream frames after the
 * link reaches READY.
 *
 *   cmake --build build --target odl_tensor_send
 *   sudo insmod driver/odl_tb5.ko
 *   ./build/mac/odl_tensor_send --width 1920 --height 1080 --fps 30
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <odl_tb5/odl_tb5.h>
#include "odinlink_mac_proto.h"

#define DEFAULT_WIDTH  1920
#define HEIGHT_DEFAULT 1080
#define FPS_DEFAULT    30
#define FRAME_BPP      4

static volatile int g_running = 1;

static void signal_handler(int sig)
{
	(void)sig;
	g_running = 0;
}

static int64_t timespec_diff_ns(struct timespec *start, struct timespec *end)
{
	return (end->tv_sec - start->tv_sec) * 1000000000LL +
	       (end->tv_nsec - start->tv_nsec);
}

static void fill_slot(uint8_t *slot, uint32_t seq, int width, int height)
{
	uint32_t *pixels = (uint32_t *)slot;
	unsigned int n = ODL_MAC_SLOT_BYTES / 4;
	unsigned int i;
	uint8_t phase = (uint8_t)(seq * 3);

	(void)width;
	(void)height;
	for (i = 0; i < n; i++) {
		uint8_t r = (uint8_t)(i + phase);
		uint8_t g = (uint8_t)((i >> 2) + phase);
		uint8_t b = (uint8_t)((i ^ seq) & 0xFF);
		pixels[i] = 0xFF000000u | ((uint32_t)b << 16) |
			    ((uint32_t)g << 8) | r;
	}
}

int main(int argc, char *argv[])
{
	int width = DEFAULT_WIDTH;
	int height = HEIGHT_DEFAULT;
	int fps = FPS_DEFAULT;
	int device_index = 0;
	int opt;
	odl_tb5_t handle = NULL;
	uint8_t sid = 0;
	int ret;
	struct odl_mac_hello hello;
	uint8_t slot[ODL_MAC_SLOT_BYTES];
	unsigned int seq = 0;
	unsigned int sent = 0;
	unsigned int failed = 0;
	struct timespec ts_start, ts_now;
	long interval_ns;

	struct option long_opts[] = {
		{ "width",  required_argument, NULL, 'w' },
		{ "height", required_argument, NULL, 'h' },
		{ "fps",    required_argument, NULL, 'f' },
		{ "device", required_argument, NULL, 'd' },
		{ "help",   no_argument,       NULL, '?' },
		{ NULL, 0, NULL, 0 }
	};

	while ((opt = getopt_long(argc, argv, "w:h:f:d:?", long_opts, NULL)) != -1) {
		switch (opt) {
		case 'w': width = atoi(optarg); break;
		case 'h': height = atoi(optarg); break;
		case 'f': fps = atoi(optarg); break;
		case 'd': device_index = atoi(optarg); break;
		default:
			fprintf(stderr,
				"Usage: %s [--width W] [--height H] [--fps F] [--device D]\n"
				"Sends 4 KB OdinLink stream frames to a Mac sink "
				"(stream %u).\n",
				argv[0], ODL_MAC_STREAM_ID);
			return (opt == '?') ? 0 : 1;
		}
	}

	if (fps < 1)
		fps = 1;
	interval_ns = 1000000000L / fps;

	printf("OdinLink Linux → Mac sender\n");
	printf("  slot %u bytes, stream %u, %dx%d @ %d FPS\n",
	       ODL_MAC_SLOT_BYTES, ODL_MAC_STREAM_ID, width, height, fps);

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	ret = odl_tb5_open(&handle, device_index);
	if (ret < 0) {
		fprintf(stderr,
			"open /dev/odl_tb5_%d failed: %s\n"
			"  Both sides must load odl_tb5. On Linux: "
			"sudo insmod odl_tb5.ko\n"
			"  No /dev means the Mac is not advertising OdinLink "
			"yet (kext + cable).\n",
			device_index, strerror(-ret));
		return 1;
	}

	printf("Waiting for READY (DMA-ping after probe)...\n");
	ret = odl_tb5_wait_peer(handle, 60000);
	if (ret < 0) {
		struct odl_tb5_peer_info peer;

		memset(&peer, 0, sizeof(peer));
		odl_tb5_get_peer(handle, &peer);
		fprintf(stderr, "wait READY failed: %s (state=%s)\n",
			strerror(-ret), odl_tb5_state_str(peer.state));
		odl_tb5_close(handle);
		return 1;
	}

	{
		struct odl_tb5_peer_info peer;

		memset(&peer, 0, sizeof(peer));
		odl_tb5_get_peer(handle, &peer);
		printf("Peer READY: %s (%s) %u Gb/s x%u\n",
		       peer.device_name, peer.vendor_name,
		       peer.link_speed, peer.link_width);
	}

	ret = odl_tb5_stream_open(handle, ODL_MAC_STREAM_ID, &sid);
	if (ret < 0) {
		fprintf(stderr, "stream_open failed: %s\n", strerror(-ret));
		odl_tb5_close(handle);
		return 1;
	}

	memset(&hello, 0, sizeof(hello));
	hello.magic = ODL_MAC_MAGIC;
	hello.version = ODL_MAC_PROTO_VER;
	hello.slot_bytes = ODL_MAC_SLOT_BYTES;
	hello.slot_count = ODL_MAC_RX_SLOTS;
	hello.width = (uint32_t)width;
	hello.height = (uint32_t)height;
	hello.fps = (uint32_t)fps;

	ret = odl_tb5_stream_send(handle, sid, ODL_MAC_STREAM_ID,
				  &hello, sizeof(hello));
	if (ret < 0) {
		fprintf(stderr, "hello send failed: %s\n", strerror(-ret));
		odl_tb5_stream_close(handle, sid);
		odl_tb5_close(handle);
		return 1;
	}
	printf("HELLO sent. Streaming... (Ctrl-C to stop)\n");

	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	while (g_running) {
		struct timespec ts_frame, ts_after;
		long sleep_ns;

		clock_gettime(CLOCK_MONOTONIC, &ts_frame);
		fill_slot(slot, seq, width, height);
		ret = odl_tb5_stream_send(handle, sid, ODL_MAC_STREAM_ID,
					  slot, ODL_MAC_SLOT_BYTES);
		if (ret < 0) {
			failed++;
			if (failed > 100) {
				fprintf(stderr, "too many send failures\n");
				break;
			}
		} else {
			sent++;
		}
		seq++;

		if (sent > 0 && sent % 256 == 0) {
			double elapsed;

			clock_gettime(CLOCK_MONOTONIC, &ts_now);
			elapsed = timespec_diff_ns(&ts_start, &ts_now) / 1e9;
			printf("  [%6.1fs] sent %u slots (%.1f /s, %u failed)\n",
			       elapsed, sent,
			       elapsed > 0 ? sent / elapsed : 0, failed);
		}

		clock_gettime(CLOCK_MONOTONIC, &ts_after);
		sleep_ns = interval_ns - timespec_diff_ns(&ts_frame, &ts_after);
		if (sleep_ns > 0) {
			struct timespec sl = {
				.tv_sec = sleep_ns / 1000000000L,
				.tv_nsec = sleep_ns % 1000000000L,
			};
			nanosleep(&sl, NULL);
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &ts_now);
	{
		double elapsed = timespec_diff_ns(&ts_start, &ts_now) / 1e9;

		printf("Done: %u slots, %u failed, %.1f /s\n",
		       sent, failed,
		       elapsed > 0 ? sent / elapsed : 0);
	}

	odl_tb5_stream_close(handle, sid);
	odl_tb5_close(handle);
	return 0;
}
