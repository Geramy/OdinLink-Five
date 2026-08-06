/*
 * OdinLink — RDMA Tensor Sender
 *
 * Sends a tensor (e.g. an image frame) from Linux to a Mac via
 * Thunderbolt 5 using OdinLink's ibv verbs API.
 *
 * The Mac side runs OdinLinkRDMA kext + Metal viewer. The kext
 * allocates a DART-mapped buffer, shares it with Metal, and
 * notifies userspace when a frame arrives.
 *
 * This program:
 *   1. Opens the OdinLink verbs device
 *   2. Registers a memory region with the tensor data
 *   3. Creates a QP (which maps to an OdinLink stream)
 *   4. Posts RDMA writes to send the tensor
 *   5. Repeats for continuous streaming (like video frames)
 *
 * Build:
 *   gcc -o odl_tensor_send odl_tensor_send.c -lodl_tb5 -libverbs -lpthread
 *
 * Run:
 *   ODL_TB5_DEVICE=0 ./odl_tensor_send --width 1920 --height 1080 --fps 30
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#define DEFAULT_WIDTH  1920
#define_HEIGHT_DEFAULT 1080
#define FPS_DEFAULT    30
#define FRAME_FMT      4 /* RGBA8 = 4 bytes per pixel */

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

static void fill_test_frame(void *buf, int width, int height, int frame_num)
{
	uint32_t *pixels = buf;
	int x, y;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			uint8_t r, g, b, a;

			int phase = (frame_num * 3) & 0xFF;
			int checker = ((x / 64) + (y / 64)) & 1;

			if (checker) {
				r = (uint8_t)((x + phase) & 0xFF);
				g = (uint8_t)((y + phase) & 0xFF);
				b = (uint8_t)((x ^ y ^ phase) & 0xFF);
			} else {
				r = (uint8_t)((255 - x - phase) & 0xFF);
				g = (uint8_t)((255 - y - phase) & 0xFF);
				b = (uint8_t)((255 - (x ^ y) - phase) & 0xFF);
			}
			a = 0xFF;

			pixels[y * width + x] = (a << 24) | (b << 16) | (g << 8) | r;
		}
	}
}

int main(int argc, char **argv)
{
	int width = DEFAULT_WIDTH;
	int height = HEIGHT_DEFAULT;
	int fps = FPS_DEFAULT;
	int device_index = 0;
	int opt;

	struct option long_opts[] = {
		{"width",  required_argument, NULL, 'w'},
		{"height", required_argument, NULL, 'h'},
		{"fps",    required_argument, NULL, 'f'},
		{"device", required_argument, NULL, 'd'},
		{"help",   no_argument,       NULL, '?'},
		{NULL, 0, NULL, 0}
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
				"\n"
				"Sends RGBA test frames to a Mac via Thunderbolt RDMA.\n"
				"\n"
				"Options:\n"
				"  -w, --width   Frame width  (default %d)\n"
				"  -h, --height  Frame height (default %d)\n"
				"  -f, --fps     Target FPS   (default %d)\n"
				"  -d, --device  OdinLink device index (default %d)\n",
				argv[0], DEFAULT_WIDTH, HEIGHT_DEFAULT, FPS_DEFAULT, 0);
			return (opt == '?') ? 0 : 1;
		}
	}

	size_t frame_size = (size_t)width * height * FRAME_FMT;
	long frame_interval_ns = 1000000000L / fps;

	printf("OdinLink Tensor Sender\n");
	printf("  Frame: %dx%d RGBA8 = %zu bytes (%.1f MB)\n",
	       width, height, frame_size, (double)frame_size / (1 << 20));
	printf("  Target: %d FPS (interval %ld ns)\n",
	       fps, frame_interval_ns);

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	/* ── Open OdinLink verbs device ──────────────────────────────── */

	struct ibv_device **dev_list = ibv_get_device_list(NULL);
	if (!dev_list || !dev_list[0]) {
		fprintf(stderr, "No RDMA devices found. Is OdinLink loaded?\n");
		return 1;
	}

	struct ibv_device *ibdev = NULL;
	for (int i = 0; dev_list[i]; i++) {
		if (i == device_index) {
			ibdev = dev_list[i];
			break;
		}
	}
	if (!ibdev) {
		fprintf(stderr, "Device index %d not found\n", device_index);
		ibv_free_device_list(dev_list);
		return 1;
	}

	printf("Using device: %s\n", ibv_get_device_name(ibdev));

	struct ibv_context *ctx = ibv_open_device(ibdev);
	if (!ctx) {
		fprintf(stderr, "ibv_open_device failed: %s\n", strerror(errno));
		ibv_free_device_list(dev_list);
		return 1;
	}

	struct ibv_device_attr dev_attr = {};
	if (ibv_query_device(ctx, &dev_attr)) {
		fprintf(stderr, "ibv_query_device failed: %s\n", strerror(errno));
		goto err_close;
	}
	printf("Device: max_mr=%d max_qp=%d max_mr_size=%lu\n",
	       dev_attr.max_mr, dev_attr.max_qp, dev_attr.max_mr_size);

	/* ── Create PD, CQ, QP ──────────────────────────────────────── */

	struct ibv_pd *pd = ibv_alloc_pd(ctx);
	if (!pd) {
		fprintf(stderr, "ibv_alloc_pd failed: %s\n", strerror(errno));
		goto err_close;
	}

	struct ibv_cq *send_cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
	if (!send_cq) {
		fprintf(stderr, "ibv_create_cq (send) failed: %s\n", strerror(errno));
		goto err_dealloc_pd;
	}

	struct ibv_cq *recv_cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
	if (!recv_cq) {
		fprintf(stderr, "ibv_create_cq (recv) failed: %s\n", strerror(errno));
		goto err_destroy_send_cq;
	}

	struct ibv_qp_init_attr qp_init = {
		.qp_type = IBV_QPT_RC,
		.send_cq = send_cq,
		.recv_cq = recv_cq,
		.cap = {
			.max_send_wr  = 16,
			.max_recv_wr  = 16,
			.max_send_sge = 1,
			.max_recv_sge = 1,
		},
	};

	struct ibv_qp *qp = ibv_create_qp(pd, &qp_init);
	if (!qp) {
		fprintf(stderr, "ibv_create_qp failed: %s\n", strerror(errno));
		goto err_destroy_recv_cq;
	}

	printf("QP created: qp_num=%u, stream active\n", qp->qp_num);

	/* ── Bring QP to RTS (connected) state ───────────────────────── */
	{
		struct ibv_qp_attr attr = {};
		attr.qp_state   = IBV_QPS_INIT;
		attr.pkey_index = 0;
		attr.port_num   = 1;
		attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
				       IBV_ACCESS_REMOTE_READ |
				       IBV_ACCESS_LOCAL_WRITE;
		if (ibv_modify_qp(qp, &attr, IBV_QP_STATE |
					     IBV_QP_PKEY_INDEX |
					     IBV_QP_PORT |
					     IBV_QP_ACCESS_FLAGS)) {
			fprintf(stderr, "INIT: ibv_modify_qp failed: %s\n",
				strerror(errno));
			goto err_destroy_qp;
		}

		attr.qp_state = IBV_QPS_RTR;
		attr.path_mtu = IBV_MTU_4096;
		attr.dest_qp_num = 1;
		attr.rq_psn = 0;
		attr.max_dest_rd_atomic = 1;
		attr.min_rnr_timer = 12;
		if (ibv_modify_qp(qp, &attr, IBV_QP_STATE |
					     IBV_QP_PATH_MTU |
					     IBV_QP_DEST_QPN |
					     IBV_QP_RQ_PSN |
					     IBV_QP_MAX_DEST_RD_ATOMIC |
					     IBV_QP_MIN_RNR_TIMER)) {
			fprintf(stderr, "RTR: ibv_modify_qp failed: %s\n",
				strerror(errno));
			goto err_destroy_qp;
		}

		attr.qp_state = IBV_QPS_RTS;
		attr.sq_psn = 0;
		attr.timeout = 14;
		attr.retry_cnt = 7;
		attr.rnr_retry = 7;
		attr.max_rd_atomic = 1;
		if (ibv_modify_qp(qp, &attr, IBV_QP_STATE |
					     IBV_QP_SQ_PSN |
					     IBV_QP_TIMEOUT |
					     IBV_QP_RETRY_CNT |
					     IBV_QP_RNR_RETRY |
					     IBV_QP_MAX_QP_RD_ATOMIC)) {
			fprintf(stderr, "RTS: ibv_modify_qp failed: %s\n",
				strerror(errno));
			goto err_destroy_qp;
		}
	}

	printf("QP in RTS state — ready to send\n");

	/* ── Allocate frame buffer and register MR ───────────────────── */

	void *frame_buf = aligned_alloc(4096, frame_size);
	if (!frame_buf) {
		fprintf(stderr, "frame alloc failed (%zu bytes): %s\n",
			frame_size, strerror(errno));
		goto err_destroy_qp;
	}
	memset(frame_buf, 0, frame_size);

	struct ibv_mr *mr = ibv_reg_mr(pd, frame_buf, frame_size,
				       IBV_ACCESS_LOCAL_WRITE |
				       IBV_ACCESS_REMOTE_WRITE |
				       IBV_ACCESS_REMOTE_READ);
	if (!mr) {
		fprintf(stderr, "ibv_reg_mr failed: %s\n", strerror(errno));
		goto err_free_frame;
	}

	printf("MR registered: addr=%p len=%zu lkey=0x%x rkey=0x%x\n",
	       frame_buf, frame_size, mr->lkey, mr->rkey);

	/* ── Stream frames ──────────────────────────────────────────── */

	int frame_num = 0;
	int frames_sent = 0;
	int frames_failed = 0;
	struct timespec ts_start, ts_frame, ts_next;

	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	printf("Streaming... (Ctrl-C to stop)\n");

	while (g_running) {
		clock_gettime(CLOCK_MONOTONIC, &ts_frame);

		fill_test_frame(frame_buf, width, height, frame_num);

		struct ibv_sge sge = {
			.addr   = (uintptr_t)frame_buf,
			.length = frame_size,
			.lkey   = mr->lkey,
		};

		struct ibv_send_wr wr = {
			.wr_id      = frame_num,
			.sg_list    = &sge,
			.num_sge    = 1,
			.opcode     = IBV_WR_SEND,
			.send_flags = IBV_SEND_SIGNALED,
		};
		struct ibv_send_wr *bad_wr = NULL;

		int ret = ibv_post_send(qp, &wr, &bad_wr);
		if (ret) {
			fprintf(stderr, "ibv_post_send frame %d failed: %s\n",
				frame_num, strerror(ret));
			frames_failed++;
			if (frames_failed > 100) {
				fprintf(stderr, "Too many failures, stopping\n");
				break;
			}
		} else {
			frames_sent++;
		}

		if (frames_sent % 100 == 0) {
			struct timespec ts_now;
			clock_gettime(CLOCK_MONOTONIC, &ts_now);
			double elapsed = timespec_diff_ns(&ts_start, &ts_now) / 1e9;
			printf("  [%6.1fs] sent %d frames (%.1f FPS, %d failed)\n",
			       elapsed, frames_sent,
			       frames_sent / elapsed, frames_failed);
		}

		frame_num++;

		clock_gettime(CLOCK_MONOTONIC, &ts_next);
		long elapsed_ns = timespec_diff_ns(&ts_frame, &ts_next);
		long sleep_ns = frame_interval_ns - elapsed_ns;
		if (sleep_ns > 0) {
			struct timespec ts_sleep = {
				.tv_sec  = sleep_ns / 1000000000L,
				.tv_nsec = sleep_ns % 1000000000L,
			};
			nanosleep(&ts_sleep, NULL);
		}
	}

	{
		struct timespec ts_now;
		clock_gettime(CLOCK_MONOTONIC, &ts_now);
		double elapsed = timespec_diff_ns(&ts_start, &ts_now) / 1e9;
		printf("\nDone: %d frames sent, %d failed, %.1f FPS average\n",
		       frames_sent, frames_failed,
		       elapsed > 0 ? frames_sent / elapsed : 0);
	}

	ibv_dereg_mr(mr);
err_free_frame:
	free(frame_buf);
err_destroy_qp:
	ibv_destroy_qp(qp);
err_destroy_recv_cq:
	ibv_destroy_cq(recv_cq);
err_destroy_send_cq:
	ibv_destroy_cq(send_cq);
err_dealloc_pd:
	ibv_dealloc_pd(pd);
err_close:
	ibv_close_device(ctx);
	ibv_free_device_list(dev_list);
	return 0;
}
