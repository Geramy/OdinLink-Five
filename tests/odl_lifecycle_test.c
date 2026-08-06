/*
 * OdinLink — stream lifecycle / teardown conformance test
 *
 * Everything else in tests/ exercises steady-state traffic. Nothing exercises
 * *teardown*, which is why BUG 24 survived: odl_tb5_stream_destroy() never
 * wakes stream->rx_waitq, so a thread parked in a blocking STREAM_RECV is
 * never released and whoever joins it blocks forever.
 *
 * That matters far more for RCCL than for ggml-rpc. RCCL closes and reopens
 * communicators on every error and abort path, and its plugin shuts a worker
 * down by closing the stream and joining the thread — precisely the sequence
 * that hangs here.
 *
 * WHAT THIS DOES NOT TEST, DELIBERATELY
 * -------------------------------------
 * There is a second, worse defect in the same area: odl_tb5_stream_free()
 * kfree()s the stream directly, with no RCU grace period, while
 * odl_tb5_stream_lookup() walks the hash under rcu_read_lock() and
 * dereferences the object before its kref_get_unless_zero(). Removing with
 * hash_del_rcu() and then freeing immediately is a use-after-free.
 *
 * Reproducing that needs a lookup/destroy race, and the likely outcome is a
 * kernel oops. On this hardware an oops in the Thunderbolt path has already
 * wedged the USB4 router *and* the GPU's firmware domain, costing a full power
 * cycle. So that defect is fixed by inspection, not by reproduction. This test
 * covers only the benign symptom.
 *
 * Every wait here is bounded. A hang is reported as a failure, never inherited.
 *
 *   gcc -O2 -o odl_lifecycle_test odl_lifecycle_test.c -lpthread
 *   ./odl_lifecycle_test /dev/odl_tb5_0
 *
 * Exit: 0 all passed, 1 a case failed, 2 could not set up.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define ODL_TB5_IOCTL_MAGIC 'O'

struct odl_tb5_stream_req {
	uint8_t stream_id;
	uint8_t flags;
};

struct odl_tb5_stream_xfer {
	uint8_t  stream_id;
	uint8_t  dst_id;
	uint8_t  src_id;
	uint8_t  flags;
	uint64_t data;
	uint32_t len;
	uint32_t actual_len;
};

#define ODL_TB5_IOCTL_STREAM_OPEN  _IOWR(ODL_TB5_IOCTL_MAGIC, 0x20, struct odl_tb5_stream_req)
#define ODL_TB5_IOCTL_STREAM_CLOSE _IOW (ODL_TB5_IOCTL_MAGIC, 0x21, struct odl_tb5_stream_req)
#define ODL_TB5_IOCTL_STREAM_RECV  _IOWR(ODL_TB5_IOCTL_MAGIC, 0x23, struct odl_tb5_stream_xfer)

/* How long to let a receive stay blocked before calling it a hang. The close
 * that should release it happens at +1s, so this is a very generous margin. */
#define HANG_VERDICT_SEC 8

static const char *dev_path;
static int         failures;

struct recv_ctx {
	int      fd;
	uint8_t  stream_id;
	int      ret;         /* ioctl return */
	int      err;         /* errno if ret < 0 */
	bool     returned;
	double   elapsed;
};

static double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void *blocking_recv(void *arg)
{
	struct recv_ctx *c = arg;
	struct odl_tb5_stream_xfer x = {0};
	static char buf[4096];
	double t0 = now_sec();

	x.stream_id = c->stream_id;
	x.data      = (uint64_t)(uintptr_t)buf;
	x.len       = sizeof(buf);

	/* No data will ever arrive: nothing is sending on this stream. The only
	 * thing that can release this is teardown waking the queue. */
	c->ret      = ioctl(c->fd, ODL_TB5_IOCTL_STREAM_RECV, &x);
	c->err      = errno;
	c->elapsed  = now_sec() - t0;
	c->returned = true;
	return NULL;
}

/*
 * Case 1 — a blocked receive must be released when its stream is closed.
 *
 * This is the exact sequence the RCCL plugin performs on shutdown: set stop,
 * close the stream to break the worker out of its blocking receive, then join.
 */
static void case_close_releases_blocked_recv(uint8_t stream_id)
{
	struct odl_tb5_stream_req req = {0};
	struct recv_ctx ctx = {0};
	pthread_t th;
	struct timespec deadline;
	int fd, rc;

	printf("── case 1: STREAM_CLOSE must release a blocked STREAM_RECV\n");

	fd = open(dev_path, O_RDWR);   /* blocking: no O_NONBLOCK */
	if (fd < 0) {
		printf("   SKIP  cannot open %s: %s\n", dev_path, strerror(errno));
		failures++;
		return;
	}

	req.stream_id = stream_id;
	if (ioctl(fd, ODL_TB5_IOCTL_STREAM_OPEN, &req) < 0) {
		printf("   SKIP  STREAM_OPEN(%u) failed: %s\n", stream_id, strerror(errno));
		close(fd);
		failures++;
		return;
	}

	ctx.fd = fd;
	ctx.stream_id = stream_id;
	if (pthread_create(&th, NULL, blocking_recv, &ctx) != 0) {
		printf("   SKIP  pthread_create failed\n");
		close(fd);
		failures++;
		return;
	}

	/* Let the receive actually reach the wait queue before closing. */
	sleep(1);

	printf("   closing the stream while a receive is parked on it...\n");
	ioctl(fd, ODL_TB5_IOCTL_STREAM_CLOSE, &req);

	clock_gettime(CLOCK_REALTIME, &deadline);
	deadline.tv_sec += HANG_VERDICT_SEC;
	rc = pthread_timedjoin_np(th, NULL, &deadline);

	if (rc == 0 && ctx.returned) {
		printf("   PASS  receive returned after %.2fs (ret=%d errno=%s)\n",
		       ctx.elapsed, ctx.ret,
		       ctx.ret < 0 ? strerror(ctx.err) : "none");
		if (ctx.ret == 0)
			printf("         note: returned success with no data — a shutdown\n"
			       "         error such as -ESHUTDOWN would be clearer\n");
	} else {
		printf("   FAIL  receive still blocked %ds after close — BUG 24 confirmed.\n",
		       HANG_VERDICT_SEC);
		printf("         odl_tb5_stream_destroy() does not wake stream->rx_waitq;\n"
		       "         the only wake site is data arrival. Any caller that closes\n"
		       "         a stream and then joins its receive worker hangs here.\n");
		failures++;
		/* Leave the thread parked. Process exit tears it down — the wait is
		 * interruptible, so exit is clean. Do NOT join it again. */
	}

	close(fd);
	printf("\n");
}

/*
 * Case 2 — closing the fd is NOT a way to cancel a blocked receive.
 *
 * This documents a constraint rather than a defect, and it is the one that
 * dictates how a consumer must be written.
 *
 * A thread sitting in ioctl() holds a reference to the open file, so close()
 * on the last user-visible descriptor does not run release(). The stream is
 * therefore not destroyed, nothing wakes the waiter, and teardown does not
 * begin until the process exits. Measured: stream created at t+0, "destroying"
 * logged at t+9s on process exit, not at the close(fd) issued at t+1s.
 *
 * No driver change can alter this — it is fd lifetime semantics.
 *
 * The consequence for the RCCL net plugin: cancel a receive worker with an
 * explicit STREAM_CLOSE (case 1), never by closing the fd and joining. This
 * case passes when it confirms the receive stays blocked.
 */
static void case_fd_close_releases_blocked_recv(uint8_t stream_id)
{
	struct odl_tb5_stream_req req = {0};
	struct recv_ctx ctx = {0};
	pthread_t th;
	struct timespec deadline;
	int fd, rc;

	printf("── case 2: closing the fd does NOT cancel a blocked STREAM_RECV (by design)\n");

	fd = open(dev_path, O_RDWR);
	if (fd < 0) {
		printf("   SKIP  cannot open %s: %s\n", dev_path, strerror(errno));
		failures++;
		return;
	}

	req.stream_id = stream_id;
	if (ioctl(fd, ODL_TB5_IOCTL_STREAM_OPEN, &req) < 0) {
		printf("   SKIP  STREAM_OPEN(%u) failed: %s\n", stream_id, strerror(errno));
		close(fd);
		failures++;
		return;
	}

	ctx.fd = fd;
	ctx.stream_id = stream_id;
	if (pthread_create(&th, NULL, blocking_recv, &ctx) != 0) {
		printf("   SKIP  pthread_create failed\n");
		close(fd);
		failures++;
		return;
	}

	sleep(1);

	printf("   closing the fd while a receive is parked on it...\n");
	close(fd);

	clock_gettime(CLOCK_REALTIME, &deadline);
	deadline.tv_sec += HANG_VERDICT_SEC;
	rc = pthread_timedjoin_np(th, NULL, &deadline);

	if (rc != 0) {
		printf("   PASS  receive still blocked after %ds, as expected.\n",
		       HANG_VERDICT_SEC);
		printf("         close(fd) cannot run release() while a thread is inside\n"
		       "         an ioctl on that fd, so teardown does not start until the\n"
		       "         process exits. Cancel with STREAM_CLOSE, never fd close.\n");
	} else {
		printf("   NOTE  receive returned after %.2fs (ret=%d errno=%s).\n",
		       ctx.elapsed, ctx.ret,
		       ctx.ret < 0 ? strerror(ctx.err) : "none");
		printf("         Unexpected but not a failure — fd close released it, which\n"
		       "         means file lifetime behaves differently than measured here.\n");
	}

	printf("\n");
}

int main(int argc, char **argv)
{
	dev_path = argc > 1 ? argv[1] : "/dev/odl_tb5_0";

	printf("OdinLink stream lifecycle test — %s\n", dev_path);
	printf("Bounded: any receive still blocked after %ds is a failure, not a hang.\n\n",
	       HANG_VERDICT_SEC);

	if (access(dev_path, R_OK | W_OK) != 0) {
		fprintf(stderr, "cannot access %s: %s\n", dev_path, strerror(errno));
		fprintf(stderr, "the driver must be loaded and the link READY.\n");
		return 2;
	}

	/* Distinct stream IDs: case 1 may leave its stream in a wedged state. */
	case_close_releases_blocked_recv(41);
	case_fd_close_releases_blocked_recv(42);

	if (failures) {
		printf("RESULT: %d case(s) failed.\n", failures);
		printf("A teardown that cannot be interrupted will wedge every RCCL\n"
		       "error path, because RCCL tears communicators down far more\n"
		       "often than the RPC transport does.\n");
		/* _exit: threads may still be parked in the kernel. */
		fflush(stdout);
		_exit(1);
	}

	printf("RESULT: all cases passed.\n");
	fflush(stdout);
	_exit(0);
}
