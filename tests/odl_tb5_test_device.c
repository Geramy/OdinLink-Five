/*
 * OdinLink — Test: Kernel Device Interface
 *
 * Opens /dev/odl_tb5_N, checks that it responds to ioctls, swaps
 * buffers, and reports peer info. Verifies the kernel <-> userspace
 * contract works at the most basic level.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>

#include <odl_tb5/odl_tb5_ioctl.h>
#include <odl_tb5/odl_tb5_types.h>

static int test_count;
static int pass_count;
static int fail_count;

#define TEST(name) do { \
	test_count++; \
	printf("  [TEST] %s... ", name); \
} while (0)

#define PASS() do { pass_count++; printf("PASS\n"); } while (0)
#define FAIL(msg) do { fail_count++; printf("FAIL: %s\n", msg); } while (0)

int odl_tb5_test_device(void)
{
	int fd;
	int ret;

	printf("\n--- Device Tests ---\n");

	/* Test: Device exists and can be opened */
	TEST("Open /dev/odl_tb5_0");
	fd = open("/dev/odl_tb5_0", O_RDWR);
	if (fd >= 0) {
		PASS();
	} else {
		FAIL(strerror(errno));
		printf("  (Is the odl_tb5 module loaded? Try: sudo insmod driver/odl_tb5.ko)\n");
		return fail_count;
	}

	/* Test: GET_BUF_INFO ioctl */
	TEST("GET_BUF_INFO ioctl");
	{
		struct odl_tb5_buf_info info;
		ret = ioctl(fd, ODL_TB5_IOCTL_GET_BUF_INFO, &info);
		if (ret == 0) {
			printf("PASS (tx=%lu, rx=%lu, count=%u)\n",
			       (unsigned long)info.tx_buf_size,
			       (unsigned long)info.rx_buf_size,
			       info.tx_buf_count);
			pass_count++;
		} else {
			FAIL(strerror(errno));
		}
	}

	/* Test: POLL_COMPLETION ioctl */
	TEST("POLL_COMPLETION ioctl");
	{
		struct odl_tb5_completion comp;
		ret = ioctl(fd, ODL_TB5_IOCTL_POLL_COMPLETION, &comp);
		if (ret == 0) {
			printf("PASS (tx_done=%u, rx_done=%u)\n",
			       comp.tx_completed, comp.rx_completed);
			pass_count++;
		} else {
			FAIL(strerror(errno));
		}
	}

	/* Test: GET_PEER ioctl */
	TEST("GET_PEER ioctl");
	{
		struct odl_tb5_peer_info peer;
		ret = ioctl(fd, ODL_TB5_IOCTL_GET_PEER, &peer);
		if (ret == 0) {
			printf("PASS (state=%u, speed=%u Gb/s)\n",
			       peer.state, peer.link_speed);
			pass_count++;
		} else if (errno == ENOTCONN) {
			printf("PASS (no peer connected, expected)\n");
			pass_count++;
		} else {
			FAIL(strerror(errno));
		}
	}

	/* Test: SEND ioctl behavior */
	TEST("SEND ioctl");
	{
		struct odl_tb5_xfer_request req = { .offset = 0, .len = 64 };
		ret = ioctl(fd, ODL_TB5_IOCTL_SEND, &req);
		if (ret == 0) {
			printf("PASS (peer connected, send ok)\n");
			pass_count++;
		} else if (errno == ENOTCONN) {
			printf("PASS (no peer, ENOTCONN expected)\n");
			pass_count++;
		} else {
			FAIL(strerror(errno));
		}
	}

	/* Test: Invalid ioctl returns ENOTTY */
	TEST("Invalid ioctl returns ENOTTY");
	{
		ret = ioctl(fd, _IO('O', 0xFF), NULL);
		if (ret < 0 && errno == ENOTTY) {
			PASS();
		} else {
			FAIL("Expected ENOTTY");
		}
	}

	/* Test: Close device */
	TEST("Close device");
	ret = close(fd);
	if (ret == 0) {
		PASS();
	} else {
		FAIL(strerror(errno));
	}

	return fail_count;
}
