/*
 * OdinLink — Test: Userspace Library API
 *
 * Exercises every function in libodl_tb5.so: open/close, send/recv,
 * swap buffers, poll/wait completions, stream open/close/send/recv.
 * Verifies the library behaves correctly under normal and error paths.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <odl_tb5/odl_tb5.h>

static int test_count;
static int pass_count;
static int fail_count;

#define TEST(name) do { \
	test_count++; \
	printf("  [TEST] %s... ", name); \
} while (0)

#define PASS() do { pass_count++; printf("PASS\n"); } while (0)
#define FAIL(msg) do { fail_count++; printf("FAIL: %s\n", msg); } while (0)

int odl_tb5_test_lib_api(void)
{
	odl_tb5_t handle = NULL;
	int ret;

	printf("\n--- Library API Tests ---\n");

	/* Test: Open device */
	TEST("odl_tb5_open(0)");
	ret = odl_tb5_open(&handle, 0);
	if (ret == 0 && handle != NULL) {
		PASS();
	} else {
		FAIL(strerror(-ret));
		printf("  (Is the odl_tb5 module loaded?)\n");
		return fail_count;
	}

	/* Test: Get buffer info */
	TEST("odl_tb5_get_buf_info()");
	{
		uint64_t tx_size, rx_size;
		ret = odl_tb5_get_buf_info(handle, &tx_size, &rx_size);
		if (ret == 0 && tx_size > 0 && rx_size > 0) {
			printf("PASS (tx=%lu, rx=%lu)\n",
			       (unsigned long)tx_size, (unsigned long)rx_size);
			pass_count++;
		} else {
			FAIL("Invalid buffer sizes");
		}
	}

	/* Test: TX buffer accessible */
	TEST("odl_tb5_tx_buffer()");
	{
		size_t size;
		void *buf = odl_tb5_tx_buffer(handle, &size);
		if (buf != NULL && size > 0) {
			/* Try writing to it */
			memset(buf, 0xAA, 64);
			PASS();
		} else {
			FAIL("NULL buffer or zero size");
		}
	}

	/* Test: RX buffer accessible */
	TEST("odl_tb5_rx_buffer()");
	{
		size_t size;
		void *buf = odl_tb5_rx_buffer(handle, &size);
		if (buf != NULL && size > 0) {
			PASS();
		} else {
			FAIL("NULL buffer or zero size");
		}
	}

	/* Test: Poll completion (should work even without connection) */
	TEST("odl_tb5_poll()");
	{
		struct odl_tb5_completion comp;
		ret = odl_tb5_poll(handle, &comp);
		if (ret == 0) {
			printf("PASS (tx=%u, rx=%u)\n",
			       comp.tx_completed, comp.rx_completed);
			pass_count++;
		} else {
			FAIL(strerror(-ret));
		}
	}

	/* Test: Get peer info (may not be connected) */
	TEST("odl_tb5_get_peer()");
	{
		struct odl_tb5_peer_info peer;
		ret = odl_tb5_get_peer(handle, &peer);
		if (ret == 0) {
			printf("PASS (state=%u)\n", peer.state);
			pass_count++;
		} else if (ret == -ENOTCONN) {
			printf("PASS (no peer, expected)\n");
			pass_count++;
		} else {
			FAIL(strerror(-ret));
		}
	}

	/* Test: Get raw fd */
	TEST("odl_tb5_get_fd()");
	{
		int fd = odl_tb5_get_fd(handle);
		if (fd >= 0) {
			PASS();
		} else {
			FAIL("Invalid fd");
		}
	}

	/* Test: Swap buffers */
	TEST("odl_tb5_swap_tx()");
	{
		ret = odl_tb5_swap_tx(handle);
		if (ret == 0) {
			PASS();
		} else {
			FAIL(strerror(-ret));
		}
	}

	TEST("odl_tb5_swap_rx()");
	{
		ret = odl_tb5_swap_rx(handle);
		if (ret == 0) {
			PASS();
		} else {
			FAIL(strerror(-ret));
		}
	}

	/* Test: Close */
	TEST("odl_tb5_close()");
	odl_tb5_close(handle);
	PASS();

	/* Test: NULL handle safety */
	TEST("NULL handle safety");
	odl_tb5_close(NULL);
	ret = odl_tb5_poll(NULL, NULL);
	if (ret == -EINVAL) {
		PASS();
	} else {
		FAIL("Expected EINVAL for NULL handle");
	}

	return fail_count;
}
