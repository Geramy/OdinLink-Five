/*
 * OdinLink — Test Suite: Smoke Tests for the Whole Stack
 *
 * Runs three test suites in sequence:
 *   1. device  — kernel ioctl interface
 *   2. lib_api — userspace library API
 *   3. plugin  — NCCL/RCCL plugin stubs
 *
 * Prerequisites: sudo insmod driver/odl_tb5.ko, device readable.
 * No test framework — plain C main(), returns number of failures.
 */
#include <stdio.h>
#include <stdlib.h>

/* Test suite entry points */
extern int odl_tb5_test_device(void);
extern int odl_tb5_test_lib_api(void);
extern int odl_tb5_test_plugin(void);

int main(int argc, char *argv[])
{
	int total_failures = 0;

	printf("=================================\n");
	printf("OdinLink TB5 Test Suite\n");
	printf("=================================\n");

	/* Run test suites */
	total_failures += odl_tb5_test_device();
	total_failures += odl_tb5_test_lib_api();
	total_failures += odl_tb5_test_plugin();

	/* Summary */
	printf("\n=================================\n");
	if (total_failures == 0) {
		printf("ALL TESTS PASSED\n");
	} else {
		printf("%d TEST(S) FAILED\n", total_failures);
	}
	printf("=================================\n");

	return total_failures > 0 ? 1 : 0;
}
