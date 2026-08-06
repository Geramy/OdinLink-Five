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
 *
 * Name suites on the command line to run a subset, e.g. "plugin" on a
 * platform with no kernel module to bind against.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test suite entry points */
extern int odl_tb5_test_device(void);
extern int odl_tb5_test_lib_api(void);
extern int odl_tb5_test_plugin(void);

static int suite_selected(int argc, char *argv[], const char *name)
{
	int i;

	if (argc < 2)
		return 1;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], name))
			return 1;
	}

	return 0;
}

int main(int argc, char *argv[])
{
	int total_failures = 0;

	printf("=================================\n");
	printf("OdinLink TB5 Test Suite\n");
	printf("=================================\n");

	if (suite_selected(argc, argv, "device"))
		total_failures += odl_tb5_test_device();
	if (suite_selected(argc, argv, "lib_api"))
		total_failures += odl_tb5_test_lib_api();
	if (suite_selected(argc, argv, "plugin"))
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
