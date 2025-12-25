/*
 * OdinLink Thunderbolt 5 - RCCL Plugin Tests
 *
 * Tests the RCCL net v7 plugin interface.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "net_v7.h"

static int test_count;
static int pass_count;
static int fail_count;

#define TEST(name) do { \
	test_count++; \
	printf("  [TEST] %s... ", name); \
} while (0)

#define PASS() do { pass_count++; printf("PASS\n"); } while (0)
#define FAIL(msg) do { fail_count++; printf("FAIL: %s\n", msg); } while (0)

/* Import plugin symbol directly (linked against librccl_net_odl_tb5.so) */
extern rcclNet_v7_t rcclNetPlugin_v7;

int odl_tb5_test_plugin(void)
{
	rcclNet_v7_t *plugin = &rcclNetPlugin_v7;
	int ret;

	printf("\n--- RCCL Plugin Tests ---\n");

	/* Test: Plugin name */
	TEST("Plugin name is ODL_TB5");
	if (plugin->name && strcmp(plugin->name, "ODL_TB5") == 0) {
		PASS();
	} else {
		FAIL(plugin->name ? plugin->name : "NULL");
	}

	/* Test: All function pointers are set */
	TEST("All function pointers non-NULL");
	if (plugin->init && plugin->devices && plugin->getProperties &&
	    plugin->listen && plugin->connect && plugin->accept &&
	    plugin->closeListen && plugin->isend && plugin->irecv &&
	    plugin->iflush && plugin->test && plugin->closeSend &&
	    plugin->closeRecv) {
		PASS();
	} else {
		FAIL("Some function pointers are NULL");
	}

	/* Test: Initialize plugin */
	TEST("plugin->init()");
	{
		rcclResult_t res = plugin->init(NULL);
		if (res == rcclSuccess) {
			PASS();
		} else {
			FAIL("init returned error");
		}
	}

	/* Test: Enumerate devices */
	TEST("plugin->devices()");
	{
		int ndev = 0;
		rcclResult_t res = plugin->devices(&ndev);
		if (res == rcclSuccess) {
			printf("PASS (found %d devices)\n", ndev);
			pass_count++;
		} else {
			FAIL("devices() returned error");
		}
	}

	/* Test: Get properties for device 0 (if available) */
	TEST("plugin->getProperties(0)");
	{
		int ndev = 0;
		plugin->devices(&ndev);

		if (ndev > 0) {
			rcclNetProperties_v7_t props;
			rcclResult_t res = plugin->getProperties(0, &props);
			if (res == rcclSuccess) {
				printf("PASS (name=%s, speed=%d, ptr=%d)\n",
				       props.name, props.speed, props.ptrSupport);
				pass_count++;
			} else {
				FAIL("getProperties returned error");
			}
		} else {
			printf("SKIP (no devices)\n");
			pass_count++;
		}
	}

	/* Test: Listen on device 0 */
	TEST("plugin->listen(0)");
	{
		int ndev = 0;
		plugin->devices(&ndev);

		if (ndev > 0) {
			void *listenComm = NULL;
			char handle[64] = {0};
			rcclResult_t res = plugin->listen(0, handle, &listenComm);
			if (res == rcclSuccess && listenComm != NULL) {
				PASS();
				plugin->closeListen(listenComm);
			} else {
				FAIL("listen returned error");
			}
		} else {
			printf("SKIP (no devices)\n");
			pass_count++;
		}
	}

	/* Test: Invalid device index */
	TEST("getProperties(-1) returns error");
	{
		rcclNetProperties_v7_t props;
		rcclResult_t res = plugin->getProperties(-1, &props);
		if (res == rcclInvalidArgument) {
			PASS();
		} else {
			FAIL("Expected rcclInvalidArgument");
		}
	}

	return fail_count;
}
