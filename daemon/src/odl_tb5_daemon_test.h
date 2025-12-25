/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2025-2026 OdinLink Project */
#ifndef ODL_TB5_DAEMON_TEST_H
#define ODL_TB5_DAEMON_TEST_H

#include <glib.h>
#include <stdbool.h>
#include <stdint.h>

#include "odl_tb5_daemon_sysinfo.h"

enum odl_daemon_test_state {
	ODL_DTEST_QUEUED = 0,
	ODL_DTEST_RUNNING,
	ODL_DTEST_COMPLETED,
	ODL_DTEST_FAILED,
	ODL_DTEST_CANCELLED,
};

struct odl_daemon_test_ctx {
	char     uuid[37];
	int      device_index;
	char     test_type[32];
	volatile int state;
	volatile int progress_pct;
	char     current_subtest[64];
	char    *result_json;
	char    *output_text;
};

enum odl_daemon_work_type {
	ODL_WORK_TEST,
	ODL_WORK_SYSINFO,
	ODL_WORK_SHUTDOWN,
};

struct odl_daemon_work_item {
	enum odl_daemon_work_type type;

	struct odl_daemon_test_ctx *test_ctx;
	struct odl_sysinfo sysinfo_result;

	GMutex   done_lock;
	GCond    done_cond;
	gboolean done;
	int      result;
	gboolean abandoned;
};

int  odl_daemon_test_init(void);
void odl_daemon_test_shutdown(void);

/* Start a test. Returns a UUID string (caller must NOT free; it's owned by the ctx). */
const char *odl_daemon_test_run(int device_index, const char *test_type);

/* Cancel a running test. Returns true if found and cancelled. */
bool odl_daemon_test_cancel(const char *test_id);

/* Query test status. Returns NULL if test_id not found. */
struct odl_daemon_test_ctx *odl_daemon_test_find(const char *test_id);

/* Request remote peer's system info via the device worker queue. */
int odl_daemon_test_request_peer_sysinfo(int device_index,
					  struct odl_sysinfo *out);

/* Dynamic device worker management (called by monitor on device add/remove) */
void odl_daemon_server_start_for_device(int index);
void odl_daemon_server_stop_for_device(int index);

#endif
