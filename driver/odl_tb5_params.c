// SPDX-License-Identifier: MIT
/*
 * OdinLink — Shared module parameters and globals
 *
 * These live in their own translation unit because the module entry point
 * is transport-specific (odl_tb5_service.c for NHI, odl_tb5_service_apple.c
 * for Apple Silicon) and only one of the two is ever linked in.  Defining
 * the parameters in both copies let them drift: odl_busy_poll_us was added
 * to the NHI copy only, which broke the link on the Apple path.
 */

#include "odl_tb5_core.h"

LIST_HEAD(odl_tb5_devices_list);
DEFINE_MUTEX(odl_tb5_devices_lock);

DEFINE_IDA(odl_tb5_ida);

const uuid_t odl_tb5_proto_uuid =
	UUID_INIT(0x4f444c4e, 0x4b54, 0x4235,
		  0x4f, 0x44, 0x49, 0x4e, 0x4c, 0x49, 0x4e, 0x4b);

unsigned int odl_ring_size = ODL_TB5_RING_SIZE_DEFAULT;
module_param(odl_ring_size, uint, 0444);
MODULE_PARM_DESC(odl_ring_size,
	"Ring entries per direction (power-of-2, default 4096 = 16 MB/batch)");

int odl_loopback_count = 0;
module_param_named(loopback, odl_loopback_count, int, 0444);
MODULE_PARM_DESC(loopback,
	"Create N software loopback devices (max 16, default 0; no hw needed)");

int odl_protocol_mode = 0;
module_param_named(protocol, odl_protocol_mode, int, 0444);
MODULE_PARM_DESC(protocol,
	"XDomain protocol mode: 0=OdinLink (0x4F4C, default), 1=Apple (0xFA57)");

bool odl_e2e = true;
module_param_named(e2e, odl_e2e, bool, 0444);
MODULE_PARM_DESC(e2e,
	"Enable end-to-end flow control (default=1). Set 0 for TB3 controllers "
	"that do not support RING_FLAG_E2E.");

/* upstream PR #21: bounded RX busy-poll before sleeping. The RX softirq
 * increments rx_complete on another CPU, so a spinning reader sees it within
 * cache-coherency latency and skips a ~10-15 us context-switch wake. */
unsigned int odl_busy_poll_us = 0;
module_param(odl_busy_poll_us, uint, 0644);
MODULE_PARM_DESC(odl_busy_poll_us,
	"Bounded RX busy-poll window in microseconds before sleeping (0 = off)");

/* Bind at most N XDomain services (0 = unlimited).  With two Thunderbolt
 * cables both peer services sit at route=2 (BUG 2) and the handshake cannot
 * complete; unbinding one afterwards runs the full teardown path, so bind
 * only one from the start instead. */
int odl_max_devices = 0;
module_param_named(max_devices, odl_max_devices, int, 0444);
MODULE_PARM_DESC(max_devices,
	"Bind at most N XDomain services (0 = unlimited; use 1 with two cables)");

bool odl_bind_any = true;
module_param_named(bind_any, odl_bind_any, bool, 0444);
MODULE_PARM_DESC(bind_any,
	"Also attach to Thunderbolt hosts that do not advertise OdinLink "
	"(default=1). Needed for a Mac sink that has no XDomain directory.");

bool odl_skip_login = false;
module_param_named(skip_login, odl_skip_login, bool, 0444);
MODULE_PARM_DESC(skip_login,
	"Skip XDomain login and DMA-ping; bring the data path up on hop 1. "
	"Use with bind_any against a Mac kext that cannot answer login.");
