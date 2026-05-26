/*
 * OdinLink — Verbs: Helper API for Programs That Want to Find Us
 *
 * If you don't want to use the LD_PRELOAD or rdma-core plugin, you can
 * include this header directly. It provides helper functions to scan
 * /dev/odl_tb5_N, wrap the fd in an ibv_device struct, and open it
 * with standard ibv_open_device.
 *
 * Usage:
 *   #include <odl_tb5/odl_tb5_verbs_wrapper.h>
 *   ...
 *   struct ibv_device *dev = odl_find_tb5_device(0);
 *   struct ibv_context *ctx = ibv_open_device(dev);
 *   // ... then standard ibv_* API as usual ...
 */

#ifndef ODL_TB5_VERBS_WRAPPER_H
#define ODL_TB5_VERBS_WRAPPER_H

#include <infiniband/verbs.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Device Discovery ───────────────────────────────────────────────── */

/* Scan /dev/odl_tb5_N and return a verbs device for the given index.
 * Returns NULL if no device is found at that index. */
struct ibv_device *odl_find_tb5_device(int dev_index);

/* Return the number of available OdinLink-Five devices. */
int odl_num_tb5_devices(void);

/* Return whether the given device is an OdinLink-Five device. */
bool odl_is_tb5_device(struct ibv_device *dev);

/* ── Query ──────────────────────────────────────────────────────────── */

/* Get the dev_index for this device (the N in /dev/odl_tb5_N). */
int odl_tb5_device_index(struct ibv_device *dev);

/* ── Performance hints ──────────────────────────────────────────────── */

/* Enable async submission mode. When enabled, ibv_post_send returns
 * immediately and work is processed by a background thread. When
 * disabled (default), ibv_post_send blocks until complete.
 * Returns 0 on success, -1 on error. */
int odl_tb5_set_async_mode(struct ibv_context *ctx, bool async);

/* Set the debug level for the verbs provider (0-5).
 * 0=off, 1=errors, 2=warnings, 3=info, 4=verbose, 5=trace
 * Can also be set via the ODL_VERBS_DEBUG environment variable. */
void odl_tb5_verbs_set_debug(int level);

#ifdef __cplusplus
}
#endif

#endif /* ODL_TB5_VERBS_WRAPPER_H */
