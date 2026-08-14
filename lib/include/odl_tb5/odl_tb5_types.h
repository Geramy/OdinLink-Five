/*
 * OdinLink — Shared Types: Structs Both Kernel and Userspace Agree On
 *
 * Connection states (disconnected → handshake → connected → ready),
 * peer info (UUID, speed, vendor), completion counters, and buffer
 * sizes. These structs are passed across the kernel-userspace boundary
 * via ioctl and must match exactly on both sides.
 */
#ifndef ODL_TB5_TYPES_H
#define ODL_TB5_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* Connection states (mirrors kernel enum odl_tb5_conn_state) */
enum odl_tb5_conn_state {
	ODL_TB5_STATE_DISCONNECTED = 0,
	ODL_TB5_STATE_HANDSHAKE    = 1,
	ODL_TB5_STATE_CONNECTED    = 2,
	ODL_TB5_STATE_ERROR        = 3,
	ODL_TB5_STATE_READY        = 4,
};

static inline const char *odl_tb5_state_str(uint32_t state)
{
	switch (state) {
	case ODL_TB5_STATE_DISCONNECTED: return "disconnected";
	case ODL_TB5_STATE_HANDSHAKE:    return "handshake";
	case ODL_TB5_STATE_CONNECTED:    return "connected";
	case ODL_TB5_STATE_ERROR:        return "error";
	case ODL_TB5_STATE_READY:        return "ready";
	default:                         return "unknown";
	}
}

/* Peer information */
struct odl_tb5_peer_info {
	uint8_t  uuid[16];
	uint32_t link_speed;
	uint32_t link_width;
	uint32_t state;
	uint32_t reserved;
	char     vendor_name[64];
	char     device_name[64];
};

/* Completion status */
struct odl_tb5_completion {
	uint32_t tx_completed;
	uint32_t rx_completed;
	uint32_t tx_submitted;
	uint32_t rx_submitted;
};

/* Buffer info */
struct odl_tb5_buf_info {
	uint64_t tx_buf_size;
	uint64_t rx_buf_size;
	uint32_t tx_buf_count;
	uint32_t rx_buf_count;
};

#endif /* ODL_TB5_TYPES_H */
