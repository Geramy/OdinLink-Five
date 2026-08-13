/* SPDX-License-Identifier: MIT */
/*
 * Wire + userspace ABI shared by the Mac kext, the Mac client, and the
 * Linux sender. No Linux-kernel or IOKit types — C99 only.
 *
 * Linux → Mac is two-sided send/recv, not one-sided RDMA WRITE:
 *   Linux posts TX frames (4 KB OdinLink stream frames).
 *   The Mac kext posts RX descriptors at the same size.
 *   Hardware copies into the DART-mapped buffer. Userspace mmaps it.
 */
#ifndef ODINLINK_MAC_PROTO_H
#define ODINLINK_MAC_PROTO_H

#include <stdint.h>

#define ODL_MAC_MAGIC			0x4F444C4Du	/* 'ODLM' */
#define ODL_MAC_PROTO_VER		1

/* Stream id the Linux sender and Mac client agree on. */
#define ODL_MAC_STREAM_ID		20

/* Must match ODL_TB5_FRAME_SIZE on the Linux driver. */
#define ODL_MAC_SLOT_BYTES		4096
#define ODL_MAC_RX_SLOTS		256
#define ODL_MAC_RX_WINDOW_BYTES		(ODL_MAC_SLOT_BYTES * ODL_MAC_RX_SLOTS)

/* Default hop the Mac RX ring listens on. Linux login negotiates the
 * real hop; this is the fallback before that lands. */
#define ODL_MAC_DEFAULT_RX_HOP		1

/* OdinLink XDomain protocol — Mac kext watches for these on
 * IOThunderboltXDomainService. 20236 = 0x4F4C "OL", 64087 = 0xFA57. */
#define ODL_MAC_PROTOCOL_ID		20236
#define ODL_MAC_PROTOCOL_ID_APPLE	64087

/*
 * First message the Linux sender writes on ODL_MAC_STREAM_ID so the
 * Mac client can print the agreed geometry even if the kext has not
 * armed hardware yet (useful over Thunderbolt-IP / debug).
 */
struct odl_mac_hello {
	uint32_t magic;
	uint32_t version;
	uint32_t slot_bytes;
	uint32_t slot_count;
	uint32_t width;
	uint32_t height;
	uint32_t fps;
	uint32_t reserved;
};

#endif /* ODINLINK_MAC_PROTO_H */
