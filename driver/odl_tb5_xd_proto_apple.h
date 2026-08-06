/* SPDX-License-Identifier: MIT */
/*
 * OdinLink — Apple XDomain Protocol Definitions
 *
 * Apple's ThunderboltRDMA uses protocol ID 0xFA57 and the key "rdma".
 * The XDomain packet format is the same as Intel's (route_hi, route_lo,
 * length_sn, uuid, type) but with Apple's protocol UUID.
 *
 * When the OdinLink module parameter `protocol=1` is set, the driver
 * uses these definitions instead of the Intel-style OdinLink protocol
 * so it can communicate with a macOS peer running AppleThunderboltRDMA.
 *
 * Key differences from Intel XDomain:
 *   - Apple uses protocol UUID 0xFA57 (vs OdinLink 0x4F4C)
 *   - Apple's login message may omit proto_version / transmit_path
 *     (short packets with just the XDomain header)
 *   - The route field uses the same format (64-bit, bit 63 = direction)
 *   - Apple's ThunderboltRDMA uses a different DMA descriptor
 *     control word layout (not yet confirmed)
 */
#ifndef ODL_TB5_XD_PROTO_APPLE_H
#define ODL_TB5_XD_PROTO_APPLE_H

#include <linux/uuid.h>
#include <linux/types.h>

/*
 * Apple ThunderboltRDMA protocol UUID.
 * From macOS kext: the "rdma" service key maps to this UUID.
 * The value 0xFA57 = "FAST" (Apple's sense of humor).
 */
#define APPLE_TB5_PROTO_UUID_VAL \
	UUID_INIT(0x00000000, 0x0000, 0x0000, \
		  0x00, 0x00, 0xfa, 0x57, 0x00, 0x00, 0x00, 0x00)

/*
 * Apple's actual RDMA protocol UUID from the kext is the standard
 * Thunderbolt XDomain UUID with the protocol ID set to 0xFA57.
 * The full UUID format is:
 *   00000000-0000-0000-0000-XXXXXXXXXXXX
 * where XXXXXXXXXXXX encodes the protocol ID.
 *
 * In practice, the kernel's tb_xdomain layer handles UUID matching.
 * We define the raw 16-byte UUID that matches what AppleThunderboltRDMA
 * advertises on the Thunderbolt bus.
 */
static const uuid_t apple_tb5_proto_uuid =
	UUID_INIT(0x00000000, 0x0000, 0x0000,
		  0x00, 0x00, 0xfa, 0x57, 0x00, 0x00, 0x00, 0x00);

/* Apple RDMA message types (same wire format as Intel, different UUID) */
#define APPLE_TB5_MSG_LOGIN		1
#define APPLE_TB5_MSG_LOGIN_RSP		2
#define APPLE_TB5_MSG_LOGOUT		3

/* Apple RDMA login timeout (ms) */
#define APPLE_TB5_LOGIN_TIMEOUT		500

/*
 * Apple XDomain header — identical wire format to Intel XDomain.
 * The difference is the UUID field: Apple uses 0xFA57, Intel uses
 * the protocol-specific UUID.
 *
 * On Apple Silicon, we don't have a tb_xdomain object from the
 * Thunderbolt bus. Instead, the route is either:
 *   1. Discovered via a future Apple NHI platform driver that
 *      creates xdomain objects
 *   2. Hard-coded to 0 (local loop) for development
 *   3. Read from the device tree property "apple,peer-route"
 */
struct apple_tb5_xd_header {
	__le32	route_hi;
	__le32	route_lo;
	__le32	length_sn;
	uuid_t	uuid;
	__le32	type;
};

/*
 * Apple login message.
 *
 * Apple's ThunderboltRDMA sends a minimal login that may only
 * contain the XDomain header (40 bytes). When present, the
 * payload fields follow the same layout as Intel:
 *   - proto_version: always 0 on Apple
 *   - transmit_path: the HopID Apple's TX ring uses to send to us
 *   - reserved: zero
 */
struct apple_tb5_login_msg {
	struct apple_tb5_xd_header xd_hdr;
	__le32	proto_version;
	__le32	transmit_path;
	__le32	reserved[2];
};

/*
 * Apple login response.
 *
 * Status 0 = success. transmit_path is our TX HopID that the
 * Apple peer should send to.
 */
struct apple_tb5_login_response {
	struct apple_tb5_xd_header xd_hdr;
	__le32	status;
	__le32	transmit_path;
	__le32	reserved[2];
};

/*
 * Apple logout message.
 *
 * Just the XDomain header with type=3. No payload.
 */
struct apple_tb5_logout_msg {
	struct apple_tb5_xd_header xd_hdr;
};

/*
 * Initialize an Apple XDomain header.
 *
 * On Apple Silicon, we don't have a tb_xdomain route. The route
 * is either:
 *   - 0 for a directly-connected peer (single-hop)
 *   - Read from device tree "apple,peer-route" property
 *   - Discovered by a future NHI platform driver
 *
 * The length_sn field encodes the packet length and a sequence
 * number. The format is:
 *   bits 27:  sequence number (0 or 1, alternates per packet)
 *   bits 0-26: total size in dwords minus 3 (XDomain header dwords)
 */
static inline void apple_tb5_xd_header_init(
	struct apple_tb5_xd_header *hdr,
	u64 route, u32 type, size_t total_size)
{
	hdr->route_hi  = cpu_to_le32(upper_32_bits(route));
	hdr->route_lo  = cpu_to_le32(lower_32_bits(route));
	hdr->length_sn = cpu_to_le32(total_size / 4 - 3);
	hdr->uuid      = apple_tb5_proto_uuid;
	hdr->type      = cpu_to_le32(type);
}

#endif /* ODL_TB5_XD_PROTO_APPLE_H */
