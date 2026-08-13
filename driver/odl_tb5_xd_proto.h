/* SPDX-License-Identifier: MIT */
/*
 * OdinLink — XDomain Protocol Definitions
 *
 * Shared between odl_tb5_proto.c (the protocol handler) and
 * odl_tb5_transport_nhi.c (the NHI backend). Apple-specific
 * protocol definitions are in odl_tb5_xd_proto_apple.h.
 */
#ifndef ODL_TB5_XD_PROTO_H
#define ODL_TB5_XD_PROTO_H

#include <linux/uuid.h>

#define ODL_TB5_MSG_LOGIN      1
#define ODL_TB5_MSG_LOGIN_RSP  2
#define ODL_TB5_MSG_LOGOUT     3

#define ODL_TB5_LOGIN_TIMEOUT  500

#define XD_HDR_SIZE_DW  3
#define XD_SN_MASK      0x18000000u

struct odl_tb5_xd_header {
	u32	route_hi;
	u32	route_lo;
	u32	length_sn;
	uuid_t	uuid;
	u32	type;
};

struct odl_tb5_login_msg {
	struct odl_tb5_xd_header xd_hdr;
	u32 proto_version;
	u32 transmit_path;
	u32 reserved[2];
};

struct odl_tb5_login_response {
	struct odl_tb5_xd_header xd_hdr;
	u32 status;
	u32 transmit_path;
	u32 reserved[2];
};

struct odl_tb5_logout_msg {
	struct odl_tb5_xd_header xd_hdr;
};

#endif
