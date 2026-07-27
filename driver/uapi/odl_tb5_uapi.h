/* SPDX-License-Identifier: MIT */
/*
 * OdinLink — Userspace API: ioctl Commands and Structures
 *
 * This header defines the contract between userspace programs
 * (libodl_tb5.so, odl_tb5_cli, NCCL plugins) and the kernel driver.
 * Both sides include this file so they agree on message formats.
 *
 * Two API layers:
 *   1. Stream-based (new) — open channels by ID, send/receive messages,
 *      wait for completion. Supports multiple concurrent streams per
 *      device, like separate phone lines through one cable.
 *   2. Legacy double-buffer (old) — simple ping-pong buffers. Still
 *      works, kept for backward compatibility.
 *
 * Constants here define frame size (4 KB), protocol IDs for peer
 * discovery, stream header format, and all ioctl numbers.
 */
#ifndef ODL_TB5_UAPI_H
#define ODL_TB5_UAPI_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint8_t  __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef int32_t  __s32;
typedef int64_t  __s64;
#endif

#define ODL_TB5_DEVICE_NAME    "odl_tb5"
#define ODL_TB5_IOCTL_MAGIC    'O'
#define ODL_TB5_MAX_DEVICES    16

#define ODL_TB5_RING_SIZE_DEFAULT  4096
#define ODL_TB5_RING_SIZE_MIN      64
#define ODL_TB5_RING_SIZE_MAX      16384
#define ODL_TB5_FRAME_SIZE         4096
#define ODL_TB5_NUM_BUFFERS        2

#define ODL_TB5_PROTOCOL_KEY   "odinlink"
#define ODL_TB5_PROTOCOL_ID    0x4F4C
#define ODL_TB5_PROTOCOL_VER   1

/* Apple ThunderboltRDMA protocol — for cross-platform Mac ↔ Linux interop
 * Source: AppleThunderboltRDMA.kext Info.plist IOPropertyMatch Protocol ID
 * Changing ODL_TB5_PROTOCOL_ID to this value and registering the property
 * dir under Apple's key allows XDomain discovery to match macOS peers. */
#define ODL_TB5_PROTOCOL_ID_APPLE   64087   /* 0xFA57 */
#define ODL_TB5_PROTOCOL_KEY_APPLE  "rdma"

#define ODL_TB5_PDF_SOF_DATA   0x01
#define ODL_TB5_PDF_EOF_DATA   0x02
#define ODL_TB5_PDF_SOF_CTRL   0x01
#define ODL_TB5_PDF_EOF_CTRL   0x02

/* ── Stream header (5 bytes, prepended to every DMA frame) ──────────── */

#define ODL_TB5_STREAM_HDR_SIZE      5
/* A Thunderbolt ring frame carries framing/CRC overhead beyond the payload
 * the driver writes, so a frame of exactly ODL_TB5_FRAME_SIZE bytes does not
 * fit the equally-sized RX buffer and is silently dropped by the NHI. Only
 * the short tail fragment of a multi-frame message would then arrive, so any
 * message larger than one frame never completed reassembly. Reserve headroom
 * so header + payload stays strictly below the buffer size. */
#define ODL_TB5_FRAME_TAIL_RESERVE   64
#define ODL_TB5_STREAM_PAYLOAD_MAX   (ODL_TB5_FRAME_SIZE - ODL_TB5_STREAM_HDR_SIZE - ODL_TB5_FRAME_TAIL_RESERVE)

#define ODL_TB5_STREAM_ID_CTRL       0
#define ODL_TB5_STREAM_ID_MAX        255

#define ODL_TB5_SHDR_F_MSG_START     0x01
#define ODL_TB5_SHDR_F_MSG_END       0x02
#define ODL_TB5_SHDR_F_SINGLE        0x03
#define ODL_TB5_SHDR_F_DMABUF        0x04

/* ── Connection state ───────────────────────────────────────────────── */

enum odl_tb5_conn_state {
	ODL_TB5_STATE_DISCONNECTED = 0,
	ODL_TB5_STATE_HANDSHAKE    = 1,
	ODL_TB5_STATE_CONNECTED    = 2,
	ODL_TB5_STATE_ERROR        = 3,
	ODL_TB5_STATE_READY        = 4,
};

/* ── Peer info (unchanged) ──────────────────────────────────────────── */

struct odl_tb5_peer_info {
	__u8  uuid[16];
	__u32 link_speed;
	__u32 link_width;
	__u32 state;
	__u32 reserved;
	char  vendor_name[64];
	char  device_name[64];
};

/* ── Stream ioctl structures ────────────────────────────────────────── */

struct odl_tb5_stream_req {
	__u8  stream_id;
	__u8  flags;
};

struct odl_tb5_stream_xfer {
	__u8  stream_id;
	__u8  dst_id;
	__u8  src_id;
	__u8  flags;
	__u64 data;
	__u32 len;
	__u32 actual_len;
};

struct odl_tb5_stream_wait {
	__u8  stream_id;
	__u8  flags;
	__u16 reserved;
	__u32 timeout_ms;
};

struct odl_tb5_stream_dmabuf {
	__u8  stream_id;
	__u8  dst_id;
	__u16 reserved;
	__s32 dmabuf_fd;
	__u64 offset;
	__u64 len;
};

/* ── Stream ioctls ──────────────────────────────────────────────────── */

#define ODL_TB5_IOCTL_STREAM_OPEN      _IOWR(ODL_TB5_IOCTL_MAGIC, 0x20, struct odl_tb5_stream_req)
#define ODL_TB5_IOCTL_STREAM_CLOSE     _IOW (ODL_TB5_IOCTL_MAGIC, 0x21, struct odl_tb5_stream_req)
#define ODL_TB5_IOCTL_STREAM_SEND      _IOW (ODL_TB5_IOCTL_MAGIC, 0x22, struct odl_tb5_stream_xfer)
#define ODL_TB5_IOCTL_STREAM_RECV      _IOWR(ODL_TB5_IOCTL_MAGIC, 0x23, struct odl_tb5_stream_xfer)
#define ODL_TB5_IOCTL_STREAM_WAIT_TX   _IOW (ODL_TB5_IOCTL_MAGIC, 0x24, struct odl_tb5_stream_wait)
#define ODL_TB5_IOCTL_STREAM_WAIT_RX   _IOW (ODL_TB5_IOCTL_MAGIC, 0x25, struct odl_tb5_stream_wait)
#define ODL_TB5_IOCTL_STREAM_SEND_DMABUF _IOW(ODL_TB5_IOCTL_MAGIC, 0x26, struct odl_tb5_stream_dmabuf)
#define ODL_TB5_IOCTL_STREAM_RECV_DMABUF _IOW(ODL_TB5_IOCTL_MAGIC, 0x27, struct odl_tb5_stream_dmabuf)

#define ODL_TB5_IOCTL_GET_PEER         _IOR (ODL_TB5_IOCTL_MAGIC, 0x07, struct odl_tb5_peer_info)

/* ── Legacy ioctls (kept for backward compatibility) ────────────────── */

struct odl_tb5_xfer_request {
	__u64 offset;
	__u64 len;
	__u32 flags;
	__u32 reserved;
};

#define ODL_TB5_XFER_FLAG_CTRL  (1 << 0)

struct odl_tb5_ring_request {
	__s32 dmabuf_fd;
	__u32 reserved;
	__s64 offset;
	__u64 len;
};

struct odl_tb5_completion {
	__u32 tx_completed;
	__u32 rx_completed;
	__u32 tx_submitted;
	__u32 rx_submitted;
};

struct odl_tb5_buf_info {
	__u64 tx_buf_size;
	__u64 rx_buf_size;
	__u32 tx_buf_count;
	__u32 rx_buf_count;
};

#define ODL_TB5_IOCTL_SEND             _IOW(ODL_TB5_IOCTL_MAGIC, 0x01, struct odl_tb5_xfer_request)
#define ODL_TB5_IOCTL_RECV             _IOW(ODL_TB5_IOCTL_MAGIC, 0x02, struct odl_tb5_xfer_request)
#define ODL_TB5_IOCTL_SEND_DMABUF      _IOW(ODL_TB5_IOCTL_MAGIC, 0x03, struct odl_tb5_ring_request)
#define ODL_TB5_IOCTL_RECV_DMABUF      _IOW(ODL_TB5_IOCTL_MAGIC, 0x04, struct odl_tb5_ring_request)
#define ODL_TB5_IOCTL_POLL_COMPLETION  _IOR(ODL_TB5_IOCTL_MAGIC, 0x05, struct odl_tb5_completion)
#define ODL_TB5_IOCTL_WAIT_COMPLETION  _IOR(ODL_TB5_IOCTL_MAGIC, 0x06, struct odl_tb5_completion)
#define ODL_TB5_IOCTL_GET_BUF_INFO     _IOR(ODL_TB5_IOCTL_MAGIC, 0x08, struct odl_tb5_buf_info)
#define ODL_TB5_IOCTL_SWAP_TX_BUF      _IO(ODL_TB5_IOCTL_MAGIC, 0x09)
#define ODL_TB5_IOCTL_SWAP_RX_BUF      _IO(ODL_TB5_IOCTL_MAGIC, 0x0A)
#define ODL_TB5_IOCTL_WAIT_TX          _IOR(ODL_TB5_IOCTL_MAGIC, 0x0B, struct odl_tb5_completion)
#define ODL_TB5_IOCTL_WAIT_RX          _IOR(ODL_TB5_IOCTL_MAGIC, 0x0C, struct odl_tb5_completion)

/* Block until peer reaches READY state (timeout in ms, 0 = forever) */
#define ODL_TB5_IOCTL_WAIT_READY      _IOW(ODL_TB5_IOCTL_MAGIC, 0x0D, __u32)

#define ODL_TB5_MMAP_TX_BUF0   0x00000000ULL
#define ODL_TB5_MMAP_TX_BUF1   0x10000000ULL
#define ODL_TB5_MMAP_RX_BUF0   0x20000000ULL
#define ODL_TB5_MMAP_RX_BUF1   0x30000000ULL

#endif /* ODL_TB5_UAPI_H */
