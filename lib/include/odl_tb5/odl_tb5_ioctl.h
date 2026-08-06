/*
 * OdinLink Thunderbolt 5 - Userspace ioctl definitions
 */
#ifndef ODL_TB5_IOCTL_H
#define ODL_TB5_IOCTL_H

#include <sys/ioctl.h>
#include <stdint.h>

#define ODL_TB5_DEVICE_NAME    "odl_tb5"
#define ODL_TB5_IOCTL_MAGIC    'O'
#define ODL_TB5_MAX_DEVICES    16

/* Transfer request for internal double buffers */
struct odl_tb5_xfer_request {
	uint64_t offset;
	uint64_t len;
	uint32_t flags;
	uint32_t reserved;
};

#define ODL_TB5_XFER_FLAG_CTRL  (1 << 0)

/* Transfer request for external DMA-buf (RCCL / GPU) */
struct odl_tb5_ring_request {
	int32_t  dmabuf_fd;
	uint32_t reserved;
	int64_t  offset;
	uint64_t len;
};

/* Ioctl commands */
#define ODL_TB5_IOCTL_SEND             _IOW(ODL_TB5_IOCTL_MAGIC, 0x01, struct odl_tb5_xfer_request)
#define ODL_TB5_IOCTL_RECV             _IOW(ODL_TB5_IOCTL_MAGIC, 0x02, struct odl_tb5_xfer_request)
#define ODL_TB5_IOCTL_SEND_DMABUF      _IOW(ODL_TB5_IOCTL_MAGIC, 0x03, struct odl_tb5_ring_request)
#define ODL_TB5_IOCTL_RECV_DMABUF      _IOW(ODL_TB5_IOCTL_MAGIC, 0x04, struct odl_tb5_ring_request)
#define ODL_TB5_IOCTL_POLL_COMPLETION  _IOR(ODL_TB5_IOCTL_MAGIC, 0x05, struct odl_tb5_completion)
#define ODL_TB5_IOCTL_GET_PEER         _IOR(ODL_TB5_IOCTL_MAGIC, 0x07, struct odl_tb5_peer_info)
#define ODL_TB5_IOCTL_GET_BUF_INFO     _IOR(ODL_TB5_IOCTL_MAGIC, 0x08, struct odl_tb5_buf_info)
#define ODL_TB5_IOCTL_SWAP_TX_BUF      _IO(ODL_TB5_IOCTL_MAGIC, 0x09)
#define ODL_TB5_IOCTL_SWAP_RX_BUF      _IO(ODL_TB5_IOCTL_MAGIC, 0x0A)

/* Direction-specific completion waits (consume counter on return) */
#define ODL_TB5_IOCTL_WAIT_TX          _IOR(ODL_TB5_IOCTL_MAGIC, 0x0B, struct odl_tb5_completion)
#define ODL_TB5_IOCTL_WAIT_RX          _IOR(ODL_TB5_IOCTL_MAGIC, 0x0C, struct odl_tb5_completion)

/* Block until peer reaches READY state (timeout in ms, 0 = forever) */
#define ODL_TB5_IOCTL_WAIT_READY       _IOW(ODL_TB5_IOCTL_MAGIC, 0x0D, uint32_t)

/* mmap offsets for buffer selection (spaced 256 MB apart for large rings) */
#define ODL_TB5_MMAP_TX_BUF0   0x00000000ULL
#define ODL_TB5_MMAP_TX_BUF1   0x10000000ULL
#define ODL_TB5_MMAP_RX_BUF0   0x20000000ULL
#define ODL_TB5_MMAP_RX_BUF1   0x30000000ULL

/* ── Stream ioctl structures ───────────────────────────────────────── */

/* 8 bytes: src_id, dst_id, flags, reserved, payload_len, frag_idx.
 * frag_idx makes fragment loss DETECTABLE - reassembly used to be blind
 * concatenation gated only by MSG_START/MSG_END, so a single dropped frame
 * silently produced a short, corrupt message. The pad keeps the payload
 * 8-byte aligned, which also makes the reassembly memcpy cheaper. */
#define ODL_TB5_STREAM_HDR_SIZE      8
#define ODL_TB5_FRAME_SIZE           4096
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

struct odl_tb5_stream_req {
	uint8_t  stream_id;
	uint8_t  flags;
};

struct odl_tb5_stream_xfer {
	uint8_t  stream_id;
	uint8_t  dst_id;
	uint8_t  src_id;
	uint8_t  flags;
	uint64_t data;
	uint32_t len;
	uint32_t actual_len;
};

struct odl_tb5_stream_wait {
	uint8_t  stream_id;
	uint8_t  flags;
	uint16_t reserved;
	uint32_t timeout_ms;
};

struct odl_tb5_stream_dmabuf {
	uint8_t  stream_id;
	uint8_t  dst_id;
	uint16_t reserved;
	int32_t  dmabuf_fd;
	uint64_t offset;
	uint64_t len;
};

/* Stream ioctls */
#define ODL_TB5_IOCTL_STREAM_OPEN      _IOWR(ODL_TB5_IOCTL_MAGIC, 0x20, struct odl_tb5_stream_req)
#define ODL_TB5_IOCTL_STREAM_CLOSE     _IOW (ODL_TB5_IOCTL_MAGIC, 0x21, struct odl_tb5_stream_req)
#define ODL_TB5_IOCTL_STREAM_SEND      _IOW (ODL_TB5_IOCTL_MAGIC, 0x22, struct odl_tb5_stream_xfer)
#define ODL_TB5_IOCTL_STREAM_RECV      _IOWR(ODL_TB5_IOCTL_MAGIC, 0x23, struct odl_tb5_stream_xfer)
#define ODL_TB5_IOCTL_STREAM_WAIT_TX   _IOW (ODL_TB5_IOCTL_MAGIC, 0x24, struct odl_tb5_stream_wait)
#define ODL_TB5_IOCTL_STREAM_WAIT_RX   _IOW (ODL_TB5_IOCTL_MAGIC, 0x25, struct odl_tb5_stream_wait)
#define ODL_TB5_IOCTL_STREAM_SEND_DMABUF _IOW(ODL_TB5_IOCTL_MAGIC, 0x26, struct odl_tb5_stream_dmabuf)
#define ODL_TB5_IOCTL_STREAM_RECV_DMABUF _IOW(ODL_TB5_IOCTL_MAGIC, 0x27, struct odl_tb5_stream_dmabuf)

/* Forward declarations - full definitions in odl_tb5_types.h */
struct odl_tb5_completion;
struct odl_tb5_peer_info;
struct odl_tb5_buf_info;

#endif /* ODL_TB5_IOCTL_H */
