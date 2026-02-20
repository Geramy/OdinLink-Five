/* SPDX-License-Identifier: MIT */
/*
 * OdinLink Thunderbolt 5 - Internal Kernel Header
 *
 * Shared across the source files that compose odl_tb5.ko:
 *   odl_tb5_service.c   - Thunderbolt service probe / remove
 *   odl_tb5_ring_dma.c  - NHI ring allocation, DMA frame pool, TX/RX workers
 *   odl_tb5_chardev.c   - Character device (stream ioctl interface)
 *   odl_tb5_proto.c     - OdinLink login/logout handshake protocol
 */
#ifndef ODL_TB5_CORE_H
#define ODL_TB5_CORE_H

#include <linux/module.h>
#include <linux/thunderbolt.h>
#include <linux/cdev.h>
#include <linux/dma-buf.h>
#include <linux/wait.h>
#include <linux/atomic.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>
#include <linux/list.h>
#include <linux/device.h>
#include <linux/idr.h>
#include <linux/hashtable.h>
#include <linux/kref.h>

#include "uapi/odl_tb5_uapi.h"

/* ── DMA control protocol (kernel-internal, stream 0) ───────────────── */

#define ODL_TB5_DMA_MAGIC	0x4F444C35
#define ODL_TB5_DMA_PING	1
#define ODL_TB5_DMA_PONG	2

struct odl_tb5_dma_hdr {
	__le32	magic;
	__le32	type;
	__le32	reserved[2];
};

/* ── Stream header (on-wire, 5 bytes at start of every DMA frame) ──── */

struct odl_tb5_stream_hdr {
	__u8   src_id;
	__u8   dst_id;
	__u8   flags;
	__le16 payload_len;
} __packed;

/* ── DMA frame pool (replaces old double-buffer scheme) ─────────────── */

#define ODL_TB5_FRAME_POOL_SIZE  256

struct odl_tb5_frame_slot {
	void			*virt;
	dma_addr_t		phys;
	struct ring_frame	frame;
	struct odl_tb5_tx_msg	*tx_msg;
	int			slot_idx;
	bool			in_use;
};

struct odl_tb5_frame_pool {
	struct odl_tb5_frame_slot *slots;
	unsigned long		*bitmap;
	spinlock_t		lock;
	int			size;
	int			free_count;
	wait_queue_head_t	avail_waitq;
};

/* ── Per-stream TX/RX queue entries ──────────────────────────────────── */

struct odl_tb5_tx_msg {
	struct list_head	list;
	u8			dst_id;
	void			*data;
	size_t			len;
	size_t			sent;
	int			frames_pending;
	bool			done;
	struct odl_tb5_stream	*stream;
};

struct odl_tb5_rx_msg {
	struct list_head	list;
	u8			src_id;
	u8			flags;
	void			*data;
	size_t			len;
};

/* ── Per-stream state ────────────────────────────────────────────────── */

struct odl_tb5_stream {
	u8			id;
	struct odl_tb5_device	*dev;
	struct odl_tb5_file_ctx	*owner;
	struct list_head	owner_list;

	struct list_head	tx_queue;
	spinlock_t		tx_lock;
	int			tx_queue_len;
	int			tx_queue_max;
	atomic_t		tx_completed;
	wait_queue_head_t	tx_waitq;

	struct list_head	rx_queue;
	spinlock_t		rx_lock;
	int			rx_queue_len;
	int			rx_queue_max;
	atomic_t		rx_complete;
	wait_queue_head_t	rx_waitq;

	struct kref		refcount;
	struct hlist_node	node;
};

/* ── Per-fd context (crash-safe auto-cleanup) ────────────────────────── */

struct odl_tb5_file_ctx {
	struct odl_tb5_device	*dev;
	struct list_head	streams;
	spinlock_t		lock;
};

/* ── DMA buffer (legacy double-buffer) ────────────────────────────────── */

struct odl_tb5_dma_buf {
	void		*virt;
	dma_addr_t	phys;
	size_t		size;
};

/* ── NHI ring context (shared TX or RX ring) ─────────────────────────── */

struct odl_tb5_ring_ctx {
	struct tb_ring		*ring;
	struct ring_frame	*frames;
	int			ring_size;
	bool			started;

	spinlock_t		lock;
	atomic_t		completed;
	atomic_t		submitted;
	wait_queue_head_t	waitq;

	/* Legacy double-buffer fields (kept for proto layer compat) */
	struct odl_tb5_dma_buf	bufs[ODL_TB5_NUM_BUFFERS];
	int			front;
	int			back;
	int			posted_buf;
	bool			frames_posted;
	bool			swapped_since_post;
};

/* ── Main device structure ───────────────────────────────────────────── */

struct odl_tb5_device {
	struct tb_service	*svc;
	struct tb_xdomain	*xd;
	int			local_tx_hopid;
	int			remote_tx_hopid;

	struct odl_tb5_ring_ctx	tx;
	struct odl_tb5_ring_ctx	rx;

	/* Login/logout handshake */
	struct delayed_work	login_work;
	struct work_struct	connect_work;
	struct work_struct	restart_work;
	int			login_retries;
	bool			login_sent;
	bool			login_received;
	int			stale_remote_tx_hopid;

	/* DMA verification (ping/pong) */
	struct work_struct	verify_work;
	struct work_struct	ctrl_reply_work;
	struct work_struct	rx_poll_work;
	wait_queue_head_t	verify_waitq;
	bool			pong_received;
	int			verify_rx_type;

	/* Connection state */
	enum odl_tb5_conn_state	state;
	struct mutex		state_lock;

	/* Character device */
	struct cdev		cdev;
	dev_t			devt;
	struct device		*dev;
	int			index;
	atomic_t		open_count;

	/* Stream management */
	DECLARE_HASHTABLE(streams, 8);
	struct ida		stream_ida;
	struct mutex		stream_lock;

	/* DMA frame pool */
	struct odl_tb5_frame_pool frame_pool;

	/* TX drain worker */
	struct work_struct	tx_drain_work;

	/* RX repost tracking */
	atomic_t		rx_posted;
	int			rx_target;

	struct list_head	list;
};

extern struct list_head odl_tb5_devices_list;
extern struct mutex     odl_tb5_devices_lock;
extern unsigned int     odl_ring_size;

/* ── Service lifecycle ───────────────────────────────────────────────── */

int  odl_tb5_service_init(void);
void odl_tb5_service_exit(void);

/* ── Ring allocation (NHI level) ─────────────────────────────────────── */

int  odl_tb5_rings_alloc(struct odl_tb5_device *dev);
void odl_tb5_rings_free(struct odl_tb5_device *dev);
int  odl_tb5_rings_start(struct odl_tb5_device *dev);
void odl_tb5_rings_stop(struct odl_tb5_device *dev);
void odl_tb5_rings_reset(struct odl_tb5_device *dev);

/* ── DMA frame pool ──────────────────────────────────────────────────── */

int  odl_tb5_frame_pool_alloc(struct odl_tb5_device *dev);
void odl_tb5_frame_pool_free(struct odl_tb5_device *dev);
struct odl_tb5_frame_slot *odl_tb5_frame_pool_get(struct odl_tb5_frame_pool *pool);
void odl_tb5_frame_pool_put(struct odl_tb5_frame_pool *pool,
			    struct odl_tb5_frame_slot *slot);

/* ── Legacy DMA buffer management (kept for proto layer) ─────────────── */

int  odl_tb5_dma_bufs_alloc(struct odl_tb5_device *dev);
void odl_tb5_dma_bufs_free(struct odl_tb5_device *dev);

/* ── Legacy submit (kept for proto layer direct ring access) ─────────── */

int  odl_tb5_submit_tx(struct odl_tb5_device *dev,
		       size_t offset, size_t len, bool ctrl);
int  odl_tb5_submit_rx(struct odl_tb5_device *dev,
		       size_t offset, size_t len);

int  odl_tb5_submit_tx_dmabuf(struct odl_tb5_device *dev,
			      int dmabuf_fd, loff_t offset, size_t len);
int  odl_tb5_submit_rx_dmabuf(struct odl_tb5_device *dev,
			      int dmabuf_fd, loff_t offset, size_t len);

/* ── Stream management ───────────────────────────────────────────────── */

struct odl_tb5_stream *odl_tb5_stream_create(struct odl_tb5_device *dev,
					     struct odl_tb5_file_ctx *owner,
					     u8 filter_id);
void odl_tb5_stream_destroy(struct odl_tb5_stream *stream);
void odl_tb5_stream_put(struct odl_tb5_stream *stream);
struct odl_tb5_stream *odl_tb5_stream_lookup(struct odl_tb5_device *dev,
					     u8 stream_id);

/* ── Stream TX/RX operations ─────────────────────────────────────────── */

int  odl_tb5_stream_send(struct odl_tb5_stream *stream,
			 u8 dst_id, const void __user *data, size_t len);
int  odl_tb5_stream_recv(struct odl_tb5_stream *stream,
			 void __user *buf, size_t buf_len,
			 u8 *src_id, u32 *actual_len);
int  odl_tb5_stream_wait_tx(struct odl_tb5_stream *stream, u32 timeout_ms);
int  odl_tb5_stream_wait_rx(struct odl_tb5_stream *stream, u32 timeout_ms);

/* ── TX drain worker ─────────────────────────────────────────────────── */

void odl_tb5_tx_drain_work_fn(struct work_struct *work);

/* ── RX poll worker (start_poll callback mechanism) ──────────────────── */

void odl_tb5_rx_poll_work_fn(struct work_struct *work);

/* ── Ring callbacks ──────────────────────────────────────────────────── */

void odl_tb5_tx_callback(struct tb_ring *ring,
			 struct ring_frame *frame, bool canceled);
void odl_tb5_rx_callback(struct tb_ring *ring,
			 struct ring_frame *frame, bool canceled);

struct odl_tb5_device *odl_tb5_rx_ring_to_dev(struct tb_ring *ring);

/* ── RX repost ───────────────────────────────────────────────────────── */

void odl_tb5_rx_repost(struct odl_tb5_device *dev);

/* ── Character device ────────────────────────────────────────────────── */

int  odl_tb5_chardev_init(void);
void odl_tb5_chardev_exit(void);
int  odl_tb5_chardev_create(struct odl_tb5_device *dev);
void odl_tb5_chardev_destroy(struct odl_tb5_device *dev);

/* ── Protocol handshake ──────────────────────────────────────────────── */

extern const uuid_t odl_tb5_proto_uuid;

int  odl_tb5_proto_register(void);
void odl_tb5_proto_unregister(void);

int  odl_tb5_proto_init(struct odl_tb5_device *dev);
void odl_tb5_proto_exit(struct odl_tb5_device *dev);
int  odl_tb5_proto_send_login(struct odl_tb5_device *dev);
int  odl_tb5_proto_send_logout(struct odl_tb5_device *dev);

#endif /* ODL_TB5_CORE_H */
