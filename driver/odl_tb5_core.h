/* SPDX-License-Identifier: MIT */
/*
 * OdinLink — Kernel Driver Internal Header
 *
 * The central wiring closet for the kernel module. Every .c file in the
 * driver shares the types and functions declared here.
 *
 * Files that use this header:
 *   odl_tb5_service.c   — Loading/unloading the driver, finding peer machines
 *   odl_tb5_ring_dma.c  — Setting up the DMA packet slots (like a conveyor belt
 *                         of fixed-size bins between two machines), sending and
 *                         receiving data through them
 *   odl_tb5_chardev.c   — The /dev/odl_tb5_N file that userspace programs open
 *                         to talk to the driver
 *   odl_tb5_proto.c     — The "hello/goodbye" handshake so both sides agree on
 *                         which DMA slots to use
 */
#ifndef ODL_TB5_CORE_H
#define ODL_TB5_CORE_H

#include <linux/module.h>
#if IS_ENABLED(CONFIG_USB4)
#include <linux/thunderbolt.h>
#endif
#include <linux/cdev.h>
#include <linux/dma-buf.h>
#include <linux/wait.h>
#include <linux/atomic.h>
#include <linux/mutex.h>
#include <linux/hrtimer.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>
#include <linux/list.h>
#include <linux/device.h>
#include <linux/idr.h>
#include <linux/hashtable.h>
#include <linux/kref.h>
#include <linux/version.h>

#if !IS_ENABLED(CONFIG_USB4)
struct tb_ring;
struct ring_frame;
typedef void (*ring_cb)(struct tb_ring *, struct ring_frame *, bool);
struct ring_frame {
	dma_addr_t buffer_phy;
	ring_cb callback;
	struct list_head list;
	u32 size:12;
	u32 flags:12;
	u32 eof:4;
	u32 sof:4;
};
#define RING_FLAG_FRAME		BIT(1)
#define RING_FLAG_E2E		BIT(2)
/* Values mirror enum ring_desc_flags in <linux/thunderbolt.h>; a backend
 * without the Thunderbolt subsystem must report descriptor errors using the
 * same bits so the shared RX callback can classify them. */
#define RING_DESC_CRC_ERROR		0x1
#define RING_DESC_BUFFER_OVERRUN	0x04
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 4, 0)
#define class_create_compat(name) class_create(THIS_MODULE, (name))
#else
#define class_create_compat(name) class_create((name))
#endif

/* hrtimer_setup was added in kernel 6.11; provide fallback for older kernels */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 11, 0)
static inline void hrtimer_setup(struct hrtimer *timer,
				 enum hrtimer_restart (*fn)(struct hrtimer *),
				 clockid_t clock, enum hrtimer_mode mode)
{
	hrtimer_init(timer, clock, mode);
	timer->function = fn;
}
#endif

#include "uapi/odl_tb5_uapi.h"
#include "odl_tb5_transport.h"
#include "odl_tb5_xd_proto.h"

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
	__u8   reserved;	/* pad: lands the payload 8-byte aligned */
	__le16 payload_len;
	__le16 frag_idx;	/* fragment index within the message, from 0 */
} __packed;

/* ── DMA frame pool (replaces old double-buffer scheme) ─────────────── */

/*
 * The frame pool is SHARED between TX and RX repost, so it must comfortably
 * back the peer's pre-posted receive ring or bidirectional traffic starves RX
 * and the NHI drops inbound frames in bursts.
 *
 * Sizing, for the ggml-rpc transport (the most demanding consumer here):
 *   256 KiB message / 4024 B payload  = 66 frames per message
 *   24 pre-posted receives            = 1587 frames of RX ring
 * The old 1024-frame pool was smaller than the RX ring alone, and the 64-frame
 * "reserve for RX" was less than a SINGLE message - so a busy sender routinely
 * left nothing for repost. Observed as 32-fragment burst losses under
 * bidirectional load.
 *
 * 4096 frames = 16 MiB per device; the reserve now guarantees the full RX ring
 * can always be reposted no matter how hard TX is pushing. Pure memory: no
 * extra work per frame, so the latency path is unaffected (and it removes
 * stalls waiting for free frames).
 */
#define ODL_TB5_FRAME_POOL_SIZE		4096
/*
 * The reserve is a floor TX may not dig below, NOT the size of the RX ring.
 * rx_target already claims pool/2 (2048) for posted receives, so setting the
 * reserve to 2048 as well left free_count == reserve exactly and
 * odl_tb5_stream_can_send() (free > reserve) was false forever - TX deadlocked
 * with a healthy link. Keep it a small guard so RX repost can always obtain a
 * few frames, while leaving TX ~1700 frames (~26 x 256 KiB messages) of room.
 */
#define ODL_TB5_TX_POOL_RESERVE		256  /* floor kept free for RX repost */
/*
 * Fallback poll interval for ring completions. NHI MSI-X fires, but the
 * ring_work it schedules can run before the descriptor write-back lands, so
 * this timer bounds how long a completion can sit unprocessed. It is the
 * dominant term in end-to-end latency: the median-minus-min spread of the
 * 22.4 us RTT is essentially this wait. 3 us keeps the tail close to the
 * 13.6 us best case at a modest timer cost.
 */
#define ODL_TB5_POLL_INTERVAL_NS	(3 * 1000)   /* 3 us */

/* ── SG batch buffer pool (throughput mode) ──────────────────────────── */

#define ODL_TB5_BATCH_BUF_SIZE		(256 * 1024)
#define ODL_TB5_BATCH_FRAMES		(ODL_TB5_BATCH_BUF_SIZE / ODL_TB5_FRAME_SIZE)
/*
 * One buffer is consumed per in-flight throughput-mode message, whatever its
 * size, and it is returned only once every one of its TX descriptors has run
 * its completion callback. So this count IS the hardware staging window: a
 * sender that posts N messages before reaping completions blocks on buffer
 * N+1 (5-second wait_event in odl_tb5_stream_send_throughput).
 *
 * ds4's bulk prefill round posts DS4_TP_RDMA_BULK_SLOTS=32 linked 128 KiB
 * sends before polling its CQ, so at 8 buffers messages 9..32 of EVERY round
 * queued behind that wait - 4x oversubscribed unconditionally, which turned
 * any transient peer-side RX slowdown into a hard multi-minute stall
 * ("throughput TX stalled waiting for batch buf (free=0 ...)"). Match the
 * window to that consumer's depth. 32 x 256 KiB = 8 MiB per device.
 */
#define ODL_TB5_BATCH_BUF_COUNT		32
#define ODL_TB5_THROUGHPUT_THRESH	65536	/* bytes: msg > 64KB → throughput */
#define ODL_TB5_MODE_HYSTERESIS		4	/* consecutive low polls to downshift */

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

/* ── SG batch buffer (contiguous DMA region for throughput mode) ──────── */

struct odl_tb5_batch_buf {
	void			*virt;
	dma_addr_t		phys;
	struct ring_frame	frames[ODL_TB5_BATCH_FRAMES];
	struct odl_tb5_tx_msg	*tx_msg;
	atomic_t		frames_pending;
	int			total_frames;
	struct list_head	list;
	bool			in_use;
};

struct odl_tb5_batch_pool {
	struct odl_tb5_batch_buf bufs[ODL_TB5_BATCH_BUF_COUNT];
	struct list_head	free_list;
	spinlock_t		lock;
	int			free_count;
	wait_queue_head_t	avail_waitq;
};

enum odl_tb5_tx_mode {
	ODL_TB5_TX_LATENCY    = 0,
	ODL_TB5_TX_THROUGHPUT = 1,
};

/* ── Per-stream TX/RX queue entries ──────────────────────────────────── */

struct odl_tb5_tx_msg {
	struct list_head	list;
	u8			dst_id;
	void			*data;
	size_t			len;
	size_t			sent;
	atomic_t		frames_pending;
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
	atomic_t		tx_in_flight;
	wait_queue_head_t	tx_waitq;

	struct list_head	rx_queue;
	spinlock_t		rx_lock;
	int			rx_queue_len;
	int			rx_queue_max;
	atomic_t		rx_complete;
	wait_queue_head_t	rx_waitq;

	/* RX message assembly — accumulates frames in callback context,
	 * enqueues only complete messages to rx_queue. */
	void			*rx_asm_buf;
	size_t			rx_asm_len;
	size_t			rx_asm_cap;
	u8			rx_asm_src_id;
	u16			rx_asm_next_frag;
	bool			rx_asm_bad;
	atomic_t		rx_frag_drops;

	/* Set once, before waking every waiter, when the stream is being torn
	 * down. Waiters must test this or they sleep on a stream that no longer
	 * exists — closing a stream is how consumers cancel a blocking receive
	 * (the RCCL plugin closes then joins its worker). */
	bool			dying;

	struct kref		refcount;
	struct hlist_node	node;

	/* The stream is looked up under RCU, so the object must outlive any
	 * reader still walking the hash chain. Freed via kfree_rcu(). */
	struct rcu_head		rcu;
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
	void			*ring_handle;
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
#if IS_ENABLED(CONFIG_USB4)
	struct tb_service	*svc;
	struct tb_xdomain	*xd;
#endif
	int			remote_tx_hopid;
	int			stale_remote_tx_hopid;
	/* BUG1 fix: authoritative record of the hop-ID actually allocated by
	 * tb_xdomain_alloc_in_hopid(), so every teardown path can release it
	 * regardless of connection state. */
	bool			in_hopid_valid;
	int			in_hopid;

	struct odl_tb5_ring_ctx	tx;
	struct odl_tb5_ring_ctx	rx;

	/* Recorded when the coherent buffers are allocated.  transport->
	 * dma_device() is not safe to call at free time on every backend
	 * (the NHI one derives it from tx.ring_handle, which teardown has
	 * usually already released). */
	struct device		*dma_dev;

	/* DMA verification (ping/pong) */
	struct work_struct	verify_work;
	struct work_struct	ctrl_reply_work;
	struct work_struct	connect_work;
	struct work_struct	restart_work;
	struct delayed_work	login_work;
	struct hrtimer		rx_poll_timer;
	wait_queue_head_t	verify_waitq;
	bool			pong_received;
	/* Set when we answer a peer PING. Unlike pong_received this is NOT
	 * cleared by verify_work, because the peer's ping can arrive BEFORE
	 * our own verify starts -- clearing it there loses the proof and the
	 * verify then times out with the link perfectly healthy. Reset only
	 * when a new connection begins. */
	bool			peer_ping_answered;
	int			verify_rx_type;
	int			login_retries;
	bool			login_sent;
	bool			login_received;

	/* Connection state */
	enum odl_tb5_conn_state	state;
	struct mutex		state_lock;
	wait_queue_head_t	state_waitq;

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

	/* SG batch buffer pool (throughput mode) */
	struct odl_tb5_batch_pool batch_pool;
	struct {
		enum odl_tb5_tx_mode	mode;
		unsigned int		consecutive_low;
		unsigned int		high_watermark;
		unsigned int		low_watermark;
	} tx_adaptive;

	const struct odl_tb5_transport_ops *transport;
	void *transport_priv;

	/* TX drain worker */
	struct work_struct	tx_drain_work;

	/* RX repost tracking */
	atomic_t		rx_posted;

	/*
	 * RX diagnostics. The receive callback used to validate only
	 * frame->size, so a frame the NHI had already flagged as bad was
	 * consumed as if it were good — and, worse, frames that never arrived
	 * left no trace anywhere. These count what was previously discarded in
	 * silence. They are reported alongside the fragment-gap warning, so a
	 * gap and its possible cause appear in the same log line.
	 */
	atomic_t		rx_err_crc;        /* NHI reported a CRC failure   */
	atomic_t		rx_err_overrun;    /* NHI reported buffer overrun  */
	atomic_t		rx_canceled;       /* ring teardown reclaimed it   */
	atomic_t		rx_err_short;      /* smaller than a stream header */
	atomic_t		rx_err_len;        /* payload_len exceeds the frame*/
	atomic_t		rx_frames_ok;      /* accepted, for a denominator  */

	/*
	 * RX repost starvation. odl_tb5_rx_repost() gives up silently when the
	 * shared frame pool has nothing left, leaving the receive ring short.
	 * Frames that then arrive with no posted buffer are dropped by the NHI
	 * itself — with no CRC or overrun flag, because nothing was wrong with
	 * them. That is invisible at every layer, and it is the leading
	 * candidate for the duplex fragment loss.
	 *
	 * This has bitten before: the pool sizing comment above records
	 * "32-fragment burst losses under bidirectional load" from exactly this
	 * cause. rx_posted_min is a low-water mark, initialised to INT_MAX.
	 */
	atomic_t		rx_repost_starved; /* repost gave up: pool empty   */
	atomic_t		rx_repost_short;   /* worst shortfall vs rx_target */
	atomic_t		rx_posted_min;     /* low-water mark of rx_posted  */
	int			rx_target;

	struct list_head	list;

    /* Cleanup synchronization — set to true when remove begins.
     * Used by callbacks for early exit during module unload,
     * preventing use-after-free after the device memory is released. */
	atomic_t			removing;
};

extern struct list_head odl_tb5_devices_list;
extern struct mutex     odl_tb5_devices_lock;
extern struct ida       odl_tb5_ida;
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
int  odl_tb5_frame_pool_get_batch(struct odl_tb5_frame_pool *pool,
				  struct odl_tb5_frame_slot **slots,
				  int requested);

/* ── SG batch buffer pool ────────────────────────────────────────────── */

int  odl_tb5_batch_pool_alloc(struct odl_tb5_device *dev);
void odl_tb5_batch_pool_free(struct odl_tb5_device *dev);
struct odl_tb5_batch_buf *odl_tb5_batch_pool_get(
				struct odl_tb5_batch_pool *pool);
void odl_tb5_batch_pool_put(struct odl_tb5_batch_pool *pool,
			    struct odl_tb5_batch_buf *buf);

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
void odl_tb5_streams_destroy_all(struct odl_tb5_device *dev);
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

/* Non-blocking availability checks (for poll/epoll support) */
bool odl_tb5_stream_can_send(struct odl_tb5_stream *stream);
bool odl_tb5_stream_can_recv(struct odl_tb5_stream *stream);

/* ── TX drain worker ─────────────────────────────────────────────────── */

void odl_tb5_tx_drain_work_fn(struct work_struct *work);

/* ── RX poll worker (start_poll callback mechanism) ──────────────────── */

enum hrtimer_restart odl_tb5_rx_poll_timer_fn(struct hrtimer *timer);

/* ── Ring callbacks ──────────────────────────────────────────────────── */

void odl_tb5_tx_callback(struct tb_ring *ring,
			 struct ring_frame *frame, bool canceled);
void odl_tb5_tx_batch_callback(struct tb_ring *ring,
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

#if IS_ENABLED(CONFIG_USB4)
struct tb_xdomain;
void odl_tb5_xd_header_init(struct odl_tb5_xd_header *hdr,
			     struct tb_xdomain *xd, u32 type,
			     size_t total_size);
#endif

/* ── Self-diagnosis (odl_tb5_debugfs.c) ─────────────────────────────── */

void odl_tb5_debugfs_init(void);
void odl_tb5_debugfs_exit(void);

/* ── Module parameters (defined in odl_tb5_params.c) ────────────────── */

extern int odl_loopback_count;
extern int odl_protocol_mode;
extern bool odl_e2e;
extern unsigned int odl_busy_poll_us;
extern int odl_max_devices;

static inline bool odl_tb5_is_loopback(struct odl_tb5_device *dev)
{
	return dev->transport && dev->transport->type == ODL_TB5_TRANSPORT_LOOPBACK;
}


int  odl_loopback_init(void);
void odl_loopback_exit(void);

int  odl_tb5_apple_init(void);
void odl_tb5_apple_exit(void);

extern struct platform_driver odl_tb5_apple_driver;

#endif /* ODL_TB5_CORE_H */
