#ifndef ODL_TB5_VERBS_H
#define ODL_TB5_VERBS_H

/*
 * OdinLink — Verbs Provider: Making OdinLink Look Like Any RDMA Card
 *
 * This is the magic that makes standard RDMA programs (NCCL, PyTorch,
 * ibv_rc_pingpong) work over a Thunderbolt cable without code changes.
 * It implements the ibv_* API (ibv_open_device, ibv_post_send, etc.)
 * on top of OdinLink's kernel driver.
 *
 * How it maps RDMA concepts to OdinLink:
 *   ibv_context     → an open /dev/odl_tb5_N + mmap'd buffers
 *   ibv_pd          → a permission domain (lightweight, just tracking)
 *   ibv_mr          → a pinned memory region (host RAM or GPU dmabuf)
 *   ibv_cq          → a completion queue backed by an eventfd
 *   ibv_qp          → a stream — messages go over one stream ID
 *
 * Async model: ibv_post_send drops work in a queue → a worker thread
 * picks it up → calls odl_tb5_stream_send → posts result to CQ.
 * This way the caller never blocks even though the kernel ioctl may.
 */

#include <infiniband/verbs.h>

#include <odl_tb5/odl_tb5.h>

#include <stdatomic.h>
#include <pthread.h>
#include <stdbool.h>
#include <sys/eventfd.h>
#include <sys/types.h>
#include <unistd.h>

#include "odl_tb5_verbs_debug.h"

/* ── Constants not in public header ─────────────────────────────────── */

#ifndef IBV_SPEED_SDR
#define IBV_SPEED_SDR      1
#define IBV_SPEED_DDR      2
#define IBV_SPEED_QDR      3
#define IBV_SPEED_FDR10   4
#define IBV_SPEED_FDR     5
#define IBV_SPEED_EDR     6
#define IBV_SPEED_HDR     7
#define IBV_SPEED_NDR     8
#define IBV_SPEED_EDR_EX  9
#define IBV_SPEED_NDR_EX 10
#endif

#ifndef IBV_WIDTH_1X
#define IBV_WIDTH_1X      1
#define IBV_WIDTH_2X      2
#define IBV_WIDTH_4X      4
#define IBV_WIDTH_8X      8
#define IBV_WIDTH_12X    12
#endif

/* ── Constants ──────────────────────────────────────────────────────── */

#define ODL_VERBS_MAX_DEVICES            16
#define ODL_VERBS_MAX_PDS               256
#define ODL_VERBS_MAX_MRS               512
#define ODL_VERBS_MAX_QPS               256
#define ODL_VERBS_MAX_CQS               128
#define ODL_VERBS_RQ_DEPTH              256
#define ODL_VERBS_COMP_CHANNEL_BACKLOG   64
#define ODL_VERBS_SQ_DEPTH              64

/* ── Forward declarations ───────────────────────────────────────────── */

struct odl_verbs_context;
struct odl_verbs_pd;
struct odl_verbs_mr;
struct odl_verbs_cq;
struct odl_verbs_qp;

/* ── Device ─────────────────────────────────────────────────────────── */

struct odl_verbs_device {
    struct ibv_device         base;
    int                       dev_index;
    char                      dev_path[64];
    char                      dev_name[32];
    bool                      is_open;
    struct odl_verbs_context *active_ctx;
};

/* ── Protection Domain ──────────────────────────────────────────────── */

struct odl_verbs_pd {
    struct ibv_pd             base;
    struct odl_verbs_context *ctx;
    uint32_t                  handle;
};

/* ── Memory Region ──────────────────────────────────────────────────── */

struct odl_verbs_mr {
    struct ibv_mr             base;
    int                       mr_type; /* 0=host, 1=dmabuf */
    int                       access_flags;
    /* Host memory */
    void                     *host_addr;
    size_t                    host_length;
    /* DMA-buf */
    int                       dmabuf_fd;
    uint64_t                  dmabuf_offset;
    uint64_t                  iova;
};

/* ── Completion Queue ───────────────────────────────────────────────── */

struct odl_verbs_cq {
    struct ibv_cq             base;
    struct odl_verbs_context *ctx;
    pthread_mutex_t           lock;
    uint32_t                  cq_handle;

    /* Completion ring buffer */
    struct ibv_wc             ring[ODL_VERBS_COMP_CHANNEL_BACKLOG];
    int                       head;
    int                       tail;

    /* Eventfd for async notification */
    int                       eventfd_fd;
    bool                      armed;

    /* QP whose receive queue feeds this CQ. ibv_poll_cq() must drive receive
     * progress: the QP worker also performs sends and can sit in its TX
     * readiness poll, during which nothing would drain RX and both peers
     * stall waiting on each other. */
    struct odl_verbs_qp      *rx_qp;
};

/* ── Queue Pair ─────────────────────────────────────────────────────── */

struct odl_verbs_qp {
    struct ibv_qp             base;
    struct odl_verbs_context *ctx;
    struct odl_verbs_pd      *pd;
    struct odl_verbs_cq      *send_cq;
    struct odl_verbs_cq      *recv_cq;
    uint8_t                   stream_id;
    /* BUG15: remote stream to address sends at, taken from
     * ibv_modify_qp(IBV_QP_DEST_QPN) at the RTR transition. Without this the
     * worker sent everything to dst_id 0 and nothing reached the peer. */
    uint8_t                   dest_qp;

    /* Work submission queue (async via worker thread) */
    pthread_mutex_t           sq_lock;
    /* BUG14: callers pass stack-allocated ibv_send_wr/ibv_sge and expect
     * post_send to return immediately, so the worker must never dereference
     * the caller's pointers. Store copies of the fields we need. */
    uint64_t                  sq_wr_id[ODL_VERBS_SQ_DEPTH];
    uint64_t                  sq_addr[ODL_VERBS_SQ_DEPTH];
    uint32_t                  sq_len[ODL_VERBS_SQ_DEPTH];
    uint32_t                  sq_lkey[ODL_VERBS_SQ_DEPTH];
    int                       sq_num_sge[ODL_VERBS_SQ_DEPTH];
    /* ibv_post_send() is defined to consume the payload before returning, so
     * callers reuse their send buffer immediately. odl_tb5_stream_send() only
     * QUEUES the data, so by the time the worker DMAs it the caller has
     * usually overwritten it -- silent corruption. Take a private copy at post
     * time and transmit from that. */
    void                     *sq_bounce[ODL_VERBS_SQ_DEPTH];
    int                       sq_head;
    int                       sq_tail;
    int                       sq_count;

    /* Worker thread */
    pthread_t                 worker;
    bool                      worker_running;

    /* Receive queue: buffers posted by the app, awaiting inbound data.
     * ibv_post_recv() must NOT block or touch the wire -- it only enqueues.
     * The worker thread drains the stream into these buffers and posts the
     * completions. Callers pass stack-allocated ibv_recv_wr/ibv_sge, so we
     * store COPIES, never the caller's pointers. */
    pthread_mutex_t           rq_lock;
    /* Serialises odl_rq_drain(): it is now called from BOTH the QP worker and
     * the application's ibv_poll_cq() thread, and it must release rq_lock
     * around stream_recv(). Without exclusion two drainers interleave their
     * receives and deliver messages out of order, corrupting the stream. */
    pthread_mutex_t           drain_lock;
    uint64_t                  rq_wr_id[ODL_VERBS_RQ_DEPTH];
    uint64_t                  rq_addr[ODL_VERBS_RQ_DEPTH];
    uint32_t                  rq_len[ODL_VERBS_RQ_DEPTH];
    int                       rq_head;
    int                       rq_tail;
    int                       rq_count;

    /* Async tracking */
    atomic_int                pending_sends;
    atomic_int                pending_recvs;
};

/* ── Context (per-device-open state) ────────────────────────────────── */

struct odl_verbs_context {
    struct ibv_context        base;
    struct odl_verbs_device  *dev;
    odl_tb5_t                 handle;
    int                       refcount;

    /* Object tracking */
    pthread_mutex_t           pd_lock;
    struct odl_verbs_pd      *pds[ODL_VERBS_MAX_PDS];
    int                       npds;

    pthread_mutex_t           mr_lock;
    struct odl_verbs_mr      *mrs[ODL_VERBS_MAX_MRS];
    int                       nmrs;

    pthread_mutex_t           cq_lock;
    struct odl_verbs_cq      *cqs[ODL_VERBS_MAX_CQS];
    int                       ncqs;

    pthread_mutex_t           qp_lock;
    struct odl_verbs_qp      *qps[ODL_VERBS_MAX_QPS];
    int                       nqps;
};

/* ── Inline Conversions ─────────────────────────────────────────────── */

static inline struct odl_verbs_context *
odl_ctx_from_ibv(struct ibv_context *ctx)
{
    return (struct odl_verbs_context *)ctx;
}

static inline struct odl_verbs_device *
odl_dev_from_ibv(const struct ibv_device *d)
{
    return (struct odl_verbs_device *)d;
}

static inline struct odl_verbs_pd *
odl_pd_from_ibv(struct ibv_pd *pd)
{
    return (struct odl_verbs_pd *)pd;
}

static inline struct odl_verbs_mr *
odl_mr_from_ibv(struct ibv_mr *mr)
{
    return (struct odl_verbs_mr *)mr;
}

static inline struct odl_verbs_cq *
odl_cq_from_ibv(struct ibv_cq *cq)
{
    return (struct odl_verbs_cq *)cq;
}

static inline struct odl_verbs_qp *
odl_qp_from_ibv(struct ibv_qp *qp)
{
    return (struct odl_verbs_qp *)qp;
}

/* ── Internal API (declared in their respective .c files) ──────────── */

/* Device */
struct ibv_context *odl_ibv_open_device(struct ibv_device *device);
int odl_query_device_ex(struct ibv_context *, const struct ibv_query_device_ex_input *,
                         struct ibv_device_attr_ex *, size_t);
int odl_query_port(struct ibv_context *, uint8_t, struct ibv_port_attr *);
int odl_query_port_speed(struct ibv_context *, uint32_t, uint64_t *);
int odl_free_context(struct ibv_context *);

/* PD */
struct ibv_pd *odl_alloc_pd(struct ibv_context *);
int odl_dealloc_pd(struct ibv_pd *);

/* MR */
struct ibv_mr *odl_reg_mr(struct ibv_pd *, void *, size_t, uint64_t, int);
struct ibv_mr *odl_reg_dmabuf_mr(struct ibv_pd *, uint64_t, size_t, uint64_t, int, int);
int odl_dereg_mr(struct ibv_mr *);

/* CQ */
struct ibv_cq *odl_create_cq(struct ibv_context *, int, struct ibv_comp_channel *, int);
int odl_destroy_cq(struct ibv_cq *);
int odl_poll_cq(struct ibv_cq *, int, struct ibv_wc *);
int odl_req_notify_cq(struct ibv_cq *, int);
void odl_cq_event(struct ibv_cq *);
int odl_cq_post(struct odl_verbs_cq *, struct ibv_wc *);

/* QP */
struct ibv_qp *odl_create_qp(struct ibv_pd *, struct ibv_qp_init_attr *);
int odl_destroy_qp(struct ibv_qp *);
int odl_modify_qp(struct ibv_qp *, struct ibv_qp_attr *, int);
int odl_query_qp(struct ibv_qp *, struct ibv_qp_attr *, int, struct ibv_qp_init_attr *);
int odl_post_send(struct ibv_qp *, struct ibv_send_wr *, struct ibv_send_wr **);
int odl_post_recv(struct ibv_qp *, struct ibv_recv_wr *, struct ibv_recv_wr **);
int odl_rq_drain(struct odl_verbs_qp *oqp);

/* Ops table init */
void odl_init_context_ops(struct ibv_context *ctx);

#endif /* ODL_TB5_VERBS_H */
