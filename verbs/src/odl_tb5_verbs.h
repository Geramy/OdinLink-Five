#ifndef ODL_TB5_VERBS_H
#define ODL_TB5_VERBS_H

/*
 * OdinLink Verbs Provider — Internal Header
 *
 * This provider implements the full libibverbs API backed by
 * OdinLink-Five Thunderbolt 5 DMA transport.
 *
 * Architecture:
 *   ibv_context     → odl_tb5_handle (device fd + mmap'd buffers)
 *   ibv_pd          → lightweight permission boundary
 *   ibv_mr          → DMA-buf fd (GPU) or host memory pointer
 *   ibv_cq          → completion ring + eventfd notification
 *   ibv_qp          → odl_tb5_stream with async workqueue worker
 *
 * Async model:
 *   ibv_post_send enqueues a work request → worker thread drains →
 *   calls blocking odl_tb5_stream_send → posts completion to CQ.
 *   This provides non-blocking submission even with the current
 *   synchronous kernel driver.
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
};

/* ── Queue Pair ─────────────────────────────────────────────────────── */

struct odl_verbs_qp {
    struct ibv_qp             base;
    struct odl_verbs_context *ctx;
    struct odl_verbs_pd      *pd;
    struct odl_verbs_cq      *send_cq;
    struct odl_verbs_cq      *recv_cq;
    uint8_t                   stream_id;

    /* Work submission queue (async via worker thread) */
    pthread_mutex_t           sq_lock;
    struct ibv_send_wr       *sq[ODL_VERBS_SQ_DEPTH];
    int                       sq_head;
    int                       sq_tail;
    int                       sq_count;

    /* Worker thread */
    pthread_t                 worker;
    bool                      worker_running;

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

/* Ops table init */
void odl_init_context_ops(struct ibv_context *ctx);

#endif /* ODL_TB5_VERBS_H */
