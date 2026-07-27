/*
 * OdinLink — Verbs: Queue Pairs (The Actual Data Path)
 *
 * A QP (Queue Pair) is just a pair of send/receive queues — like a
 * pipe between two machines. This file maps an RDMA QP to an OdinLink
 * stream.
 *
 * How async I/O works here:
 * 1. ibv_post_send → enqueue the work request → return immediately
 * 2. A background worker thread polls the device fd (EPOLLOUT)
 * 3. When the hardware is ready, it dequeues a WR and calls the
 *    kernel's stream_send ioctl (which is O_NONBLOCK)
 * 4. If the kernel says EAGAIN (busy), re-enqueue and poll again
 * 5. On success, post a Work Completion to the CQ and wake the app
 *    via eventfd
 *
 * This gives true async I/O: the calling thread never blocks even
 * though the kernel path is synchronous under the hood.
 */

#include "odl_tb5_verbs.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <poll.h>

/* ── Worker Thread ──────────────────────────────────────────────────── */

static int odl_worker_poll_fd(struct odl_verbs_qp *qp, int timeout_ms)
{
    struct odl_verbs_context *ctx = qp->ctx;
    int dev_fd = ctx->base.cmd_fd;
    if (dev_fd < 0) return -EBADF;

    struct pollfd pfd = {
        .fd     = dev_fd,
        .events = POLLOUT,
    };

    int ret = poll(&pfd, 1, timeout_ms);
    if (ret < 0) return -errno;
    if (ret == 0) return -ETIMEDOUT;
    return (pfd.revents & POLLOUT) ? 0 : -EAGAIN;
}

/* Look up a dmabuf fd by lkey. Returns -1 if not a dmabuf MR. */
int odl_lookup_dmabuf_pub(struct odl_verbs_context *ctx, uint32_t lkey);
static int odl_lookup_dmabuf(struct odl_verbs_context *ctx, uint32_t lkey)
{
    if (!lkey) return -1;
    pthread_mutex_lock(&ctx->mr_lock);
    for (int i = 0; i < ctx->nmrs; i++) {
        struct odl_verbs_mr *mr = ctx->mrs[i];
        if (mr && mr->base.lkey == lkey && mr->mr_type == 1) {
            int fd = mr->dmabuf_fd;
            pthread_mutex_unlock(&ctx->mr_lock);
            return fd;
        }
    }
    pthread_mutex_unlock(&ctx->mr_lock);
    return -1;
}

static void *odl_qp_worker(void *arg)
{
    struct odl_verbs_qp *qp = arg;
    struct odl_verbs_context *ctx = qp->ctx;
    odl_tb5_t h = ctx->handle;

    while (qp->worker_running) {
        /* Make receive progress on EVERY iteration, not only when idle:
         * a QP with continuous send traffic would otherwise never drain RX. */
        odl_rq_drain(qp);

        bool     have_wr = false;
        void    *w_bounce = NULL;
        uint64_t w_wr_id = 0, w_addr = 0;
        uint32_t w_len = 0, w_lkey = 0;
        int      w_num_sge = 0;

        /* Dequeue one work request (copies, not caller pointers) */
        pthread_mutex_lock(&qp->sq_lock);
        if (qp->sq_count > 0) {
            w_wr_id   = qp->sq_wr_id[qp->sq_head];
            w_addr    = qp->sq_addr[qp->sq_head];
            w_len     = qp->sq_len[qp->sq_head];
            w_lkey    = qp->sq_lkey[qp->sq_head];
            w_num_sge = qp->sq_num_sge[qp->sq_head];
            w_bounce  = qp->sq_bounce[qp->sq_head];
            qp->sq_head = (qp->sq_head + 1) % ODL_VERBS_SQ_DEPTH;
            qp->sq_count--;
            have_wr = true;
        }
        pthread_mutex_unlock(&qp->sq_lock);

        if (!have_wr) {
            /* No send work: make receive progress, then yield briefly.
             * The RX drain is what turns posted buffers into completions. */
            odl_worker_poll_fd(qp, 2);
            continue;
        }

        /* Poll the device fd until it signals TX readiness.
         * Since the fd is O_NONBLOCK, the stream_send ioctl will
         * return -EAGAIN immediately if frames aren't available.
         * We poll first to avoid unnecessary ioctl calls.
         *
         * If poll fails (e.g., bad fd in mock mode), fall back to
         * immediate non-blocking send without waiting. */
        int poll_ret = odl_worker_poll_fd(qp, 5000);
        if (poll_ret == -ETIMEDOUT) {
            /* No response in 5s — try anyway, the send may still work */
        } else if (poll_ret != 0) {
            /* Bad fd or signal — try a non-blocking send directly,
             * then back off if it fails. */
            odl_logverbose("worker poll failed for stream %u: %d, "
                            "falling back to direct send",
                            qp->stream_id, poll_ret);
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000 };
            nanosleep(&ts, NULL);
        }

        /* Execute the send (non-blocking — fd is O_NONBLOCK) */
        int ret;
        struct ibv_wc wc;
        memset(&wc, 0, sizeof(wc));

        if (w_num_sge > 0) {
            int dmabuf_fd = odl_lookup_dmabuf(ctx, w_lkey);

            if (dmabuf_fd >= 0) {
                ret = odl_tb5_stream_send_dmabuf(
                    h, qp->stream_id, qp->dest_qp,
                    dmabuf_fd, 0, w_len);
            } else {
                void *data = (void *)(uintptr_t)w_addr;
                ret = odl_tb5_stream_send(
                    h, qp->stream_id, qp->dest_qp,
                    data, w_len);
            }

            if (ret == -EAGAIN) {
                /* Non-blocking send couldn't proceed — re-queue and retry */
                odl_logverbose("send EAGAIN stream=%u, re-queueing", qp->stream_id);
                pthread_mutex_lock(&qp->sq_lock);
                if (qp->sq_count < ODL_VERBS_SQ_DEPTH) {
                    qp->sq_wr_id[qp->sq_tail]   = w_wr_id;
                    qp->sq_addr[qp->sq_tail]    = w_addr;
                    qp->sq_len[qp->sq_tail]     = w_len;
                    qp->sq_lkey[qp->sq_tail]    = w_lkey;
                    qp->sq_num_sge[qp->sq_tail] = w_num_sge;
                    qp->sq_bounce[qp->sq_tail]  = w_bounce;
                    qp->sq_tail = (qp->sq_tail + 1) % ODL_VERBS_SQ_DEPTH;
                    qp->sq_count++;
                }
                pthread_mutex_unlock(&qp->sq_lock);
                continue;
            }

            wc.wr_id    = w_wr_id;
            wc.status   = (ret == 0) ? IBV_WC_SUCCESS : IBV_WC_GENERAL_ERR;
            wc.byte_len = (ret == 0) ? w_len : 0;
            wc.opcode   = IBV_WC_SEND;
            wc.qp_num   = qp->base.qp_num;

            if (ret != 0)
                odl_logerr("send failed stream=%u ret=%d", qp->stream_id, ret);
        } else {
            /* Zero-length send */
            wc.wr_id    = w_wr_id;
            wc.status   = IBV_WC_SUCCESS;
            wc.byte_len = 0;
            wc.opcode   = IBV_WC_SEND;
            wc.qp_num   = qp->base.qp_num;
        }

        free(w_bounce);

        /* Post completion to send CQ */
        if (qp->send_cq)
            odl_cq_post(qp->send_cq, &wc);

        atomic_fetch_sub(&qp->pending_sends, 1);
    }

    return NULL;
}

/* ── Create QP ──────────────────────────────────────────────────────── */

struct ibv_qp *odl_create_qp(struct ibv_pd *pd,
                              struct ibv_qp_init_attr *attr)
{
    ODL_TRACE_ENTRY();
    ODL_RETURN_NULL_IF(!pd || !attr, "null argument");

    struct odl_verbs_context *ctx = odl_ctx_from_ibv(pd->context);

    if (attr->qp_type != IBV_QPT_RC) {
        odl_logerr("unsupported QP type: %d", attr->qp_type);
        errno = EOPNOTSUPP;
        return NULL;
    }

    struct odl_verbs_qp *qp = calloc(1, sizeof(*qp));
    if (!qp) { errno = ENOMEM; return NULL; }

    /* Open an OdinLink-Five stream */
    uint8_t stream_id = 0;
    int ret = odl_tb5_stream_open(ctx->handle, 0, &stream_id);
    if (ret != 0) {
        odl_logerr("stream_open failed: %d", ret);
        free(qp);
        errno = ENODEV;
        return NULL;
    }

    qp->base.context     = pd->context;
    qp->base.pd          = pd;
    qp->base.send_cq     = attr->send_cq;
    qp->base.recv_cq     = attr->recv_cq;
    qp->base.qp_num      = stream_id;
    qp->base.qp_type     = attr->qp_type;
    qp->base.state       = IBV_QPS_RESET;
    qp->ctx              = ctx;
    qp->pd               = odl_pd_from_ibv(pd);
    qp->send_cq          = attr->send_cq ? odl_cq_from_ibv(attr->send_cq) : NULL;
    qp->recv_cq          = attr->recv_cq ? odl_cq_from_ibv(attr->recv_cq) : NULL;
    if (qp->recv_cq)
        qp->recv_cq->rx_qp = qp;   /* let poll_cq drive RX progress */
    qp->stream_id        = stream_id;

    /* Initialize SQ */
    pthread_mutex_init(&qp->sq_lock, NULL);
    pthread_mutex_init(&qp->rq_lock, NULL);
    pthread_mutex_init(&qp->drain_lock, NULL);
    qp->rq_head = qp->rq_tail = qp->rq_count = 0;
    qp->sq_head  = 0;
    qp->sq_tail  = 0;
    qp->sq_count = 0;

    atomic_init(&qp->pending_sends, 0);
    atomic_init(&qp->pending_recvs, 0);

    /* Track in context */
    pthread_mutex_lock(&ctx->qp_lock);
    if (ctx->nqps >= ODL_VERBS_MAX_QPS) {
        pthread_mutex_unlock(&ctx->qp_lock);
        odl_tb5_stream_close(ctx->handle, stream_id);
        pthread_mutex_destroy(&qp->sq_lock);
        free(qp);
        errno = ENOMEM;
        return NULL;
    }
    ctx->qps[ctx->nqps++] = qp;
    pthread_mutex_unlock(&ctx->qp_lock);

    /* Start async worker thread */
    qp->worker_running = true;
    ret = pthread_create(&qp->worker, NULL, odl_qp_worker, qp);
    if (ret != 0) {
        odl_logerr("pthread_create failed: %d", ret);
        qp->worker_running = false;
        odl_tb5_stream_close(ctx->handle, stream_id);
        pthread_mutex_destroy(&qp->sq_lock);
        free(qp);
        errno = EAGAIN;
        return NULL;
    }

    odl_loginfo("create_qp: stream=%u qp_num=%u", stream_id, qp->base.qp_num);
    ODL_TRACE_EXIT();
    return &qp->base;
}

/* ── Destroy QP ─────────────────────────────────────────────────────── */

int odl_destroy_qp(struct ibv_qp *qp)
{
    ODL_TRACE_ENTRY();
    if (!qp) return -EINVAL;

    struct odl_verbs_qp *oqp = odl_qp_from_ibv(qp);
    struct odl_verbs_context *ctx = oqp->ctx;

    /* Stop worker thread */
    oqp->worker_running = false;
    pthread_join(oqp->worker, NULL);

    /* Close the OdinLink-Five stream */
    if (oqp->stream_id > 0)
        odl_tb5_stream_close(ctx->handle, oqp->stream_id);

    /* Remove from context */
    pthread_mutex_lock(&ctx->qp_lock);
    for (int i = 0; i < ctx->nqps; i++) {
        if (ctx->qps[i] == oqp) {
            ctx->qps[i] = ctx->qps[--ctx->nqps];
            break;
        }
    }
    pthread_mutex_unlock(&ctx->qp_lock);

    pthread_mutex_destroy(&oqp->sq_lock);
    free(oqp);

    odl_loginfo("destroy_qp: stream=%u", oqp->stream_id);
    ODL_TRACE_EXIT_VAL(0);
}

/* ── Modify QP ─────────────────────────────────────────────────────── */

int odl_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                   int attr_mask)
{
    ODL_TRACE_ENTRY();
    ODL_RETURN_EINVAL_IF(!qp, "null qp");

    if (attr_mask & IBV_QP_DEST_QPN) {
        struct odl_verbs_qp *oqp = odl_qp_from_ibv(qp);
        oqp->dest_qp = (uint8_t)attr->dest_qp_num;
        odl_loginfo("modify_qp: qp_num=%u dest_qp_num=%u",
                    qp->qp_num, attr->dest_qp_num);
    }

    if (attr_mask & IBV_QP_STATE) {
        qp->state = attr->qp_state;
        odl_loginfo("modify_qp: qp_num=%u state=%d -> %d",
                     qp->qp_num, qp->state, attr->qp_state);

        /* On RTS transition, ensure peer is ready */
        if (attr->qp_state == IBV_QPS_RTS) {
            struct odl_verbs_qp *oqp = odl_qp_from_ibv(qp);
            odl_tb5_wait_peer(oqp->ctx->handle, 5000);
        }
    }

    ODL_TRACE_EXIT_VAL(0);
}

/* ── Post Send (async) ──────────────────────────────────────────────── */

int odl_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                   struct ibv_send_wr **bad_wr)
{
    struct odl_verbs_qp *oqp = odl_qp_from_ibv(qp);
    *bad_wr = NULL;

    pthread_mutex_lock(&oqp->sq_lock);

    while (wr) {
        if (oqp->sq_count >= ODL_VERBS_SQ_DEPTH) {
            *bad_wr = wr;
            pthread_mutex_unlock(&oqp->sq_lock);
            odl_logerr("post_send: SQ full on QP %u", qp->qp_num);
            return -ENOMEM;
        }

        /* BUG14 fix: copy, never store the caller's stack pointer. */
        oqp->sq_wr_id[oqp->sq_tail]   = wr->wr_id;
        oqp->sq_num_sge[oqp->sq_tail] = wr->num_sge;
        if (wr->num_sge > 0) {
            uint32_t blen = wr->sg_list[0].length;
            void *bounce = NULL;

            /* Copy now; the caller may reuse its buffer the moment we return.
             * dmabuf MRs are zero-copy by definition and are left alone. */
            if (odl_lookup_dmabuf_pub(oqp->ctx, wr->sg_list[0].lkey) < 0 &&
                blen > 0) {
                bounce = malloc(blen);
                if (!bounce) {
                    *bad_wr = wr;
                    pthread_mutex_unlock(&oqp->sq_lock);
                    odl_logerr("post_send: bounce alloc %u failed", blen);
                    return -ENOMEM;
                }
                memcpy(bounce, (const void *)(uintptr_t)wr->sg_list[0].addr,
                       blen);
            }
            oqp->sq_bounce[oqp->sq_tail] = bounce;
            oqp->sq_addr[oqp->sq_tail] = bounce ? (uint64_t)(uintptr_t)bounce
                                               : wr->sg_list[0].addr;
            oqp->sq_len[oqp->sq_tail]  = blen;
            oqp->sq_lkey[oqp->sq_tail] = wr->sg_list[0].lkey;
        } else {
            oqp->sq_bounce[oqp->sq_tail] = NULL;
            oqp->sq_addr[oqp->sq_tail] = 0;
            oqp->sq_len[oqp->sq_tail]  = 0;
            oqp->sq_lkey[oqp->sq_tail] = 0;
        }
        oqp->sq_tail = (oqp->sq_tail + 1) % ODL_VERBS_SQ_DEPTH;
        oqp->sq_count++;
        atomic_fetch_add(&oqp->pending_sends, 1);

        wr = wr->next;
    }

    pthread_mutex_unlock(&oqp->sq_lock);
    return 0;
}

/* ── Post Recv (async via poll + non-blocking) ──────────────────────── */

int odl_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                   struct ibv_recv_wr **bad_wr)
{
    struct odl_verbs_qp *oqp = odl_qp_from_ibv(qp);
    *bad_wr = NULL;

    /*
     * BUG13 fix: this used to poll(5000) and perform the receive inline,
     * returning -ETIMEDOUT/-EAGAIN when no data had arrived yet. Every verbs
     * consumer PRE-POSTS receive buffers before any traffic exists (llama.cpp
     * posts 24, RCCL and perftest do the same), so the first post always
     * failed and RDMA setup aborted. A real post_recv only enqueues a buffer
     * and returns immediately; the completion arrives later via the CQ.
     */
    pthread_mutex_lock(&oqp->rq_lock);

    while (wr) {
        if (oqp->rq_count >= ODL_VERBS_RQ_DEPTH) {
            *bad_wr = wr;
            pthread_mutex_unlock(&oqp->rq_lock);
            odl_logerr("post_recv: RQ full on QP %u", qp->qp_num);
            return -ENOMEM;
        }
        if (wr->num_sge > 0) {
            /* Copy: the caller's wr/sge are typically stack-allocated. */
            oqp->rq_wr_id[oqp->rq_tail] = wr->wr_id;
            oqp->rq_addr[oqp->rq_tail]  = wr->sg_list[0].addr;
            oqp->rq_len[oqp->rq_tail]   = wr->sg_list[0].length;
            oqp->rq_tail = (oqp->rq_tail + 1) % ODL_VERBS_RQ_DEPTH;
            oqp->rq_count++;
            atomic_fetch_add(&oqp->pending_recvs, 1);
        }
        wr = wr->next;
    }

    pthread_mutex_unlock(&oqp->rq_lock);
    return 0;
}

/* Drain inbound stream data into posted RX buffers; post one WC per message.
 * Non-blocking: returns the number of completions generated. */
int odl_rq_drain(struct odl_verbs_qp *oqp)
{
    int completions = 0;

    /* Exclusive: only one drainer at a time, so receives stay ordered.
     * trylock, never block -- ibv_poll_cq() must not stall behind the worker. */
    if (pthread_mutex_trylock(&oqp->drain_lock) != 0)
        return 0;

    for (;;) {
        uint64_t wr_id, addr;
        uint32_t len;

        pthread_mutex_lock(&oqp->rq_lock);
        if (oqp->rq_count == 0) {
            pthread_mutex_unlock(&oqp->rq_lock);
            break;
        }
        wr_id = oqp->rq_wr_id[oqp->rq_head];
        addr  = oqp->rq_addr[oqp->rq_head];
        len   = oqp->rq_len[oqp->rq_head];
        pthread_mutex_unlock(&oqp->rq_lock);

        uint8_t  src_id = 0;
        uint32_t actual = 0;
        int ret = odl_tb5_stream_recv(oqp->ctx->handle, oqp->stream_id,
                                      (void *)(uintptr_t)addr, len,
                                      &src_id, &actual);
        if (ret == -EAGAIN)
            break;                      /* nothing pending -- leave buffer posted */

        /* Consume the buffer only once we know data (or an error) arrived. */
        pthread_mutex_lock(&oqp->rq_lock);
        if (oqp->rq_count > 0) {
            oqp->rq_head = (oqp->rq_head + 1) % ODL_VERBS_RQ_DEPTH;
            oqp->rq_count--;
        }
        pthread_mutex_unlock(&oqp->rq_lock);
        atomic_fetch_sub(&oqp->pending_recvs, 1);

        struct ibv_wc wc;
        memset(&wc, 0, sizeof(wc));
        wc.wr_id  = wr_id;
        wc.qp_num = oqp->base.qp_num;
        wc.src_qp = src_id;
        wc.opcode = IBV_WC_RECV;
        if (ret == 0) {
            wc.status   = IBV_WC_SUCCESS;
            wc.byte_len = actual;
        } else {
            wc.status   = IBV_WC_GENERAL_ERR;
            wc.byte_len = 0;
            odl_logerr("recv failed: stream=%u ret=%d", oqp->stream_id, ret);
        }
        if (oqp->recv_cq)
            odl_cq_post(oqp->recv_cq, &wc);
        completions++;
    }

    pthread_mutex_unlock(&oqp->drain_lock);
    return completions;
}

/* ── Query QP ───────────────────────────────────────────────────────── */

int odl_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                  int attr_mask, struct ibv_qp_init_attr *init_attr)
{
    ODL_TRACE_ENTRY();
    ODL_RETURN_EINVAL_IF(!qp, "null qp");
    (void)attr_mask;

    memset(attr, 0, sizeof(*attr));
    attr->qp_state         = qp->state;
    attr->cur_qp_state     = qp->state;
    attr->path_mtu         = IBV_MTU_4096;
    attr->path_mig_state   = IBV_MIG_MIGRATED;
    attr->qkey             = 0;
    attr->rq_psn           = 0;
    attr->sq_psn           = 0;
    attr->dest_qp_num      = 0;
    attr->qp_access_flags  = IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ;
    attr->cap.max_send_wr  = ODL_VERBS_SQ_DEPTH;
    attr->cap.max_recv_wr  = ODL_VERBS_SQ_DEPTH;
    attr->cap.max_send_sge = 1;
    attr->cap.max_recv_sge = 1;
    attr->cap.max_inline_data = 0;
    attr->port_num         = 1;
    attr->timeout          = 0;
    attr->retry_cnt        = 0;
    attr->rnr_retry        = 0;
    attr->alt_port_num     = 0;
    attr->alt_timeout      = 0;

    if (init_attr) {
        memset(init_attr, 0, sizeof(*init_attr));
        init_attr->qp_type = qp->qp_type;
        init_attr->send_cq = qp->send_cq;
        init_attr->recv_cq = qp->recv_cq;
        init_attr->cap     = attr->cap;
    }

    ODL_TRACE_EXIT_VAL(0);
}

int odl_lookup_dmabuf_pub(struct odl_verbs_context *ctx, uint32_t lkey)
{
    return odl_lookup_dmabuf(ctx, lkey);
}
