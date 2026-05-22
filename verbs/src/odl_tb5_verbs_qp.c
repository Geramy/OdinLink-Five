/*
 * OdinLink Verbs Provider — Queue Pairs
 *
 * Async Queue Pair with poll()-based non-blocking I/O:
 *
 * ibv_post_send → enqueue to SQ → return immediately
 * Worker thread → poll(fd, EPOLLOUT) → dequeue WR → ioctl(nonblock)
 *              → on -EAGAIN → re-enqueue WR → poll again
 *              → on success → post WC to CQ → signal eventfd
 *
 * The device fd is set to O_NONBLOCK at context init time, so all
 * stream_send/recv ioctls return -EAGAIN instead of blocking.
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
        struct ibv_send_wr *wr = NULL;

        /* Dequeue one work request */
        pthread_mutex_lock(&qp->sq_lock);
        if (qp->sq_count > 0) {
            wr = qp->sq[qp->sq_head];
            qp->sq_head = (qp->sq_head + 1) % ODL_VERBS_SQ_DEPTH;
            qp->sq_count--;
        }
        pthread_mutex_unlock(&qp->sq_lock);

        if (!wr) {
            /* No work — poll for TX readiness with 100ms timeout.
             * This also yields the CPU when idle instead of busy-waiting. */
            odl_worker_poll_fd(qp, 100);
            continue;
        }

        /* Poll the device fd until it signals TX readiness.
         * Since the fd is O_NONBLOCK, the stream_send ioctl will
         * return -EAGAIN immediately if frames aren't available.
         * We poll first to avoid unnecessary ioctl calls. */
        int poll_ret = odl_worker_poll_fd(qp, 5000);
        if (poll_ret != 0) {
            odl_logerr("worker poll failed for stream %u: %d",
                        qp->stream_id, poll_ret);
            /* Re-queue the WR and retry */
            pthread_mutex_lock(&qp->sq_lock);
            if (qp->sq_count < ODL_VERBS_SQ_DEPTH) {
                qp->sq[qp->sq_tail] = wr;
                qp->sq_tail = (qp->sq_tail + 1) % ODL_VERBS_SQ_DEPTH;
                qp->sq_count++;
            }
            pthread_mutex_unlock(&qp->sq_lock);
            continue;
        }

        /* Execute the send (non-blocking — fd is O_NONBLOCK) */
        int ret;
        struct ibv_wc wc;
        memset(&wc, 0, sizeof(wc));

        if (wr->num_sge > 0) {
            struct ibv_sge *sge = &wr->sg_list[0];
            int dmabuf_fd = odl_lookup_dmabuf(ctx, sge->lkey);

            if (dmabuf_fd >= 0) {
                ret = odl_tb5_stream_send_dmabuf(
                    h, qp->stream_id, 0,
                    dmabuf_fd, 0, sge->length);
            } else {
                void *data = (void *)(uintptr_t)sge->addr;
                ret = odl_tb5_stream_send(
                    h, qp->stream_id, 0,
                    data, sge->length);
            }

            if (ret == -EAGAIN) {
                /* Non-blocking send couldn't proceed — re-queue and retry */
                odl_logverbose("send EAGAIN stream=%u, re-queueing", qp->stream_id);
                pthread_mutex_lock(&qp->sq_lock);
                if (qp->sq_count < ODL_VERBS_SQ_DEPTH) {
                    qp->sq[qp->sq_tail] = wr;
                    qp->sq_tail = (qp->sq_tail + 1) % ODL_VERBS_SQ_DEPTH;
                    qp->sq_count++;
                }
                pthread_mutex_unlock(&qp->sq_lock);
                continue;
            }

            wc.status   = (ret == 0) ? IBV_WC_SUCCESS : IBV_WC_GENERAL_ERR;
            wc.byte_len = (ret == 0) ? sge->length : 0;
            wc.opcode   = IBV_WC_SEND;
            wc.qp_num   = qp->base.qp_num;

            if (ret != 0)
                odl_logerr("send failed stream=%u ret=%d", qp->stream_id, ret);
        } else {
            /* Zero-length send */
            wc.status   = IBV_WC_SUCCESS;
            wc.byte_len = 0;
            wc.opcode   = IBV_WC_SEND;
            wc.qp_num   = qp->base.qp_num;
        }

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
    qp->stream_id        = stream_id;

    /* Initialize SQ */
    pthread_mutex_init(&qp->sq_lock, NULL);
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

        oqp->sq[oqp->sq_tail] = wr;
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
    struct odl_verbs_context *ctx = oqp->ctx;
    int dev_fd = ctx->base.cmd_fd;
    *bad_wr = NULL;

    while (wr) {
        if (wr->num_sge > 0) {
            struct ibv_sge *sge = &wr->sg_list[0];
            uint8_t src_id = 0;
            uint32_t actual = 0;

            /* Poll for RX readiness (up to 5 second timeout).
             * Since the fd is O_NONBLOCK, stream_recv returns -EAGAIN
             * immediately if no data is available. */
            if (dev_fd >= 0) {
                struct pollfd pfd = {
                    .fd     = dev_fd,
                    .events = POLLIN,
                };
                int pr = poll(&pfd, 1, 5000);
                if (pr <= 0 || !(pfd.revents & POLLIN)) {
                    *bad_wr = wr;
                    return pr == 0 ? -ETIMEDOUT : -EAGAIN;
                }
            }

            /* Non-blocking recv */
            int ret = odl_tb5_stream_recv(
                ctx->handle, oqp->stream_id,
                (void *)(uintptr_t)sge->addr,
                sge->length,
                &src_id, &actual);

            if (ret == -EAGAIN) {
                /* No data yet despite poll saying ready — rare race.
                 * Return the bad WR and let the caller retry. */
                *bad_wr = wr;
                return -EAGAIN;
            }

            struct ibv_wc wc;
            memset(&wc, 0, sizeof(wc));
            wc.qp_num   = qp->qp_num;
            wc.src_qp   = src_id;
            wc.opcode   = IBV_WC_RECV;
            wc.slid     = 0;
            wc.sl       = 0;
            wc.vendor_err = 0;

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

            atomic_fetch_sub(&oqp->pending_recvs, 1);
        }
        wr = wr->next;
    }

    return 0;
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
