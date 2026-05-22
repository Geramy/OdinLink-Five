/*
 * OdinLink Verbs Provider — Completion Queues
 *
 * Completion queues aggregate per-operation completion events.
 * Supports eventfd-based notification for async wakeup.
 *
 * CQ lifecycle:
 *   odl_create_cq  → allocate ring buffer + eventfd
 *   odl_cq_post    → push completion (from worker thread)
 *   odl_poll_cq    → drain completions (from application)
 *   odl_req_notify_cq → arm eventfd for edge-triggered notification
 *   odl_destroy_cq → close eventfd + free
 */

#include "odl_tb5_verbs.h"
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

struct ibv_cq *odl_create_cq(struct ibv_context *context, int cqe,
                              struct ibv_comp_channel *channel,
                              int comp_vector)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);
    (void)comp_vector;

    struct odl_verbs_cq *cq = calloc(1, sizeof(*cq));
    if (!cq) { errno = ENOMEM; return NULL; }

    cq->base.context = context;
    cq->base.channel = channel;
    cq->base.cqe     = cqe > 0 ? cqe : 1;
    cq->ctx = ctx;
    pthread_mutex_init(&cq->lock, NULL);

    /* Create eventfd for async notification */
    cq->eventfd_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (cq->eventfd_fd < 0) {
        odl_logerr("eventfd failed: %s", strerror(errno));
        pthread_mutex_destroy(&cq->lock);
        free(cq);
        return NULL;
    }

    /* Wire up completion channel if provided */
    if (channel) {
        channel->fd     = cq->eventfd_fd;
        channel->refcnt = 0;
    }

    pthread_mutex_lock(&ctx->cq_lock);
    if (ctx->ncqs >= ODL_VERBS_MAX_CQS) {
        pthread_mutex_unlock(&ctx->cq_lock);
        close(cq->eventfd_fd);
        pthread_mutex_destroy(&cq->lock);
        free(cq);
        errno = ENOMEM;
        return NULL;
    }
    ctx->cqs[ctx->ncqs++] = cq;
    pthread_mutex_unlock(&ctx->cq_lock);

    odl_loginfo("create_cq: cqe=%d eventfd=%d", cqe, cq->eventfd_fd);
    ODL_TRACE_EXIT();
    return &cq->base;
}

int odl_destroy_cq(struct ibv_cq *cq)
{
    ODL_TRACE_ENTRY();
    if (!cq) return -EINVAL;

    struct odl_verbs_cq *ocq = odl_cq_from_ibv(cq);
    struct odl_verbs_context *ctx = ocq->ctx;

    pthread_mutex_lock(&ctx->cq_lock);
    for (int i = 0; i < ctx->ncqs; i++) {
        if (ctx->cqs[i] == ocq) {
            ctx->cqs[i] = ctx->cqs[--ctx->ncqs];
            break;
        }
    }
    pthread_mutex_unlock(&ctx->cq_lock);

    if (ocq->eventfd_fd >= 0) close(ocq->eventfd_fd);
    pthread_mutex_destroy(&ocq->lock);
    free(ocq);

    ODL_TRACE_EXIT_VAL(0);
}

int odl_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc)
{
    struct odl_verbs_cq *ocq = odl_cq_from_ibv(cq);
    int polled = 0;

    pthread_mutex_lock(&ocq->lock);

    while (polled < num_entries && ocq->head != ocq->tail) {
        wc[polled] = ocq->ring[ocq->head];
        ocq->head = (ocq->head + 1) % ODL_VERBS_COMP_CHANNEL_BACKLOG;
        polled++;
    }

    /* Clear eventfd if we drained the ring */
    if (ocq->head == ocq->tail) {
        eventfd_t val;
        eventfd_read(ocq->eventfd_fd, &val);
    }

    pthread_mutex_unlock(&ocq->lock);
    return polled;
}

int odl_req_notify_cq(struct ibv_cq *cq, int solicited_only)
{
    struct odl_verbs_cq *ocq = odl_cq_from_ibv(cq);
    (void)solicited_only;

    pthread_mutex_lock(&ocq->lock);
    ocq->armed = true;
    pthread_mutex_unlock(&ocq->lock);

    return 0;
}

void odl_cq_event(struct ibv_cq *cq)
{
    (void)cq;
}

/* ── Internal: Post Completion ─────────────────────────────────────── */

int odl_cq_post(struct odl_verbs_cq *cq, struct ibv_wc *wc)
{
    pthread_mutex_lock(&cq->lock);

    int next = (cq->tail + 1) % ODL_VERBS_COMP_CHANNEL_BACKLOG;
    if (next == cq->head) {
        /* CQ full — drop completion */
        odl_logerr("CQ %p ring full! dropping completion", (void*)cq);
        pthread_mutex_unlock(&cq->lock);
        return -ENOSPC;
    }

    cq->ring[cq->tail] = *wc;
    cq->tail = next;

    /* Signal eventfd if armed */
    if (cq->armed) {
        eventfd_t val = 1;
        eventfd_write(cq->eventfd_fd, val);
        cq->armed = false;
    }

    pthread_mutex_unlock(&cq->lock);
    return 0;
}
