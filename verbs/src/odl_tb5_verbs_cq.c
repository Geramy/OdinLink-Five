#include <fcntl.h>
#include <stdbool.h>
/*
 * OdinLink — Verbs: Completion Queues (Where Finished Ops Are Reported)
 *
 * After you post a send or recv, the result shows up in a CQ —
 * a ring buffer of "done" notifications. This file implements:
 *
 *   odl_create_cq     → allocate the ring + an eventfd for wakeups
 *   odl_cq_post       → push a completion (called by the worker thread)
 *   odl_poll_cq       → application checks what's done (non-blocking)
 *   odl_req_notify_cq → arm the eventfd so poll/select wakes the app
 *   odl_destroy_cq    → cleanup
 *
 * The eventfd is how the worker thread tells the app "hey, new results".
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
    if (cq->eventfd_fd >= 0) {
        /* Belt and braces: ibv_poll_cq must never block, so guarantee the
         * flag rather than trusting it (BUG19). */
        int fl = fcntl(cq->eventfd_fd, F_GETFL, 0);
        if (fl >= 0 && !(fl & O_NONBLOCK))
            fcntl(cq->eventfd_fd, F_SETFL, fl | O_NONBLOCK);
    }
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

    /*
     * Drive receive progress from the polling thread BEFORE taking the CQ
     * lock (odl_rq_drain -> odl_cq_post takes it). Relying on the QP worker
     * alone deadlocks under sustained load: the worker blocks in its TX
     * readiness poll, RX is never drained, and both peers spin forever.
     */
    if (ocq->rx_qp)
        odl_rq_drain(ocq->rx_qp);

    pthread_mutex_lock(&ocq->lock);

    while (polled < num_entries && ocq->head != ocq->tail) {
        wc[polled] = ocq->ring[ocq->head];
        ocq->head = (ocq->head + 1) % ODL_VERBS_COMP_CHANNEL_BACKLOG;
        polled++;
    }

    bool drained = (ocq->head == ocq->tail);

    pthread_mutex_unlock(&ocq->lock);

    /*
     * BUG19: this eventfd drain used to run INSIDE the mutex with a plain
     * eventfd_read(). ibv_poll_cq() must never block -- consumers busy-poll
     * it -- but when the ring was empty the read blocked on a zero counter
     * while still holding ocq->lock, and odl_cq_post() needs that same lock
     * to deliver a completion. The only thread that could wake the poller was
     * therefore locked out of doing so: a self-deadlock that froze the first
     * bulk transfer.
     *
     * Drain outside the lock, and force O_NONBLOCK on the fd rather than
     * trusting the creation flags, so an empty CQ can only ever return 0.
     */
    /*
     * Only touch the eventfd when we actually consumed completions and
     * emptied the ring. On an empty CQ -- the overwhelmingly common case in a
     * busy-poll loop -- do nothing: two syscalls per poll iteration would
     * dominate the transfer.
     */
    if (polled > 0 && drained && ocq->eventfd_fd >= 0) {
        eventfd_t val;
        (void)eventfd_read(ocq->eventfd_fd, &val);   /* EAGAIN when empty */
    }

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
