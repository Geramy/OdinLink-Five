/*
 * OdinLink — Verbs: Protection Domains (Permission Boundaries)
 *
 * A PD is a lightweight permission domain — it groups together memory
 * regions and queue pairs that are allowed to talk to each other.
 * In OdinLink, PDs are mostly tracking state; the real security is
 * in the kernel driver.
 */
#include "odl_tb5_verbs.h"
#include <stdlib.h>
#include <errno.h>

struct ibv_pd *odl_alloc_pd(struct ibv_context *context)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);

    struct odl_verbs_pd *pd = calloc(1, sizeof(*pd));
    if (!pd) { errno = ENOMEM; return NULL; }

    pd->base.context = context;
    pd->base.handle  = 0;
    pd->ctx = ctx;

    pthread_mutex_lock(&ctx->pd_lock);
    if (ctx->npds >= ODL_VERBS_MAX_PDS) {
        pthread_mutex_unlock(&ctx->pd_lock);
        free(pd);
        errno = ENOMEM;
        return NULL;
    }
    ctx->pds[ctx->npds++] = pd;
    pthread_mutex_unlock(&ctx->pd_lock);

    ODL_TRACE_EXIT();
    return &pd->base;
}

int odl_dealloc_pd(struct ibv_pd *pd)
{
    ODL_TRACE_ENTRY();
    if (!pd) return -EINVAL;
    struct odl_verbs_pd *opd = odl_pd_from_ibv(pd);
    struct odl_verbs_context *ctx = opd->ctx;

    pthread_mutex_lock(&ctx->pd_lock);
    for (int i = 0; i < ctx->npds; i++) {
        if (ctx->pds[i] == opd) {
            ctx->pds[i] = ctx->pds[--ctx->npds];
            break;
        }
    }
    pthread_mutex_unlock(&ctx->pd_lock);

    free(opd);
    ODL_TRACE_EXIT_VAL(0);
}
