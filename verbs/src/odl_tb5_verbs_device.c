/*
 * OdinLink — Verbs: Opening and Querying an OdinLink Device
 *
 * When a program calls ibv_open_device on an OdinLink device, this
 * code creates the ibv_context — the central handle that everything
 * else (PDs, MRs, CQs, QPs) hangs off. It also implements
 * ibv_query_device and ibv_query_port to report link speed, max
 * QPs, etc.
 *
 * For non-OdinLink devices, it uses dlsym to pass through to the
 * real libibverbs — so LD_PRELOAD of this library doesn't break
 * your Mellanox card.
 */

#include "odl_tb5_verbs.h"

#include <dlfcn.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
/* Private libibverbs symbols resolved at runtime */
/* verbs_context is defined in <infiniband/verbs.h> (rdma-core ≥ 50) */
typedef void *(*verbs_init_alloc_fn_t)(struct ibv_device *, int, size_t,
                                        struct verbs_context *, uint32_t);
typedef void (*verbs_set_ops_fn_t)(struct verbs_context *, const void *);

static verbs_init_alloc_fn_t  p_verbs_init_alloc = NULL;
static verbs_set_ops_fn_t     p_verbs_set_ops    = NULL;

static int resolve_verbs_private(void)
{
    static int done = 0;
    if (done) return 0;
    void *h = dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return -1;
    p_verbs_init_alloc = dlsym(h, "_verbs_init_and_alloc_context");
    p_verbs_set_ops    = dlsym(h, "verbs_set_ops");
    dlclose(h);
    done = (p_verbs_init_alloc && p_verbs_set_ops);
    return done ? 0 : -1;
}

/* ── Chaining to real libibverbs ────────────────────────────────────── */

/* Resolve real libibverbs functions via dlsym for forwarding */
typedef struct ibv_context *(*real_open_device_fn)(struct ibv_device *, int);

static real_open_device_fn real_ibv_open_device = NULL;
static int real_symbols_resolved = 0;

static int resolve_real_verbs(void)
{
    if (real_symbols_resolved) return real_symbols_resolved > 0 ? 0 : -1;
    void *h = dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) { real_symbols_resolved = -1; return -1; }
    real_ibv_open_device = dlsym(h, "ibv_open_device");
    dlclose(h);
    real_symbols_resolved = real_ibv_open_device ? 1 : -1;
    return real_symbols_resolved > 0 ? 0 : -1;
}

/* ── Open Device ────────────────────────────────────────────────────── */

struct ibv_context *odl_ibv_open_device(struct ibv_device *device)
{
    ODL_TRACE_ENTRY();
    ODL_RETURN_NULL_IF(!device, "device is NULL");

    /* Chain to real libibverbs for non-ODL devices */
    bool is_odl = device->name &&
                  strncmp(device->name, "odl_tb5_", 8) == 0;

    if (!is_odl) {
        if (resolve_real_verbs() == 0 && real_ibv_open_device) {
            odl_loginfo("forwarding non-ODL device %s to real libibverbs",
                         device->name);
            struct ibv_context *ctx = real_ibv_open_device(device, -1);
            ODL_TRACE_EXIT();
            return ctx;
        }
        odl_logerr("non-ODL device %s but real libibverbs unavailable",
                    device->name);
        errno = ENODEV;
        return NULL;
    }

    struct odl_verbs_device *odl_dev = odl_dev_from_ibv(device);

    /* Allocate context */
    struct odl_verbs_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        odl_logerr("calloc failed");
        errno = ENOMEM;
        return NULL;
    }

    /* Open the OdinLink-Five char device */
    odl_tb5_t handle;
    int ret = odl_tb5_open(&handle, odl_dev->dev_index);
    if (ret != 0) {
        odl_logerr("odl_tb5_open(%d) failed: %d",
                    odl_dev->dev_index, ret);
        free(ctx);
        errno = ENODEV;
        return NULL;
    }

    /* Set non-blocking mode on the device fd so stream_send/recv
     * ioctls return -EAGAIN instead of blocking. The verbs provider
     * uses poll() + non-blocking ioctls for true async behavior. */
    int dev_fd = odl_tb5_get_fd(handle);
    if (dev_fd >= 0) {
        int flags = fcntl(dev_fd, F_GETFL, 0);
        if (flags >= 0)
            fcntl(dev_fd, F_SETFL, flags | O_NONBLOCK);
        ctx->base.cmd_fd = dev_fd;
    }

    /* Initialize context fields */
    ctx->base.device        = device;
    ctx->base.cmd_fd        = -1;
    ctx->base.async_fd      = -1;
    ctx->base.num_comp_vectors = 1;
    ctx->dev                = odl_dev;
    ctx->handle             = handle;
    ctx->refcount           = 1;
    pthread_mutex_init(&ctx->pd_lock, NULL);
    pthread_mutex_init(&ctx->mr_lock, NULL);
    pthread_mutex_init(&ctx->cq_lock, NULL);
    pthread_mutex_init(&ctx->qp_lock, NULL);

    /* Initialize the ops table that libibverbs dispatches to */
    odl_init_context_ops(&ctx->base);

    odl_dev->active_ctx = ctx;
    odl_dev->is_open     = true;

    odl_loginfo("context opened: dev=%s handle=%p ctx=%p",
                 odl_dev->dev_name, (void*)handle, (void*)ctx);

    ODL_TRACE_EXIT();
    return &ctx->base;
}

/* ── Query Device ───────────────────────────────────────────────────── */

int odl_query_device_ex(struct ibv_context *context,
                         const struct ibv_query_device_ex_input *input,
                         struct ibv_device_attr_ex *attr,
                         size_t attr_size)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);
    (void)input;

    if (attr_size < sizeof(*attr)) return -EINVAL;

    memset(attr, 0, attr_size);

    struct odl_tb5_peer_info peer;
    bool connected = (odl_tb5_get_peer(ctx->handle, &peer) == 0 &&
                      peer.state >= ODL_TB5_STATE_CONNECTED);

    attr->orig_attr.phys_port_cnt    = 1;
    attr->orig_attr.max_qp           = ODL_VERBS_MAX_QPS;
    attr->orig_attr.max_qp_wr        = ODL_VERBS_SQ_DEPTH;
    attr->orig_attr.max_sge          = 1;
    attr->orig_attr.max_sge_rd       = 1;
    attr->orig_attr.max_cq           = ODL_VERBS_MAX_CQS;
    attr->orig_attr.max_mr           = ODL_VERBS_MAX_MRS;
    attr->orig_attr.max_pd           = ODL_VERBS_MAX_PDS;
    attr->orig_attr.max_mr_size      = SIZE_MAX;
    attr->orig_attr.max_qp_rd_atom   = 1;
    attr->orig_attr.max_qp_init_rd_atom = 1;
    attr->orig_attr.max_res_rd_atom  = 1;

    if (connected) {
        odl_loginfo("device active: link_speed=%u", peer.link_speed);
    } else {
        odl_loginfo("device idle (no peer)");
    }

    ODL_TRACE_EXIT_VAL(0);
}

/* ── Query Port ─────────────────────────────────────────────────────── */

int odl_query_port(struct ibv_context *context, uint8_t port_num,
                    struct ibv_port_attr *attr)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);

    if (port_num != 1) return -EINVAL;

    memset(attr, 0, sizeof(*attr));

    struct odl_tb5_peer_info peer;
    bool connected = (odl_tb5_get_peer(ctx->handle, &peer) == 0 &&
                      peer.state >= ODL_TB5_STATE_CONNECTED);

    attr->max_mtu        = IBV_MTU_4096;
    attr->active_mtu     = IBV_MTU_4096;
    attr->gid_tbl_len    = 1;
    attr->port_cap_flags = IBV_PORT_CM_SUP;
    attr->max_msg_sz     = 1 << 20;
    attr->bad_pkey_cntr  = 0;
    attr->qkey_viol_cntr = 0;
    attr->pkey_tbl_len   = 0;
    attr->lid            = 0;
    attr->sm_lid         = 0;
    attr->lmc            = 0;
    attr->sm_sl          = 0;
    attr->subnet_timeout = 0;
    attr->init_type_reply = 0;

    if (connected) {
        attr->state        = IBV_PORT_ACTIVE;
        attr->phys_state   = 5;
        attr->active_width = IBV_WIDTH_4X;
        attr->active_speed = IBV_SPEED_EDR;
    } else {
        attr->state        = IBV_PORT_DOWN;
        attr->phys_state   = 3;
        attr->active_width = IBV_WIDTH_1X;
        attr->active_speed = IBV_SPEED_SDR;
    }

    ODL_TRACE_EXIT_VAL(0);
}

/* ── Query Port Speed ───────────────────────────────────────────────── */

int odl_query_port_speed(struct ibv_context *context, uint32_t port_num,
                          uint64_t *speed)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);
    (void)port_num;

    struct odl_tb5_peer_info peer;
    if (odl_tb5_get_peer(ctx->handle, &peer) == 0 && peer.link_speed > 0)
        *speed = (uint64_t)peer.link_speed * 1000000000ULL;
    else
        *speed = 80000000000ULL; /* 80 Gbps TB5 default */

    ODL_TRACE_EXIT_VAL(0);
}

/* ── Free Context ───────────────────────────────────────────────────── */

int odl_free_context(struct ibv_context *context)
{
    ODL_TRACE_ENTRY();
    if (!context) return -EINVAL;

    /* Provider plugin path: handle was stashed in abi_compat.
     * Close it and free the plain context. */
    if (context->abi_compat) {
        odl_tb5_t handle = (odl_tb5_t)(intptr_t)context->abi_compat;
        odl_loginfo("closing provider context %p (handle=%p)",
                     (void*)context, (void*)handle);
        odl_tb5_close(handle);
        free(context);
        ODL_TRACE_EXIT_VAL(0);
    }

    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);

    odl_loginfo("closing context %p (handle=%p)",
                 (void*)ctx, (void*)ctx->handle);

    /* Destroy all QPs */
    pthread_mutex_lock(&ctx->qp_lock);
    for (int i = 0; i < ctx->nqps; i++) {
        struct odl_verbs_qp *qp = ctx->qps[i];
        if (qp) {
            qp->worker_running = false;
            pthread_join(qp->worker, NULL);
            pthread_mutex_destroy(&qp->sq_lock);
            if (qp->stream_id > 0)
                odl_tb5_stream_close(ctx->handle, qp->stream_id);
            free(qp);
        }
    }
    ctx->nqps = 0;
    pthread_mutex_unlock(&ctx->qp_lock);

    /* Destroy all CQs */
    pthread_mutex_lock(&ctx->cq_lock);
    for (int i = 0; i < ctx->ncqs; i++) {
        struct odl_verbs_cq *cq = ctx->cqs[i];
        if (cq) {
            if (cq->eventfd_fd >= 0) close(cq->eventfd_fd);
            pthread_mutex_destroy(&cq->lock);
            free(cq);
        }
    }
    ctx->ncqs = 0;
    pthread_mutex_unlock(&ctx->cq_lock);

    /* Destroy all MRs */
    pthread_mutex_lock(&ctx->mr_lock);
    for (int i = 0; i < ctx->nmrs; i++) {
        struct odl_verbs_mr *mr = ctx->mrs[i];
        if (mr) {
            if (mr->dmabuf_fd >= 0) close(mr->dmabuf_fd);
            free(mr);
        }
    }
    ctx->nmrs = 0;
    pthread_mutex_unlock(&ctx->mr_lock);

    /* Destroy all PDs */
    pthread_mutex_lock(&ctx->pd_lock);
    for (int i = 0; i < ctx->npds; i++) {
        free(ctx->pds[i]);
    }
    ctx->npds = 0;
    pthread_mutex_unlock(&ctx->pd_lock);

    /* Close the OdinLink-Five device */
    if (ctx->handle) {
        odl_tb5_close(ctx->handle);
    }

    if (ctx->dev)
        ctx->dev->active_ctx = NULL;

    pthread_mutex_destroy(&ctx->pd_lock);
    pthread_mutex_destroy(&ctx->mr_lock);
    pthread_mutex_destroy(&ctx->cq_lock);
    pthread_mutex_destroy(&ctx->qp_lock);

    free(ctx);
    odl_loginfo("context freed");
    return 0;
}
