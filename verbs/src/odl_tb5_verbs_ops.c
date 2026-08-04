/*
 * OdinLink — Verbs: The Wiring That Connects ibv_* Calls to Our Code
 *
 * When an app calls ibv_query_device, ibv_post_send, ibv_poll_cq, etc.,
 * libibverbs looks up the function pointer in a dispatch table. This
 * file sets up that table so every standard verbs call routes to our
 * OdinLink implementation.
 *
 * For ibv_query_device and ibv_query_port, modern rdma-core uses an
 * internal struct (verbs_context) instead of the old ops table, so we
 * intercept those symbols directly via symbol interposition. The rest
 * (poll_cq, post_send, post_recv, etc.) go through the ops fields we
 * populate here.
 */

#include "odl_tb5_verbs.h"
#define _GNU_SOURCE
#include <dlfcn.h>

/* The struct _compat_ibv_port_attr definition from rdma-core:
 * Same layout as ibv_port_attr but used for the legacy dispatch path. */
struct _compat_ibv_port_attr {
	enum ibv_port_state	state;
	enum ibv_mtu		max_mtu;
	enum ibv_mtu		active_mtu;
	int			phys_state;
	/*
	 * ibv_query_port() on the context's ops may fill in _compat fields
	 * only. In that case gid_tbl_len, port_cap_flags, and max_msg_sz
	 * are set to zero.
	 */
	uint16_t		gid_tbl_len;
	uint32_t		port_cap_flags;
	uint32_t		max_msg_sz;
	uint16_t		bad_pkey_cnt;
	uint16_t		qkey_viol_cnt;
	uint16_t		pkey_tbl_len;
	uint32_t		lid;
	uint32_t		sm_lid;
	uint8_t			lmc;
	uint8_t			sm_sl;
	uint8_t			subnet_timeout;
	uint8_t			init_type_reply;
};

/* ── Legacy compat: ibv_query_device (via _compat_query_device) ─────── */

static int odl_compat_query_device(struct ibv_context *context,
                                    struct ibv_device_attr *attr)
{
    ODL_TRACE_ENTRY();
    (void)context;

    memset(attr, 0, sizeof(*attr));
    attr->phys_port_cnt    = 1;
    attr->max_qp           = ODL_VERBS_MAX_QPS;
    attr->max_qp_wr        = ODL_VERBS_SQ_DEPTH_MAX;
    attr->max_sge          = 1;
    attr->max_cq           = ODL_VERBS_MAX_CQS;
    attr->max_cqe          = ODL_VERBS_COMP_CHANNEL_MAX - 1;
    attr->max_mr           = ODL_VERBS_MAX_MRS;
    attr->max_pd           = ODL_VERBS_MAX_PDS;
    attr->max_mr_size      = SIZE_MAX;

    ODL_TRACE_EXIT_VAL(0);
}

/* ── Legacy compat: ibv_query_port (via _compat_query_port) ─────────── */

static int odl_compat_query_port(struct ibv_context *context,
                                  uint8_t port_num,
                                  struct _compat_ibv_port_attr *attr)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);

    if (port_num != 1) return -EINVAL;

    memset(attr, 0, sizeof(*attr));

    struct odl_tb5_peer_info peer;
    bool connected = (odl_tb5_get_peer(ctx->handle, &peer) == 0 &&
                      peer.state >= ODL_TB5_STATE_CONNECTED);

    attr->max_mtu       = IBV_MTU_4096;
    attr->active_mtu    = IBV_MTU_4096;
    attr->gid_tbl_len   = 1;
    attr->port_cap_flags = IBV_PORT_CM_SUP;
    attr->max_msg_sz    = 1 << 20;
    attr->bad_pkey_cnt  = 0;
    attr->qkey_viol_cnt = 0;
    attr->pkey_tbl_len  = 0;
    attr->lid           = 0;
    attr->sm_lid        = 0;
    attr->lmc           = 0;
    attr->sm_sl         = 0;
    attr->subnet_timeout = 0;
    attr->init_type_reply = 0;

    if (connected) {
        attr->state        = IBV_PORT_ACTIVE;
        attr->phys_state   = 5;
    } else {
        attr->state        = IBV_PORT_DOWN;
        attr->phys_state   = 3;
    }

    ODL_TRACE_EXIT_VAL(0);
}

/* ── Connpletion channel and async event stubs ──────────────────────── */

static struct ibv_comp_channel *odl_compat_create_comp_channel(
    struct ibv_context *context)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(context);
    (void)ctx;
    errno = ENOSYS;
    return NULL;
}

static int odl_compat_destroy_comp_channel(
    struct ibv_comp_channel *channel)
{
    ODL_TRACE_ENTRY();
    (void)channel;
    return 0;
}

/* ── Symbol Interposition ──────────────────────────────────────────────
 *
 * Modern rdma-core (≥ 50) dispatches many ibv_* functions through the
 * internal verbs_context struct rather than ibv_context_ops. Since our
 * standalone library creates ibv_context without a verbs_context wrapper,
 * we intercept these functions via symbol interposition.
 *
 * For manual ODL contexts: dispatch directly to our implementation.
 * For all other contexts: chain to the real libibverbs function.
 *
 * Note: some libibverbs functions are declared as static inline in verbs.h
 * (poll_cq, post_send, post_recv, req_notify_cq). These dispatch through
 * ibv_context_ops which we set up — no interposition needed. Others are
 * real libibverbs symbols that need interposition. Undef them below. */
#undef ibv_query_port
#undef ibv_reg_mr
#undef ibv_dereg_mr
#undef ibv_create_cq
#undef ibv_destroy_cq
#undef ibv_create_qp
#undef ibv_destroy_qp
#undef ibv_modify_qp
#undef ibv_query_qp
#undef ibv_alloc_pd
#undef ibv_dealloc_pd

/* Helper: check if an ibv_context or ibv_pd/ibv_cq/ibv_qp belongs to us */
static bool is_odl_ctx(struct ibv_context *ctx)
{
    return ctx && ctx->device && ctx->device->name &&
           strncmp(ctx->device->name, "odl_tb5_", 8) == 0;
}
static bool is_odl_pd(struct ibv_pd *pd)
{
    return pd && is_odl_ctx(pd->context);
}
static bool is_odl_cq(struct ibv_cq *cq)
{
    return cq && is_odl_ctx(cq->context);
}
static bool is_odl_qp(struct ibv_qp *qp)
{
    return qp && is_odl_ctx(qp->context);
}

/* Resolve a real libibverbs function via dlsym(RTLD_NEXT) */
/* Note: only use for functions that return int and have a matching signature.
 * The typeof() approach fails on ARM64 with some GCC versions. */
static void *resolve_verbs_func(const char *name)
{
    static void *(*dlsym_fn)(void *, const char *) = NULL;
    if (!dlsym_fn) dlsym_fn = dlsym;
    return dlsym_fn(RTLD_NEXT, name);
}

/* ── ibv_close_device ──────────────────────────────────────────────── */

int ibv_close_device(struct ibv_context *context)
{
    ODL_TRACE_ENTRY();
    if (is_odl_ctx(context))
        return odl_free_context(context);
    int (*real_fn)(struct ibv_context *) = resolve_verbs_func("ibv_close_device");
    return real_fn ? real_fn(context) : -ENOSYS;
}

/* ── ibv_query_device ───────────────────────────────────────────────── */

int ibv_query_device(struct ibv_context *context,
                      struct ibv_device_attr *device_attr)
{
    ODL_TRACE_ENTRY();
    if (is_odl_ctx(context))
        return odl_compat_query_device(context, device_attr);
    int (*real_fn)(struct ibv_context *, struct ibv_device_attr *) = resolve_verbs_func("ibv_query_device");
    return real_fn ? real_fn(context, device_attr) : -ENOSYS;
}

/* ── ibv_query_port ─────────────────────────────────────────────────── */

int ibv_query_port(struct ibv_context *context, uint8_t port_num,
                    struct _compat_ibv_port_attr *port_attr)
{
    ODL_TRACE_ENTRY();
    if (is_odl_ctx(context))
        return odl_compat_query_port(context, port_num, port_attr);
    int (*real_fn)(struct ibv_context *, uint8_t, struct _compat_ibv_port_attr *) = resolve_verbs_func("ibv_query_port");
    return real_fn ? real_fn(context, port_num, port_attr) : -ENOSYS;
}

/* ── ibv_alloc_pd / ibv_dealloc_pd ──────────────────────────────────── */

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context)
{
    ODL_TRACE_ENTRY();
    if (is_odl_ctx(context))
        return odl_alloc_pd(context);
    static struct ibv_pd *(*real_fn)(struct ibv_context *);
    if (!real_fn) { real_fn = dlsym(RTLD_NEXT, "ibv_alloc_pd"); }
    return real_fn ? real_fn(context) : NULL;
}

int ibv_dealloc_pd(struct ibv_pd *pd)
{
    ODL_TRACE_ENTRY();
    if (is_odl_pd(pd))
        return odl_dealloc_pd(pd);
    int (*real_fn)(struct ibv_pd *) = resolve_verbs_func("ibv_dealloc_pd");
    return real_fn ? real_fn(pd) : -ENOSYS;
}

/* ── ibv_reg_mr / ibv_dereg_mr ──────────────────────────────────────── */

struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                           int access)
{
    ODL_TRACE_ENTRY();
    if (is_odl_pd(pd))
        return odl_reg_mr(pd, addr, length, 0, access);
    static struct ibv_mr *(*real_fn)(struct ibv_pd *, void *, size_t, int);
    if (!real_fn) { real_fn = dlsym(RTLD_NEXT, "ibv_reg_mr"); }
    return real_fn ? real_fn(pd, addr, length, access) : NULL;
}

int ibv_dereg_mr(struct ibv_mr *mr)
{
    ODL_TRACE_ENTRY();
    if (mr && mr->context && is_odl_ctx(mr->context))
        return odl_dereg_mr(mr);
    int (*real_fn)(struct ibv_mr *) = resolve_verbs_func("ibv_dereg_mr");
    return real_fn ? real_fn(mr) : -ENOSYS;
}

/* ── ibv_create_cq / ibv_destroy_cq ─────────────────────────────────── */

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                              void *cq_context,
                              struct ibv_comp_channel *channel,
                              int comp_vector)
{
    ODL_TRACE_ENTRY();
    if (is_odl_ctx(context))
        return odl_create_cq(context, cqe, channel, comp_vector);
    static struct ibv_cq *(*real_fn)(struct ibv_context *, int, void *,
                                      struct ibv_comp_channel *, int);
    if (!real_fn) { real_fn = dlsym(RTLD_NEXT, "ibv_create_cq"); }
    return real_fn ? real_fn(context, cqe, cq_context, channel, comp_vector) : NULL;
}

int ibv_destroy_cq(struct ibv_cq *cq)
{
    ODL_TRACE_ENTRY();
    if (is_odl_cq(cq))
        return odl_destroy_cq(cq);
    int (*real_fn)(struct ibv_cq *) = resolve_verbs_func("ibv_destroy_cq");
    return real_fn ? real_fn(cq) : -ENOSYS;
}

/* ── ibv_create_qp / ibv_destroy_qp / ibv_modify_qp / ibv_query_qp ──── */

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
                              struct ibv_qp_init_attr *attr)
{
    ODL_TRACE_ENTRY();
    if (is_odl_pd(pd))
        return odl_create_qp(pd, attr);
    static struct ibv_qp *(*real_fn)(struct ibv_pd *, struct ibv_qp_init_attr *);
    if (!real_fn) { real_fn = dlsym(RTLD_NEXT, "ibv_create_qp"); }
    return real_fn ? real_fn(pd, attr) : NULL;
}

int ibv_destroy_qp(struct ibv_qp *qp)
{
    ODL_TRACE_ENTRY();
    if (is_odl_qp(qp))
        return odl_destroy_qp(qp);
    int (*real_fn)(struct ibv_qp *) = resolve_verbs_func("ibv_destroy_qp");
    return real_fn ? real_fn(qp) : -ENOSYS;
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                   int attr_mask)
{
    ODL_TRACE_ENTRY();
    if (is_odl_qp(qp))
        return odl_modify_qp(qp, attr, attr_mask);
    int (*real_fn)(struct ibv_qp *, struct ibv_qp_attr *, int) = resolve_verbs_func("ibv_modify_qp");
    return real_fn ? real_fn(qp, attr, attr_mask) : -ENOSYS;
}

int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                  int attr_mask, struct ibv_qp_init_attr *init_attr)
{
    ODL_TRACE_ENTRY();
    if (is_odl_qp(qp))
        return odl_query_qp(qp, attr, attr_mask, init_attr);
    int (*real_fn)(struct ibv_qp *, struct ibv_qp_attr *, int, struct ibv_qp_init_attr *) = resolve_verbs_func("ibv_query_qp");
    return real_fn ? real_fn(qp, attr, attr_mask, init_attr) : -ENOSYS;
}

/* ── Context Ops Table ─────────────────────────────────────────────── */

void odl_init_context_ops(struct ibv_context *ctx)
{
    /* Legacy _compat_ entries */
    ctx->ops._compat_query_device = odl_compat_query_device;
    ctx->ops._compat_query_port   = odl_compat_query_port;
    ctx->ops._compat_alloc_pd     = (void*)odl_alloc_pd;
    ctx->ops._compat_dealloc_pd   = (void*)odl_dealloc_pd;
    ctx->ops._compat_reg_mr       = (void*)odl_reg_mr;
    ctx->ops._compat_dereg_mr     = (void*)odl_dereg_mr;
    ctx->ops._compat_create_cq    = (void*)odl_create_cq;
    ctx->ops._compat_destroy_cq   = (void*)odl_destroy_cq;
    ctx->ops._compat_resize_cq    = NULL;
    ctx->ops._compat_cq_event     = (void*)odl_cq_event;
    ctx->ops._compat_create_qp    = (void*)odl_create_qp;
    ctx->ops._compat_destroy_qp   = (void*)odl_destroy_qp;
    ctx->ops._compat_modify_qp    = (void*)odl_modify_qp;
    ctx->ops._compat_query_qp     = (void*)odl_query_qp;
    ctx->ops._compat_create_srq   = NULL;
    ctx->ops._compat_modify_srq   = NULL;
    ctx->ops._compat_query_srq    = NULL;
    ctx->ops._compat_destroy_srq  = NULL;
    ctx->ops._compat_create_ah    = NULL;
    ctx->ops._compat_destroy_ah   = NULL;
    ctx->ops._compat_attach_mcast  = NULL;
    ctx->ops._compat_detach_mcast  = NULL;
    ctx->ops._compat_async_event  = NULL;
    ctx->ops._compat_rereg_mr     = NULL;

    /* Non-compat entries (used by poll_cq, post_send, etc.) */
    ctx->ops.poll_cq         = odl_poll_cq;
    ctx->ops.req_notify_cq   = odl_req_notify_cq;
    ctx->ops.post_send       = odl_post_send;
    ctx->ops.post_recv       = odl_post_recv;
    ctx->ops.post_srq_recv   = NULL;
    ctx->ops.alloc_mw        = NULL;
    ctx->ops.dealloc_mw      = NULL;
    ctx->ops.bind_mw         = NULL;
}
