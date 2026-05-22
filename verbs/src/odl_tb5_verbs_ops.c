/*
 * OdinLink Verbs Provider — ibv_context Ops Dispatch Table
 *
 * Sets up the ibv_context.ops table that libibverbs dispatches to.
 * Both legacy (_compat_*) and modern dispatch entries are filled in.
 */

#include "odl_tb5_verbs.h"

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
    attr->max_qp_wr        = ODL_VERBS_SQ_DEPTH;
    attr->max_sge          = 1;
    attr->max_cq           = ODL_VERBS_MAX_CQS;
    attr->max_cqe          = ODL_VERBS_COMP_CHANNEL_BACKLOG;
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
