// SPDX-License-Identifier: MIT
/*
 * OdinLink — Intel NHI Transport Backend
 *
 * The original transport: Intel's Thunderbolt Host Interface (NHI).
 * This backend wraps the Linux kernel's tb_ring API and XDomain path
 * management into the odl_tb5_transport_ops interface so the rest of
 * the driver doesn't touch NHI internals directly.
 *
 * When a future Apple-Silicon backend is written, it will implement
 * the same ops struct and the driver core won't need to change.
 */

#include "odl_tb5_core.h"
#include "odl_tb5_xd_proto.h"
#include <linux/dma-mapping.h>

struct nhi_priv {
	struct tb_xdomain	*xd;
	int			local_tx_hopid;
};

static inline struct nhi_priv *nhi_priv(struct odl_tb5_device *dev)
{
	return dev->transport_priv;
}

/* ── Ring alloc/free ────────────────────────────────────────────────── */

static int nhi_ring_alloc(struct odl_tb5_device *dev)
{
	struct tb_xdomain *xd = nhi_priv(dev)->xd;
	unsigned int sof_mask, eof_mask;
	unsigned int rs = odl_ring_size;
	int ret;

	if (rs < ODL_TB5_RING_SIZE_MIN)
		rs = ODL_TB5_RING_SIZE_MIN;
	if (rs > ODL_TB5_RING_SIZE_MAX)
		rs = ODL_TB5_RING_SIZE_MAX;
	rs = roundup_pow_of_two(rs);

	dev->tx.ring_size = rs;
	dev->rx.ring_size = rs;

	pr_info("odl_tb5: ring_size=%u (%u MB per batch, %u MB total)\n",
		rs,
		(rs * ODL_TB5_FRAME_SIZE) >> 20,
		(rs * ODL_TB5_FRAME_SIZE * ODL_TB5_NUM_BUFFERS * 2) >> 20);

	dev->tx.frames = kvzalloc(rs * sizeof(struct ring_frame), GFP_KERNEL);
	if (!dev->tx.frames)
		return -ENOMEM;

	dev->rx.frames = kvzalloc(rs * sizeof(struct ring_frame), GFP_KERNEL);
	if (!dev->rx.frames) {
		ret = -ENOMEM;
		goto err_free_tx_frames;
	}

	ret = tb_xdomain_alloc_out_hopid(xd, -1);
	if (ret < 0) {
		pr_err("odl_tb5: failed to allocate output HopID: %d\n", ret);
		goto err_free_rx_frames;
	}
	nhi_priv(dev)->local_tx_hopid = ret;

	unsigned int ring_flags = RING_FLAG_FRAME;
	if (odl_e2e)
		ring_flags |= RING_FLAG_E2E;

	dev->tx.ring_handle = tb_ring_alloc_tx(xd->tb->nhi, -1, rs,
					       ring_flags);
	if (!dev->tx.ring_handle) {
		pr_err("odl_tb5: failed to allocate TX ring\n");
		ret = -ENOMEM;
		goto err_free_hopid;
	}

	sof_mask = BIT(ODL_TB5_PDF_SOF_DATA);
	eof_mask = BIT(ODL_TB5_PDF_EOF_DATA);

	dev->rx.ring_handle = tb_ring_alloc_rx(xd->tb->nhi, -1, rs,
					       ring_flags,
					       ((struct tb_ring *)dev->tx.ring_handle)->hop,
					       sof_mask, eof_mask,
					       NULL, NULL);
	if (!dev->rx.ring_handle) {
		pr_err("odl_tb5: failed to allocate RX ring\n");
		ret = -ENOMEM;
		goto err_free_tx_ring;
	}

	pr_info("odl_tb5: rings allocated: TX hop=%d, RX hop=%d, "
		"local_tx_hopid=%d (E2E enabled, e2e_tx_hop=%d)\n",
		((struct tb_ring *)dev->tx.ring_handle)->hop,
		((struct tb_ring *)dev->rx.ring_handle)->hop,
		nhi_priv(dev)->local_tx_hopid,
		((struct tb_ring *)dev->tx.ring_handle)->hop);

	return 0;

err_free_tx_ring:
	tb_ring_free(dev->tx.ring_handle);
	dev->tx.ring_handle = NULL;
err_free_hopid:
	tb_xdomain_release_out_hopid(xd, nhi_priv(dev)->local_tx_hopid);
	nhi_priv(dev)->local_tx_hopid = -1;
err_free_rx_frames:
	kvfree(dev->rx.frames);
	dev->rx.frames = NULL;
err_free_tx_frames:
	kvfree(dev->tx.frames);
	dev->tx.frames = NULL;
	return ret;
}

static void nhi_ring_free(struct odl_tb5_device *dev)
{
	if (dev->rx.ring_handle) {
		tb_ring_free(dev->rx.ring_handle);
		dev->rx.ring_handle = NULL;
	}

	if (dev->tx.ring_handle) {
		tb_ring_free(dev->tx.ring_handle);
		dev->tx.ring_handle = NULL;
	}

	if (nhi_priv(dev)->local_tx_hopid >= 0) {
		tb_xdomain_release_out_hopid(nhi_priv(dev)->xd,
					     nhi_priv(dev)->local_tx_hopid);
		nhi_priv(dev)->local_tx_hopid = -1;
	}

	kvfree(dev->tx.frames);
	dev->tx.frames = NULL;
	kvfree(dev->rx.frames);
	dev->rx.frames = NULL;
}

/* ── Ring start/stop/reset ─────────────────────────────────────────── */

static int nhi_ring_start(struct odl_tb5_device *dev)
{
	tb_ring_start(dev->tx.ring_handle);
	tb_ring_start(dev->rx.ring_handle);
	return 0;
}

static void nhi_ring_stop(struct odl_tb5_device *dev)
{
	if (dev->tx.ring_handle && dev->tx.started)
		tb_ring_stop(dev->tx.ring_handle);

	if (dev->rx.ring_handle && dev->rx.started)
		tb_ring_stop(dev->rx.ring_handle);
}

static void nhi_ring_reset(struct odl_tb5_device *dev)
{
	if (dev->tx.ring_handle && dev->tx.started) {
		tb_ring_stop(dev->tx.ring_handle);
		tb_ring_start(dev->tx.ring_handle);
	}

	if (dev->rx.ring_handle && dev->rx.started) {
		tb_ring_stop(dev->rx.ring_handle);
		tb_ring_start(dev->rx.ring_handle);
	}
}

/* ── Frame submit ──────────────────────────────────────────────────── */

static int nhi_ring_tx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	return tb_ring_tx(dev->tx.ring_handle, frame);
}

static int nhi_ring_rx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	return tb_ring_rx(dev->rx.ring_handle, frame);
}

/* ── DMA device access ─────────────────────────────────────────────── */

static struct device *nhi_dma_device(struct odl_tb5_device *dev)
{
	return tb_ring_dma_device(dev->tx.ring_handle);
}

static struct odl_tb5_transport_ring_info nhi_tx_ring_info(
	struct odl_tb5_device *dev)
{
	struct odl_tb5_transport_ring_info info = { .hop = -1 };
	if (dev->tx.ring_handle)
		info.hop = ((struct tb_ring *)dev->tx.ring_handle)->hop;
	return info;
}

static struct odl_tb5_transport_ring_info nhi_rx_ring_info(
	struct odl_tb5_device *dev)
{
	struct odl_tb5_transport_ring_info info = { .hop = -1 };
	if (dev->rx.ring_handle)
		info.hop = ((struct tb_ring *)dev->rx.ring_handle)->hop;
	return info;
}

static int nhi_local_tx_hopid(struct odl_tb5_device *dev)
{
	return nhi_priv(dev)->local_tx_hopid;
}

/* ── Path management ───────────────────────────────────────────────── */

static int nhi_path_enable(struct odl_tb5_device *dev)
{
	struct tb_xdomain *xd = nhi_priv(dev)->xd;
	struct odl_tb5_transport_ring_info tx_info = nhi_tx_ring_info(dev);
	struct odl_tb5_transport_ring_info rx_info = nhi_rx_ring_info(dev);
	int ret, i;

	ret = tb_xdomain_alloc_in_hopid(xd, dev->remote_tx_hopid);
	if (ret < 0)
		return ret;

	for (i = 0; i < ODL_TB5_ENABLE_RETRIES; i++) {
		ret = tb_xdomain_enable_paths(xd,
					      nhi_priv(dev)->local_tx_hopid,
					      tx_info.hop,
					      dev->remote_tx_hopid,
					      rx_info.hop);
		if (!ret)
			return 0;
		if (i < ODL_TB5_ENABLE_RETRIES - 1)
			msleep(ODL_TB5_ENABLE_DELAY);
	}

	tb_xdomain_release_in_hopid(xd, dev->remote_tx_hopid);
	return ret;
}

static void nhi_path_disable(struct odl_tb5_device *dev)
{
	struct tb_xdomain *xd = nhi_priv(dev)->xd;
	struct odl_tb5_transport_ring_info tx_info = nhi_tx_ring_info(dev);
	struct odl_tb5_transport_ring_info rx_info = nhi_rx_ring_info(dev);

	if (dev->tx.started || dev->rx.started) {
		tb_xdomain_disable_paths(xd,
					 nhi_priv(dev)->local_tx_hopid,
					 tx_info.hop,
					 dev->remote_tx_hopid,
					 rx_info.hop);
		tb_xdomain_release_in_hopid(xd, dev->remote_tx_hopid);
	}
}

/* ── Login/Logout (XDomain) ───────────────────────────────────────── */

static int nhi_peer_send_login(struct odl_tb5_device *dev)
{
	struct odl_tb5_login_msg msg = { };
	struct odl_tb5_login_response resp = { };
	int ret;

	odl_tb5_xd_header_init(&msg.xd_hdr, nhi_priv(dev)->xd,
			       ODL_TB5_MSG_LOGIN, sizeof(msg));
	msg.proto_version = ODL_TB5_PROTOCOL_VER;
	msg.transmit_path = nhi_priv(dev)->local_tx_hopid;

	ret = tb_xdomain_request(nhi_priv(dev)->xd,
				 &msg, sizeof(msg),
				 TB_CFG_PKG_XDOMAIN_REQ,
				 &resp, sizeof(resp),
				 TB_CFG_PKG_XDOMAIN_RESP,
				 ODL_TB5_LOGIN_TIMEOUT);
	if (ret)
		return ret;

	if (odl_protocol_mode == 1) {
		dev->remote_tx_hopid = resp.transmit_path;
		return 0;
	}

	if (!uuid_equal(&resp.xd_hdr.uuid, &odl_tb5_proto_uuid))
		return -EPROTO;

	if (resp.xd_hdr.type != ODL_TB5_MSG_LOGIN_RSP)
		return -EPROTO;

	if (resp.status != 0)
		return -ECONNREFUSED;

	dev->remote_tx_hopid = resp.transmit_path;
	return 0;
}

static int nhi_peer_send_logout(struct odl_tb5_device *dev)
{
	struct odl_tb5_logout_msg msg = { };
	struct odl_tb5_xd_header resp = { };

	odl_tb5_xd_header_init(&msg.xd_hdr, nhi_priv(dev)->xd,
			       ODL_TB5_MSG_LOGOUT, sizeof(msg));

	tb_xdomain_request(nhi_priv(dev)->xd, &msg, sizeof(msg),
			   TB_CFG_PKG_XDOMAIN_REQ,
			   &resp, sizeof(resp),
			   TB_CFG_PKG_XDOMAIN_RESP,
			   ODL_TB5_LOGIN_TIMEOUT);
	return 0;
}

/* ── Kick (for hrtimer poll) ───────────────────────────────────────── */

static void nhi_kick_tx(struct odl_tb5_device *dev)
{
	if (dev->tx.ring_handle && dev->tx.started)
		schedule_work(&((struct tb_ring *)dev->tx.ring_handle)->work);
}

static void nhi_kick_rx(struct odl_tb5_device *dev)
{
	if (dev->rx.ring_handle && dev->rx.started)
		schedule_work(&((struct tb_ring *)dev->rx.ring_handle)->work);
}

/* ── Ops table ─────────────────────────────────────────────────────── */

const struct odl_tb5_transport_ops odl_tb5_nhi_transport = {
	.type		= ODL_TB5_TRANSPORT_NHI,
	.name		= "nhi",
	.ring_alloc	= nhi_ring_alloc,
	.ring_free	= nhi_ring_free,
	.ring_start	= nhi_ring_start,
	.ring_stop	= nhi_ring_stop,
	.ring_reset	= nhi_ring_reset,
	.ring_tx	= nhi_ring_tx,
	.ring_rx	= nhi_ring_rx,
	.dma_device	= nhi_dma_device,
	.tx_ring_info	= nhi_tx_ring_info,
	.rx_ring_info	= nhi_rx_ring_info,
	.local_tx_hopid = nhi_local_tx_hopid,
	.path_enable	= nhi_path_enable,
	.path_disable	= nhi_path_disable,
	.peer_send_login  = nhi_peer_send_login,
	.peer_send_logout = nhi_peer_send_logout,
	.kick_tx	= nhi_kick_tx,
	.kick_rx	= nhi_kick_rx,
};
