// SPDX-License-Identifier: MIT
/*
 * OdinLink Thunderbolt 5 - Login/Logout Handshake Protocol
 *
 * Implements the XDomain login/logout handshake between two OdinLink
 * peers.  The protocol follows the same pattern as thunderbolt-net
 * (drivers/net/thunderbolt.c):
 *
 *   1. Both sides advertise the "odinlink" protocol via their property
 *      directory so that XDomain discovery matches the service.
 *   2. Both sides schedule a login work item that sends a login request
 *      via tb_xdomain_request() and waits for a response.
 *   3. A global protocol handler receives incoming login requests from
 *      the peer and sends back a login response.
 *   4. When a side has both sent and received a successful login the
 *      XDomain paths are enabled, DMA rings are started, and the
 *      connection enters the CONNECTED state.
 *
 * Part of the odl_tb5.ko multi-file module alongside:
 *   odl_tb5_service.c   - Thunderbolt service probe / remove
 *   odl_tb5_ring_dma.c  - NHI ring allocation, DMA buffer management
 *   odl_tb5_chardev.c   - Character device (ioctl / mmap interface)
 */

#include "odl_tb5_core.h"
#include <linux/delay.h>

#define ODL_TB5_MSG_LOGIN      1
#define ODL_TB5_MSG_LOGIN_RSP  2
#define ODL_TB5_MSG_LOGOUT     3

#define ODL_TB5_LOGIN_TIMEOUT  500
#define ODL_TB5_ENABLE_RETRIES 5
#define ODL_TB5_ENABLE_DELAY   200

struct odl_tb5_xd_header {
	u32	route_hi;
	u32	route_lo;
	u32	length_sn;
	uuid_t	uuid;
	u32	type;
};

struct odl_tb5_login_msg {
	struct odl_tb5_xd_header xd_hdr;
	u32 proto_version;
	u32 transmit_path;
	u32 reserved[2];
};

struct odl_tb5_login_response {
	struct odl_tb5_xd_header xd_hdr;
	u32 status;
	u32 transmit_path;
	u32 reserved[2];
};

struct odl_tb5_logout_msg {
	struct odl_tb5_xd_header xd_hdr;
};

static void odl_tb5_login_work_fn(struct work_struct *work);
static void odl_tb5_connect_work_fn(struct work_struct *work);
static void odl_tb5_restart_work_fn(struct work_struct *work);
static int  odl_tb5_complete_connection(struct odl_tb5_device *dev);

#define XD_HDR_SIZE_DW  3
#define XD_SN_MASK      0x18000000u

static void odl_tb5_xd_header_init(struct odl_tb5_xd_header *hdr,
				    struct tb_xdomain *xd, u32 type,
				    size_t total_size)
{
	hdr->route_hi  = upper_32_bits(xd->route);
	hdr->route_lo  = lower_32_bits(xd->route);
	hdr->length_sn = total_size / 4 - XD_HDR_SIZE_DW;
	hdr->uuid      = odl_tb5_proto_uuid;
	hdr->type      = type;
}

static struct odl_tb5_device *odl_tb5_find_device_by_route(u64 route)
{
	struct odl_tb5_device *dev;

	list_for_each_entry(dev, &odl_tb5_devices_list, list) {
		if (dev->xd->route == route)
			return dev;
	}
	return NULL;
}

static int odl_tb5_proto_handle_packet(const void *buf, size_t size,
				       void *data)
{
	const struct odl_tb5_xd_header *hdr = buf;
	struct odl_tb5_device *dev;
	u64 route;
	bool need_complete = false;

	if (size < sizeof(*hdr))
		return 0;

	route = (((u64)hdr->route_hi << 32) | hdr->route_lo) & ~BIT_ULL(63);

	mutex_lock(&odl_tb5_devices_lock);
	dev = odl_tb5_find_device_by_route(route);
	mutex_unlock(&odl_tb5_devices_lock);

	if (!dev) {
		pr_warn("OdinLink: incoming packet route %llx — no matching device\n",
			route);
		return 0;
	}

	switch (hdr->type) {
	case ODL_TB5_MSG_LOGIN: {
		const struct odl_tb5_login_msg *pkg = buf;
		struct odl_tb5_login_response resp = { };
		int ret;

		if (size < sizeof(*pkg))
			return 0;

		pr_info("OdinLink: received login from peer "
			"(version=%u, tx_path=%u)\n",
			pkg->proto_version, pkg->transmit_path);

		resp.xd_hdr.route_hi  = upper_32_bits(dev->xd->route);
		resp.xd_hdr.route_lo  = lower_32_bits(dev->xd->route);
		resp.xd_hdr.length_sn =
			(hdr->length_sn & XD_SN_MASK) |
			(sizeof(resp) / 4 - XD_HDR_SIZE_DW);
		resp.xd_hdr.uuid      = odl_tb5_proto_uuid;
		resp.xd_hdr.type      = ODL_TB5_MSG_LOGIN_RSP;
		resp.status            = 0;
		resp.transmit_path     = dev->local_tx_hopid;

		ret = tb_xdomain_response(dev->xd, &resp, sizeof(resp),
					  TB_CFG_PKG_XDOMAIN_RESP);
		pr_info("OdinLink: sent login response (ret=%d, route=%llx, "
			"sn=%u, tx_hopid=%d)\n",
			ret, dev->xd->route,
			(hdr->length_sn & XD_SN_MASK) >> 27,
			dev->local_tx_hopid);

		mutex_lock(&dev->state_lock);
		if (dev->state != ODL_TB5_STATE_HANDSHAKE) {
			pr_info("OdinLink: peer restarted (our state=%d), "
				"scheduling restart\n", dev->state);
			dev->stale_remote_tx_hopid = dev->remote_tx_hopid;
			dev->remote_tx_hopid = pkg->transmit_path;
			dev->login_received = true;
			mutex_unlock(&dev->state_lock);
			schedule_work(&dev->restart_work);
			return 1;
		}

		dev->remote_tx_hopid = pkg->transmit_path;
		dev->login_received = true;
		if (dev->login_sent)
			need_complete = true;
		mutex_unlock(&dev->state_lock);

		if (need_complete)
			schedule_work(&dev->connect_work);

		return 1;
	}

	case ODL_TB5_MSG_LOGOUT:
		pr_info("OdinLink: received logout from peer\n");

		mutex_lock(&dev->state_lock);
		dev->stale_remote_tx_hopid = dev->remote_tx_hopid;
		dev->login_received = false;
		dev->login_sent = false;
		mutex_unlock(&dev->state_lock);

		schedule_work(&dev->restart_work);

		return 1;

	default:
		return 0;
	}
}

static struct tb_protocol_handler odl_tb5_handler = {
	.uuid     = &odl_tb5_proto_uuid,
	.callback = odl_tb5_proto_handle_packet,
};

int odl_tb5_proto_register(void)
{
	return tb_register_protocol_handler(&odl_tb5_handler);
}

void odl_tb5_proto_unregister(void)
{
	tb_unregister_protocol_handler(&odl_tb5_handler);
}

/* Send a login request to the peer and process the response. */
int odl_tb5_proto_send_login(struct odl_tb5_device *dev)
{
	struct odl_tb5_login_msg msg = { };
	struct odl_tb5_login_response resp = { };
	bool need_complete = false;
	int ret;

	odl_tb5_xd_header_init(&msg.xd_hdr, dev->xd, ODL_TB5_MSG_LOGIN,
			       sizeof(msg));
	msg.proto_version = ODL_TB5_PROTOCOL_VER;
	msg.transmit_path = dev->local_tx_hopid;

	ret = tb_xdomain_request(dev->xd, &msg, sizeof(msg),
				 TB_CFG_PKG_XDOMAIN_REQ,
				 &resp, sizeof(resp),
				 TB_CFG_PKG_XDOMAIN_RESP,
				 ODL_TB5_LOGIN_TIMEOUT);
	if (ret) {
		pr_warn("OdinLink: login request failed: %d\n", ret);
		return ret;
	}

	if (!uuid_equal(&resp.xd_hdr.uuid, &odl_tb5_proto_uuid)) {
		pr_warn("OdinLink: login response UUID mismatch\n");
		return -EPROTO;
	}

	if (resp.xd_hdr.type != ODL_TB5_MSG_LOGIN_RSP) {
		pr_warn("OdinLink: unexpected response type %u\n",
			resp.xd_hdr.type);
		return -EPROTO;
	}

	if (resp.status != 0) {
		pr_warn("OdinLink: peer rejected login with status %u\n",
			resp.status);
		return -ECONNREFUSED;
	}

	mutex_lock(&dev->state_lock);
	dev->remote_tx_hopid = resp.transmit_path;
	dev->login_sent = true;
	if (dev->login_received && dev->state == ODL_TB5_STATE_HANDSHAKE)
		need_complete = true;
	mutex_unlock(&dev->state_lock);

	pr_info("OdinLink: login sent OK, remote_tx_hopid=%d\n",
		dev->remote_tx_hopid);

	if (need_complete)
		schedule_work(&dev->connect_work);

	return 0;
}

/* Finish handshake and bring link up. */
static int odl_tb5_complete_connection(struct odl_tb5_device *dev)
{
	int ret, i;

	ret = tb_xdomain_alloc_in_hopid(dev->xd, dev->remote_tx_hopid);
	if (ret < 0) {
		pr_err("OdinLink: failed to allocate input HopID: %d\n", ret);
		return ret;
	}
	ret = odl_tb5_rings_start(dev);
	if (ret) {
		pr_err("OdinLink: failed to start rings: %d\n", ret);
		tb_xdomain_release_in_hopid(dev->xd, dev->remote_tx_hopid);
		return ret;
	}

	{
		size_t rx_prime = (size_t)ODL_TB5_FRAME_SIZE * 16;

		ret = odl_tb5_submit_rx(dev, 0, rx_prime);
		if (ret) {
			pr_err("OdinLink: failed to prime RX: %d\n", ret);
			odl_tb5_rings_stop(dev);
			tb_xdomain_release_in_hopid(dev->xd,
						    dev->remote_tx_hopid);
			return ret;
		}
		pr_info("OdinLink: RX primed with 16 frames before "
			"enable_paths\n");
	}

	for (i = 0; i < ODL_TB5_ENABLE_RETRIES; i++) {
		ret = tb_xdomain_enable_paths(dev->xd,
					      dev->local_tx_hopid,
					      dev->tx.ring->hop,
					      dev->remote_tx_hopid,
					      dev->rx.ring->hop);
		if (!ret)
			break;

		if (i < ODL_TB5_ENABLE_RETRIES - 1) {
			pr_warn("OdinLink: enable_paths failed (%d), "
				"retry %d/%d in %d ms\n",
				ret, i + 1, ODL_TB5_ENABLE_RETRIES,
				ODL_TB5_ENABLE_DELAY);
			msleep(ODL_TB5_ENABLE_DELAY);
		}
	}

	if (ret) {
		pr_err("OdinLink: failed to enable XDomain paths "
		       "after %d attempts: %d\n",
		       ODL_TB5_ENABLE_RETRIES, ret);
		odl_tb5_rings_stop(dev);
		tb_xdomain_release_in_hopid(dev->xd, dev->remote_tx_hopid);
		return ret;
	}

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_CONNECTED;
	mutex_unlock(&dev->state_lock);

	pr_info("OdinLink: connected to peer "
		"(local_tx_hopid=%d, remote_tx_hopid=%d, "
		"tx_ring_hop=%d, rx_ring_hop=%d, "
		"ring_size=%d, no E2E)\n",
		dev->local_tx_hopid, dev->remote_tx_hopid,
		dev->tx.ring->hop, dev->rx.ring->hop,
		dev->tx.ring_size);

	schedule_delayed_work(&dev->rx_poll_work, msecs_to_jiffies(1));
	schedule_work(&dev->verify_work);

	return 0;
}

/* Deferred connection completion in safe work context. */
static void odl_tb5_connect_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, connect_work);
	int ret;

	ret = odl_tb5_complete_connection(dev);
	if (ret) {
		mutex_lock(&dev->state_lock);
		dev->login_sent = false;
		dev->login_received = false;
		mutex_unlock(&dev->state_lock);

		pr_warn("OdinLink: connection completion failed (%d), "
			"retrying handshake\n", ret);
		schedule_delayed_work(&dev->login_work,
				      msecs_to_jiffies(1000));
	}
}

/* Write a kernel DMA control message (ping or pong) via frame pool. */
static int odl_tb5_send_dma_msg(struct odl_tb5_device *dev, u32 type)
{
	struct odl_tb5_frame_slot *slot;
	struct odl_tb5_stream_hdr *shdr;
	struct odl_tb5_dma_hdr *dhdr;
	int ret;

	/* Try frame pool path first (new stream model) */
	if (dev->frame_pool.slots) {
		slot = odl_tb5_frame_pool_get(&dev->frame_pool);
		if (!slot)
			return -ENOMEM;

		shdr = slot->virt;
		shdr->src_id = ODL_TB5_STREAM_ID_CTRL;
		shdr->dst_id = ODL_TB5_STREAM_ID_CTRL;
		shdr->flags  = ODL_TB5_SHDR_F_SINGLE;
		shdr->payload_len = cpu_to_le16(sizeof(*dhdr));

		dhdr = slot->virt + ODL_TB5_STREAM_HDR_SIZE;
		memset(dhdr, 0, sizeof(*dhdr));
		dhdr->magic = cpu_to_le32(ODL_TB5_DMA_MAGIC);
		dhdr->type  = cpu_to_le32(type);

		slot->frame.size = ODL_TB5_STREAM_HDR_SIZE + sizeof(*dhdr);
		slot->frame.sof = ODL_TB5_PDF_SOF_CTRL;
		slot->frame.eof = ODL_TB5_PDF_EOF_CTRL;
		slot->frame.callback = odl_tb5_tx_callback;
		slot->tx_msg = NULL;

		ret = tb_ring_tx(dev->tx.ring, &slot->frame);
		if (ret < 0) {
			odl_tb5_frame_pool_put(&dev->frame_pool, slot);
			return ret;
		}

		return 0;
	}

	/* Legacy fallback (before frame pool is allocated) */
	{
		struct odl_tb5_dma_hdr *hdr;

		hdr = dev->tx.bufs[dev->tx.front].virt;
		memset(hdr, 0, sizeof(*hdr));
		hdr->magic = cpu_to_le32(ODL_TB5_DMA_MAGIC);
		hdr->type  = cpu_to_le32(type);

		return odl_tb5_submit_tx(dev, 0, sizeof(*hdr), true);
	}
}

/* Respond to incoming DMA PING messages with a PONG. */
static void odl_tb5_ctrl_reply_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, ctrl_reply_work);
	int type = dev->verify_rx_type;

	if (type == ODL_TB5_DMA_PING) {
		pr_info("OdinLink: DMA ping received, sending pong\n");
		odl_tb5_send_dma_msg(dev, ODL_TB5_DMA_PONG);
	} else {
		pr_warn("OdinLink: unexpected DMA ctrl message type %d\n",
			type);
	}
}


/* Post-connection DMA verification via ping/pong exchange. */
static void odl_tb5_verify_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, verify_work);
	size_t buf_size;
	long ret;
	int attempt;

	dev->pong_received = false;

	buf_size = (size_t)ODL_TB5_FRAME_SIZE * 16;
	ret = odl_tb5_submit_rx(dev, 0, buf_size);
	if (ret) {
		pr_warn("OdinLink: DMA verify: failed to post RX (%ld)\n",
			ret);
		goto out_reset;
	}

	for (attempt = 0; ; attempt++) {
		if (dev->state != ODL_TB5_STATE_CONNECTED)
			goto out_reset;

		flush_work(&dev->ctrl_reply_work);

		ret = odl_tb5_send_dma_msg(dev, ODL_TB5_DMA_PING);
		if (ret) {
			pr_warn("OdinLink: DMA verify: send ping failed "
				"(%ld)\n", ret);
			goto out_reset;
		}

		if (attempt == 0)
			pr_info("OdinLink: DMA ping sent, "
				"waiting for pong\n");

		ret = wait_event_interruptible_timeout(
				dev->verify_waitq,
				dev->pong_received,
				msecs_to_jiffies(1000));
		if (ret > 0)
			break;

		if (ret < 0)
			goto out_reset;

		if ((attempt + 1) % 10 == 0)
			pr_info("OdinLink: DMA ping attempt %d, "
				"still waiting for pong\n", attempt + 1);

		flush_delayed_work(&dev->rx_poll_work);
		tb_ring_stop(dev->rx.ring);
		tb_ring_start(dev->rx.ring);
		dev->rx.frames_posted = false;
		dev->rx.swapped_since_post = false;
		atomic_set(&dev->rx.completed, 0);
		atomic_set(&dev->rx.submitted, 0);

		ret = odl_tb5_submit_rx(dev, 0, buf_size);
		if (ret) {
			pr_warn("OdinLink: DMA verify: re-post RX failed "
				"(%ld)\n", ret);
			goto out_reset;
		}
	}

	pr_info("OdinLink: DMA path verified, resetting rings for userspace\n");

	flush_work(&dev->ctrl_reply_work);
	cancel_delayed_work_sync(&dev->rx_poll_work);
	odl_tb5_rings_reset(dev);

	/* Allocate frame pool for stream multiplexing */
	if (!dev->frame_pool.slots) {
		int pool_ret = odl_tb5_frame_pool_alloc(dev);

		if (pool_ret)
			pr_warn("OdinLink: frame pool alloc failed (%d), "
				"streams unavailable\n", pool_ret);
	}

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_READY;
	mutex_unlock(&dev->state_lock);

	/*
	 * Don't post pool frames yet — legacy consumers (daemon, CLI)
	 * don't use stream headers.  Pool RX repost starts when the
	 * first stream is opened via STREAM_OPEN ioctl.
	 */
	dev->rx_target = 0;
	atomic_set(&dev->rx_posted, 0);

	pr_info("OdinLink: entering READY state\n");
	return;

out_reset:
	cancel_delayed_work_sync(&dev->rx_poll_work);
	odl_tb5_rings_reset(dev);
}

/* Tear down stale connection and restart the handshake. */
static void odl_tb5_restart_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, restart_work);

	cancel_delayed_work_sync(&dev->rx_poll_work);
	cancel_work_sync(&dev->verify_work);
	cancel_work_sync(&dev->ctrl_reply_work);
	cancel_work_sync(&dev->connect_work);
	cancel_delayed_work_sync(&dev->login_work);
	dev->pong_received = false;

	if (dev->tx.started) {
		tb_xdomain_disable_paths(dev->xd,
					 dev->local_tx_hopid,
					 dev->tx.ring->hop,
					 dev->stale_remote_tx_hopid,
					 dev->rx.ring->hop);
		odl_tb5_rings_stop(dev);
		tb_xdomain_release_in_hopid(dev->xd,
					    dev->stale_remote_tx_hopid);
	}

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_HANDSHAKE;
	dev->login_sent = false;
	dev->login_retries = 0;
	mutex_unlock(&dev->state_lock);

	pr_info("OdinLink: connection restarted, beginning handshake\n");
	schedule_delayed_work(&dev->login_work, 0);
}

/* Delayed work handler that retries login with exponential backoff. */
static void odl_tb5_login_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, login_work.work);
	unsigned long delay_ms;
	int ret;

	ret = odl_tb5_proto_send_login(dev);
	if (ret) {
		dev->login_retries++;

		delay_ms = ODL_TB5_LOGIN_TIMEOUT <<
			   min_t(int, dev->login_retries, 4);
		if (delay_ms > 5000)
			delay_ms = 5000;

		if (dev->login_retries <= 5 ||
		    dev->login_retries % 10 == 0)
			pr_info("OdinLink: login attempt %d failed (%d), "
				"retrying in %lu ms\n",
				dev->login_retries, ret, delay_ms);

		schedule_delayed_work(&dev->login_work,
				      msecs_to_jiffies(delay_ms));
	}
}

/* Send a logout notification to the peer. */
int odl_tb5_proto_send_logout(struct odl_tb5_device *dev)
{
	struct odl_tb5_logout_msg msg = { };
	struct odl_tb5_xd_header resp = { };

	odl_tb5_xd_header_init(&msg.xd_hdr, dev->xd, ODL_TB5_MSG_LOGOUT,
			       sizeof(msg));

	tb_xdomain_request(dev->xd, &msg, sizeof(msg),
			   TB_CFG_PKG_XDOMAIN_REQ,
			   &resp, sizeof(resp),
			   TB_CFG_PKG_XDOMAIN_RESP,
			   ODL_TB5_LOGIN_TIMEOUT);

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_DISCONNECTED;
	mutex_unlock(&dev->state_lock);

	pr_info("OdinLink: logout sent to peer\n");

	return 0;
}

/* Initialise the protocol layer for a device. */
int odl_tb5_proto_init(struct odl_tb5_device *dev)
{
	INIT_DELAYED_WORK(&dev->login_work, odl_tb5_login_work_fn);
	INIT_WORK(&dev->connect_work, odl_tb5_connect_work_fn);
	INIT_WORK(&dev->restart_work, odl_tb5_restart_work_fn);
	INIT_WORK(&dev->verify_work, odl_tb5_verify_work_fn);
	INIT_WORK(&dev->ctrl_reply_work, odl_tb5_ctrl_reply_work_fn);
	INIT_DELAYED_WORK(&dev->rx_poll_work, odl_tb5_rx_poll_work_fn);
	init_waitqueue_head(&dev->verify_waitq);

	dev->login_retries  = 0;
	dev->login_sent     = false;
	dev->login_received = false;
	dev->pong_received  = false;

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_HANDSHAKE;
	mutex_unlock(&dev->state_lock);

	schedule_delayed_work(&dev->login_work, 0);

	return 0;
}

/* Tear down the protocol layer for a device. */
void odl_tb5_proto_exit(struct odl_tb5_device *dev)
{
	cancel_delayed_work_sync(&dev->rx_poll_work);
	cancel_work_sync(&dev->verify_work);
	cancel_work_sync(&dev->ctrl_reply_work);
	cancel_work_sync(&dev->restart_work);
	cancel_work_sync(&dev->connect_work);
	cancel_delayed_work_sync(&dev->login_work);
}
