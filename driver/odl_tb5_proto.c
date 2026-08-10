// SPDX-License-Identifier: MIT
/*
 * OdinLink — The Handshake: How Two Machines Agree to Talk
 *
 * Before any data moves, both sides need to agree on which DMA slots to
 * use. This is the XDomain login/logout protocol:
 *
 *   1. Each machine advertises its protocol ID ("OdinLink" = 0x4F4C, or
 *      "Apple" = 0xFA57 for macOS compat) so the other end can find it.
 *   2. Both sides send a "login" request saying "hey, use DMA slot X for
 *      your outgoing data" and wait for the other to reply.
 *   3. When both have sent and received a login, we enable the paths
 *      (like opening a gate), start the DMA rings, and sanity-check the
 *      link with a ping/pong exchange.
 *   4. If either side disconnects, a "logout" tears it all down so the
 *      next connection starts fresh.
 *
 * This is the same pattern Linux uses for Thunderbolt networking —
 * just with our own protocol ID so we don't collide with IP-over-TB.
 */

#include "odl_tb5_core.h"
#include "odl_tb5_xd_proto.h"
#include <linux/delay.h>

/* Stop retrying the XDomain login after this many failures (0 = forever).
 * See HAZARD note in odl_tb5_login_work_fn(). */
int odl_login_max_retries = 20;
module_param_named(login_max_retries, odl_login_max_retries, int, 0644);
MODULE_PARM_DESC(login_max_retries,
	"Give up XDomain login after N failures (0 = retry forever; default 20)");

static void odl_tb5_login_work_fn(struct work_struct *work);
static void odl_tb5_connect_work_fn(struct work_struct *work);
static void odl_tb5_restart_work_fn(struct work_struct *work);
static int  odl_tb5_complete_connection(struct odl_tb5_device *dev);

#if IS_ENABLED(CONFIG_USB4)
void odl_tb5_xd_header_init(struct odl_tb5_xd_header *hdr,
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
		if (dev->transport->type == ODL_TB5_TRANSPORT_NHI &&
		    dev->xd && dev->xd->route == route)
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

	if (!dev) {
		mutex_unlock(&odl_tb5_devices_lock);
		pr_warn("odl_tb5: incoming packet route %llx — no matching device\n",
			route);
		return 0;
	}

	switch (hdr->type) {
	case ODL_TB5_MSG_LOGIN: {
		const struct odl_tb5_login_msg *pkg = buf;
		struct odl_tb5_login_response resp = { };
		u32 remote_tx_hopid = 0;
		u32 proto_ver = 0;
		int ret;

		/* Minimum: XDomain header (40 bytes). The payload fields
		 * (transmit_path, proto_version) may be absent if the peer
		 * uses a different login format (e.g., Apple ThunderboltRDMA).
		 * Default missing fields to 0 and proceed. */
		if (size < sizeof(struct odl_tb5_xd_header)) {
			mutex_unlock(&odl_tb5_devices_lock);
			return 0;
		}

		if (size >= sizeof(*pkg)) {
			remote_tx_hopid = pkg->transmit_path;
			proto_ver = pkg->proto_version;
		} else if (size >= sizeof(struct odl_tb5_xd_header) + 8) {
			/* Apple-style: transmit_path at offset 40, version at 44 */
			const u32 *payload = (const u32 *)(hdr + 1);
			remote_tx_hopid = payload[1];
			proto_ver = payload[0];
		}

		pr_info("odl_tb5: received login from peer "
			"(version=%u, tx_path=%u, size=%zu)\n",
			proto_ver, remote_tx_hopid, size);

		resp.xd_hdr.route_hi  = upper_32_bits(dev->xd->route);
		resp.xd_hdr.route_lo  = lower_32_bits(dev->xd->route);
		resp.xd_hdr.length_sn =
			(hdr->length_sn & XD_SN_MASK) |
			(sizeof(resp) / 4 - XD_HDR_SIZE_DW);
		resp.xd_hdr.uuid      = odl_tb5_proto_uuid;
		resp.xd_hdr.type      = ODL_TB5_MSG_LOGIN_RSP;
		resp.status            = ODL_TB5_LOGIN_STATUS_OK;
		resp.transmit_path     = dev->transport->local_tx_hopid(dev);

		/*
		 * Full OdinLink login packets carry proto_version and must
		 * match. Short / Apple-style packets omit a meaningful
		 * version (often 0) — skip the check for those and when
		 * protocol=1 (macOS compat).
		 */
		if (odl_protocol_mode == 0 &&
		    size >= sizeof(*pkg) &&
		    proto_ver != ODL_TB5_PROTOCOL_VER) {
			pr_err("odl_tb5: peer protocol version %u != local %u "
			       "— refusing (update both nodes to the same "
			       "driver revision)\n",
			       proto_ver, ODL_TB5_PROTOCOL_VER);
			resp.status = ODL_TB5_LOGIN_STATUS_PROTO_MISMATCH;
			ret = tb_xdomain_response(dev->xd, &resp, sizeof(resp),
						  TB_CFG_PKG_XDOMAIN_RESP);
			mutex_unlock(&odl_tb5_devices_lock);
			return 1;
		}

		ret = tb_xdomain_response(dev->xd, &resp, sizeof(resp),
					  TB_CFG_PKG_XDOMAIN_RESP);
		pr_info("odl_tb5: sent login response (ret=%d, route=%llx, "
			"sn=%u, tx_hopid=%d)\n",
			ret, dev->xd->route,
			(hdr->length_sn & XD_SN_MASK) >> 27,
			dev->transport->local_tx_hopid(dev));

		mutex_lock(&dev->state_lock);
		if (dev->state != ODL_TB5_STATE_HANDSHAKE) {
			pr_info("odl_tb5: peer restarted (our state=%d), "
				"scheduling restart\n", dev->state);
			dev->stale_remote_tx_hopid = dev->remote_tx_hopid;
			dev->remote_tx_hopid = remote_tx_hopid;
			dev->login_received = true;
			mutex_unlock(&dev->state_lock);
			mutex_unlock(&odl_tb5_devices_lock);
			schedule_work(&dev->restart_work);
			return 1;
		}

		dev->remote_tx_hopid = remote_tx_hopid;
		dev->login_received = true;
		if (dev->login_sent)
			need_complete = true;
		mutex_unlock(&dev->state_lock);

		mutex_unlock(&odl_tb5_devices_lock);
		if (need_complete)
			schedule_work(&dev->connect_work);

		return 1;
	}

	case ODL_TB5_MSG_LOGOUT:
		pr_info("odl_tb5: received logout from peer\n");

		mutex_lock(&dev->state_lock);
		dev->stale_remote_tx_hopid = dev->remote_tx_hopid;
		dev->login_received = false;
		dev->login_sent = false;
		mutex_unlock(&dev->state_lock);

		mutex_unlock(&odl_tb5_devices_lock);
		schedule_work(&dev->restart_work);

		return 1;

	default:
		mutex_unlock(&odl_tb5_devices_lock);
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
#endif /* CONFIG_USB4 */

#if !IS_ENABLED(CONFIG_USB4)
int odl_tb5_proto_register(void)
{
	return 0;
}

void odl_tb5_proto_unregister(void)
{
}
#endif

int odl_tb5_proto_send_login(struct odl_tb5_device *dev)
{
	bool need_complete = false;
	int ret;

	if (!dev->transport || !dev->transport->peer_send_login)
		return -ENODEV;

	/* peer_send_login validates the response (UUID/type/status),
	 * including PROTOCOL_VER mismatch → -ECONNREFUSED. */
	ret = dev->transport->peer_send_login(dev);
	if (ret) {
		pr_warn("odl_tb5: login request failed: %d\n", ret);
		return ret;
	}

	mutex_lock(&dev->state_lock);
	dev->login_sent = true;
	if (dev->login_received && dev->state == ODL_TB5_STATE_HANDSHAKE)
		need_complete = true;
	mutex_unlock(&dev->state_lock);

	pr_info("odl_tb5: login sent OK, remote_tx_hopid=%d\n",
		dev->remote_tx_hopid);

	if (need_complete)
		schedule_work(&dev->connect_work);

	return 0;
}

/* Finish handshake and bring link up. */
static int odl_tb5_complete_connection(struct odl_tb5_device *dev)
{
	int ret;

	if (!dev->transport)
		return -ENODEV;

	/* BUG1 fix (re-entry): this function is called again on every handshake
	 * restart.  Without releasing the hop-ID we already hold, each retry
	 * orphans the previous allocation (dev->in_hopid is simply overwritten)
	 * until enable_paths fails with -ENOMEM.  Release first. */
	if (dev->in_hopid_valid) {
		if (dev->tx.started && dev->transport->path_disable)
			dev->transport->path_disable(dev, dev->in_hopid);
		if (dev->transport->path_release_hopid)
			dev->transport->path_release_hopid(dev, dev->in_hopid);
		dev->in_hopid_valid = false;
	}

	if (dev->transport->path_alloc_hopid) {
		ret = dev->transport->path_alloc_hopid(dev,
						       dev->remote_tx_hopid);
		if (ret < 0) {
			pr_err("odl_tb5: failed to allocate input HopID: %d\n",
			       ret);
			return ret;
		}
	}
	dev->in_hopid = dev->remote_tx_hopid;
	dev->in_hopid_valid = true;

	ret = odl_tb5_rings_start(dev);
	if (ret) {
		pr_err("odl_tb5: failed to start rings: %d\n", ret);
		goto err_release_hopid;
	}

	{
		size_t rx_prime = (size_t)ODL_TB5_FRAME_SIZE * 16;

		ret = odl_tb5_submit_rx(dev, 0, rx_prime);
		if (ret) {
			pr_err("odl_tb5: failed to prime RX: %d\n", ret);
			odl_tb5_rings_stop(dev);
			goto err_release_hopid;
		}
		pr_info("odl_tb5: RX primed with 16 frames before "
			"enable_paths\n");
	}

	if (dev->transport->path_enable) {
		ret = dev->transport->path_enable(dev);
		if (ret) {
			pr_err("odl_tb5: failed to enable paths: %d\n", ret);
			odl_tb5_rings_stop(dev);
			goto err_release_hopid;
		}
	}

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_CONNECTED;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	{
		struct odl_tb5_transport_ring_info tx_info = { .hop = -1 };
		struct odl_tb5_transport_ring_info rx_info = { .hop = -1 };

		if (dev->transport->tx_ring_info)
			tx_info = dev->transport->tx_ring_info(dev);
		if (dev->transport->rx_ring_info)
			rx_info = dev->transport->rx_ring_info(dev);

		pr_info("odl_tb5: connected to peer "
			"(remote_tx_hopid=%d, "
			"tx_ring_hop=%d, rx_ring_hop=%d, "
			"ring_size=%d, E2E enabled)\n",
			dev->remote_tx_hopid,
			tx_info.hop, rx_info.hop,
			dev->tx.ring_size);
	}

	/* Allocate frame pool early so verify uses the non-blocking
	 * pool path for PING/PONG instead of the legacy submit_tx
	 * which blocks waiting for TX completion (unreliable on
	 * Barlow Ridge). */
	if (!dev->frame_pool.slots) {
		int pool_ret = odl_tb5_frame_pool_alloc(dev);

		if (pool_ret)
			pr_warn("odl_tb5: frame pool alloc failed (%d), "
				"verify will use legacy path\n", pool_ret);
	}

	/* Allocate batch buffer pool for throughput-mode TX. */
	if (!dev->batch_pool.bufs[0].virt) {
		int bret = odl_tb5_batch_pool_alloc(dev);

		if (bret)
			pr_warn("odl_tb5: batch pool alloc failed (%d), "
				"throughput mode disabled\n", bret);
	}

	hrtimer_start(&dev->rx_poll_timer,
		      ns_to_ktime(ODL_TB5_POLL_INTERVAL_NS),
		      HRTIMER_MODE_REL);
	schedule_work(&dev->verify_work);

	return 0;

err_release_hopid:
	if (dev->transport->path_release_hopid)
		dev->transport->path_release_hopid(dev, dev->in_hopid);
	dev->in_hopid_valid = false;
	return ret;
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

		pr_warn("odl_tb5: connection completion failed (%d), "
			"retrying handshake\n", ret);
		schedule_delayed_work(&dev->login_work,
				      msecs_to_jiffies(1000));
	}
}

/* Write a kernel DMA control message (ping or pong) via frame pool. */
static int odl_tb5_send_dma_msg(struct odl_tb5_device *dev, u32 type)
{
	struct odl_tb5_frame_slot *slot;
	struct odl_tb5_dma_hdr *dhdr;
	int ret;

	/* Use frame pool for non-blocking send (each msg gets its own
	 * slot, no drain wait).  Write raw DMA header at offset 0
	 * (no stream header) — the RX side uses legacy frames during
	 * verify and expects DMA magic at offset 0. */
	if (dev->frame_pool.slots) {
		slot = odl_tb5_frame_pool_get(&dev->frame_pool);
		if (!slot)
			return -ENOMEM;

		dhdr = slot->virt;
		memset(dhdr, 0, sizeof(*dhdr));
		dhdr->magic = cpu_to_le32(ODL_TB5_DMA_MAGIC);
		dhdr->type  = cpu_to_le32(type);

		slot->frame.size = sizeof(*dhdr);
		slot->frame.sof = ODL_TB5_PDF_SOF_CTRL;
		slot->frame.eof = ODL_TB5_PDF_EOF_CTRL;
		slot->frame.callback = odl_tb5_tx_callback;
		slot->tx_msg = NULL;

		ret = dev->transport->ring_tx(dev, &slot->frame);
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
		int ret;

		pr_info("odl_tb5: DMA ping received, sending pong\n");
		ret = odl_tb5_send_dma_msg(dev, ODL_TB5_DMA_PONG);

		/* BUG 10 fix: answering a PING is itself proof the DMA path
		 * works in both directions -- we received a frame (RX good)
		 * and submitted one (TX good).  Without this the handshake is
		 * an unwinnable race: the peer that verifies first calls
		 * odl_tb5_rings_reset() and sets rx_target/rx_posted to 0
		 * (no RX frames are posted until a STREAM_OPEN ioctl), so our
		 * PINGs land nowhere and our own PONG never arrives.  The
		 * second node then always fails "DMA verify failed after 300
		 * attempts" even though the link is perfectly healthy. */
		if (!ret) {
			dev->peer_ping_answered = true;
			if (!dev->pong_received)
				dev->pong_received = true;
			wake_up_all(&dev->verify_waitq);
		}
	} else {
		pr_warn("odl_tb5: unexpected DMA ctrl message type %d\n",
			type);
	}
}


/* Post-connection DMA verification via ping/pong exchange.
 *
 * IMPORTANT: We must NOT stop/start the RX ring between attempts.
 * Each ring stop cancels all pending RX frames, creating a dead
 * window where incoming PONGs are lost.  Instead, post enough RX
 * frames up front and just keep sending PINGs until a PONG arrives.
 */
static void odl_tb5_verify_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, verify_work);
	long ret;
	int attempt;

	dev->pong_received = false;   /* peer_ping_answered deliberately kept */

	/* Use frame pool for RX during verify — each pool slot is
	 * independent and auto-reposts after consumption, so we never
	 * run out of RX frames.  The legacy submit_rx only posts 16
	 * frames and can't repost without a ring reset. */
	if (dev->frame_pool.slots) {
		dev->rx_target = dev->frame_pool.size / 2;
		odl_tb5_rx_repost(dev);
		pr_info("odl_tb5: verify using pool RX (target=%d)\n",
			dev->rx_target);
	} else {
		size_t buf_size = (size_t)ODL_TB5_FRAME_SIZE * 16;

		ret = odl_tb5_submit_rx(dev, 0, buf_size);
		if (ret) {
			pr_warn("odl_tb5: DMA verify: failed to post RX "
				"(%ld)\n", ret);
			goto out_reset;
		}
	}

	for (attempt = 0; attempt < 300; attempt++) {
		if (dev->state != ODL_TB5_STATE_CONNECTED)
			goto out_reset;

		/* Answering the peer's PING already proved both directions. */
		if (dev->peer_ping_answered)
			break;

		flush_work(&dev->ctrl_reply_work);

		ret = odl_tb5_send_dma_msg(dev, ODL_TB5_DMA_PING);
		if (ret) {
			pr_warn("odl_tb5: DMA verify: send ping failed "
				"(%ld)\n", ret);
			goto out_reset;
		}

		if (attempt == 0)
			pr_info("odl_tb5: DMA ping sent, "
				"waiting for pong\n");

		ret = wait_event_interruptible_timeout(
				dev->verify_waitq,
				dev->pong_received || dev->peer_ping_answered,
				msecs_to_jiffies(100));
		if (ret > 0)
			break;

		if (ret < 0)
			goto out_reset;

		if ((attempt + 1) % 50 == 0)
			pr_info("odl_tb5: DMA ping attempt %d, "
				"still waiting for pong\n", attempt + 1);

		/* Do NOT reset the RX ring here — that kills in-flight
		 * frames and creates dead windows where PONGs are lost. */
	}

	if (!dev->pong_received && !dev->peer_ping_answered) {
		pr_err("odl_tb5: DMA verify failed after %d attempts\n",
		       attempt);
		goto out_reset;
	}

	pr_info("odl_tb5: DMA path verified, resetting rings for userspace\n");

	flush_work(&dev->ctrl_reply_work);
	hrtimer_cancel(&dev->rx_poll_timer);
	odl_tb5_rings_reset(dev);

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_READY;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	/*
	 * Don't post pool frames yet — legacy consumers (daemon, CLI)
	 * don't use stream headers.  Pool RX repost starts when the
	 * first stream is opened via STREAM_OPEN ioctl.
	 */
	dev->rx_target = 0;
	atomic_set(&dev->rx_posted, 0);

	/* Restart the hrtimer poll for stream data — NHI MSI-X
	 * interrupts fire but descriptor write-back can lag, so we poll
	 * at 50 us to keep latency low. */
	hrtimer_start(&dev->rx_poll_timer,
		      ns_to_ktime(ODL_TB5_POLL_INTERVAL_NS),
		      HRTIMER_MODE_REL);

	pr_info("odl_tb5: entering READY state\n");
	return;

out_reset:
	hrtimer_cancel(&dev->rx_poll_timer);
	odl_tb5_rings_reset(dev);
}

/* Tear down stale connection and restart the handshake. */
static void odl_tb5_restart_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, restart_work);

	hrtimer_cancel(&dev->rx_poll_timer);
	cancel_work_sync(&dev->verify_work);
	cancel_work_sync(&dev->ctrl_reply_work);
	cancel_work_sync(&dev->connect_work);
	cancel_delayed_work_sync(&dev->login_work);
	dev->pong_received = false;

	if (dev->tx.started) {
		/* Tear down using the hop-ID we actually allocated
		 * (dev->in_hopid), not the current peer value — a
		 * re-login may already have overwritten remote_tx_hopid. */
		if (dev->transport && dev->transport->path_disable)
			dev->transport->path_disable(dev, dev->in_hopid);
		odl_tb5_rings_stop(dev);
		if (dev->in_hopid_valid) {
			if (dev->transport &&
			    dev->transport->path_release_hopid)
				dev->transport->path_release_hopid(
					dev, dev->in_hopid);
			dev->in_hopid_valid = false;
		}
	}

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_HANDSHAKE;
	wake_up_all(&dev->state_waitq);
	dev->login_sent = false;
	dev->login_retries = 0;
	mutex_unlock(&dev->state_lock);

	pr_info("odl_tb5: connection restarted, beginning handshake\n");
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
			pr_info("odl_tb5: login attempt %d failed (%d), "
				"retrying in %lu ms\n",
				dev->login_retries, ret, delay_ms);

		/* HAZARD FIX: bound the retry storm.  Retrying forever hammers
		 * tb_cfg_read() against a router that may already be wedged;
		 * observed to take down the USB4 controller and (shared power/
		 * firmware domain on Strix Halo) the GPU with it, ending in
		 * amdgpu_irq_put/drm_buddy_fini NULL-deref and a hard freeze.
		 * Give up and stay quiescent instead. */
		if (odl_login_max_retries > 0 &&
		    dev->login_retries >= odl_login_max_retries) {
			pr_warn("odl_tb5: giving up after %d login attempts "
				"(last err %d); staying idle.  Check the peer "
				"is up and only ONE service is bound.\n",
				dev->login_retries, ret);
			return;
		}

		schedule_delayed_work(&dev->login_work,
				      msecs_to_jiffies(delay_ms));
	}
}

/* Send a logout notification to the peer. */
int odl_tb5_proto_send_logout(struct odl_tb5_device *dev)
{
	if (dev->transport && dev->transport->peer_send_logout)
		dev->transport->peer_send_logout(dev);

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_DISCONNECTED;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	pr_info("odl_tb5: logout sent to peer\n");

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
	hrtimer_setup(&dev->rx_poll_timer, odl_tb5_rx_poll_timer_fn,
		      CLOCK_MONOTONIC, HRTIMER_MODE_REL);
	init_waitqueue_head(&dev->verify_waitq);

	dev->login_retries  = 0;
	dev->login_sent     = false;
	dev->login_received = false;
	dev->pong_received  = false;
	dev->peer_ping_answered = false;

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_HANDSHAKE;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	schedule_delayed_work(&dev->login_work, 0);

	return 0;
}

/* Tear down the protocol layer for a device. */
void odl_tb5_proto_exit(struct odl_tb5_device *dev)
{
	hrtimer_cancel(&dev->rx_poll_timer);
	cancel_work_sync(&dev->verify_work);
	cancel_work_sync(&dev->ctrl_reply_work);
	cancel_work_sync(&dev->restart_work);
	cancel_work_sync(&dev->connect_work);
	cancel_delayed_work_sync(&dev->login_work);
}
