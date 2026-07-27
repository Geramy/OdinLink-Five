// SPDX-License-Identifier: MIT
/*
 * OdinLink — Driver Lifecycle: Load, Probe, Remove
 *
 * What happens when you run `sudo insmod odl_tb5.ko`:
 *   1. Advertises "OdinLink is here" to any Thunderbolt peer on the other
 *      end of the cable (via XDomain property directories).
 *   2. When a peer matches our protocol ID, the kernel calls probe(),
 *      which creates a device struct, allocates DMA rings, and kicks off
 *      the login handshake.
 *   3. remove() tears everything down when the cable is unplugged or the
 *      module is unloaded.
 *
 * Also handles module parameters: ring_size, loopback, protocol, e2e.
 */

#include "odl_tb5_core.h"

LIST_HEAD(odl_tb5_devices_list);
DEFINE_MUTEX(odl_tb5_devices_lock);

static DEFINE_IDA(odl_tb5_ida);

unsigned int odl_ring_size = ODL_TB5_RING_SIZE_DEFAULT;
module_param(odl_ring_size, uint, 0444);
MODULE_PARM_DESC(odl_ring_size,
	"NHI ring entries per direction (power-of-2, default 4096 = 16 MB/batch)");

int odl_loopback_count = 0;
module_param_named(loopback, odl_loopback_count, int, 0444);
MODULE_PARM_DESC(loopback,
	"Create N software loopback devices (max 16, default 0; no NHI hw needed)");

int odl_protocol_mode = 0;
module_param_named(protocol, odl_protocol_mode, int, 0444);
MODULE_PARM_DESC(protocol,
	"XDomain protocol mode: 0=OdinLink (0x4F4C, default), 1=Apple (0xFA57)");

bool odl_e2e = true;
module_param_named(e2e, odl_e2e, bool, 0444);

/* upstream PR #21: bounded RX busy-poll before sleeping. The RX softirq
 * increments rx_complete on another CPU, so a spinning reader sees it within
 * cache-coherency latency and skips a ~10-15 us context-switch wake. */
unsigned int odl_busy_poll_us = 0;
module_param(odl_busy_poll_us, uint, 0644);
MODULE_PARM_DESC(odl_busy_poll_us,
	"Bounded RX busy-poll window in microseconds before sleeping (0 = off)");

/* Bind at most N XDomain services (0 = unlimited).  With two Thunderbolt
 * cables both peer services sit at route=2 (BUG 2) and the handshake cannot
 * complete; unbinding one afterwards runs the full teardown path, so bind
 * only one from the start instead. */
int odl_max_devices = 0;
module_param_named(max_devices, odl_max_devices, int, 0444);
MODULE_PARM_DESC(max_devices,
	"Bind at most N XDomain services (0 = unlimited; use 1 with two cables)");
static atomic_t odl_bound_count = ATOMIC_INIT(0);
MODULE_PARM_DESC(e2e,
	"Enable end-to-end flow control (default=1). Set 0 for TB3 controllers "
	"that do not support RING_FLAG_E2E.");

/* Apple protocol uses its own property key and registers as an alternate
 * service so macOS ThunderboltRDMA can discover us via XDomain matching. */
static struct tb_property_dir *odl_tb5_apple_property_dir;

const uuid_t odl_tb5_proto_uuid =
	UUID_INIT(0x4f444c4e, 0x4b54, 0x4235,
		  0x4f, 0x44, 0x49, 0x4e, 0x4c, 0x49, 0x4e, 0x4b);

static struct tb_property_dir *odl_tb5_property_dir;

static const struct tb_service_id odl_tb5_ids[] = {
	{ TB_SERVICE(ODL_TB5_PROTOCOL_KEY, ODL_TB5_PROTOCOL_ID) },
	{ TB_SERVICE(ODL_TB5_PROTOCOL_KEY_APPLE, ODL_TB5_PROTOCOL_ID_APPLE) },
	{ }
};
MODULE_DEVICE_TABLE(tbsvc, odl_tb5_ids);

static int odl_tb5_probe(struct tb_service *svc,
			 const struct tb_service_id *id)
{
	if (odl_max_devices > 0 &&
	    atomic_inc_return(&odl_bound_count) > odl_max_devices) {
		atomic_dec(&odl_bound_count);
		pr_info("odl_tb5: max_devices=%d reached, skipping service\n",
			odl_max_devices);
		return -ENODEV;
	}
	struct odl_tb5_device *dev;
	int ret;

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return -ENOMEM;

	dev->svc = svc;
	dev->xd  = tb_service_parent(svc);

	ret = ida_alloc_max(&odl_tb5_ida, ODL_TB5_MAX_DEVICES - 1, GFP_KERNEL);
	if (ret < 0) {
		kfree(dev);
		return ret;
	}
	dev->index = ret;
	dev->local_tx_hopid = -1;

	dev->state = ODL_TB5_STATE_DISCONNECTED;

	mutex_init(&dev->state_lock);
	init_waitqueue_head(&dev->state_waitq);
	spin_lock_init(&dev->tx.lock);
	spin_lock_init(&dev->rx.lock);
	init_waitqueue_head(&dev->tx.waitq);
	init_waitqueue_head(&dev->rx.waitq);
	atomic_set(&dev->tx.completed, 0);
	atomic_set(&dev->tx.submitted, 0);
	atomic_set(&dev->rx.completed, 0);
	atomic_set(&dev->rx.submitted, 0);
	atomic_set(&dev->open_count, 0);

	/* Stream management init */
	hash_init(dev->streams);
	ida_init(&dev->stream_ida);
	mutex_init(&dev->stream_lock);
	INIT_WORK(&dev->tx_drain_work, odl_tb5_tx_drain_work_fn);
	atomic_set(&dev->rx_posted, 0);
	dev->rx_target = 0;

	atomic_set(&dev->removing, 0);

	/* Adaptive TX mode defaults */
	dev->tx_adaptive.mode = ODL_TB5_TX_LATENCY;
	dev->tx_adaptive.consecutive_low = 0;
	/* Watermarks gate the shared frame pool (ODL_TB5_FRAME_POOL_SIZE
	 * slots), NOT the NHI ring depth.  With large odl_ring_size the raw
	 * ring*3/4 exceeds the pool, so the adaptive logic can never trip and
	 * TX flow control is miscalibrated.  Clamp to the usable pool.
	 * (upstream PR #20) */
	dev->tx_adaptive.high_watermark =
		min_t(unsigned int, odl_ring_size * 3 / 4,
		      ODL_TB5_FRAME_POOL_SIZE - ODL_TB5_TX_POOL_RESERVE);
	dev->tx_adaptive.low_watermark  =
		min_t(unsigned int, odl_ring_size / 4,
		      (ODL_TB5_FRAME_POOL_SIZE - ODL_TB5_TX_POOL_RESERVE) / 2);

	ret = odl_tb5_chardev_create(dev);
	if (ret) {
		pr_err("odl_tb5: chardev create failed for index %d: %d\n",
		       dev->index, ret);
		goto err_free_dev;
	}

	ret = odl_tb5_rings_alloc(dev);
	if (ret) {
		pr_err("odl_tb5: ring alloc failed for index %d: %d\n",
		       dev->index, ret);
		goto err_chardev;
	}

	ret = odl_tb5_dma_bufs_alloc(dev);
	if (ret) {
		pr_err("odl_tb5: DMA buf alloc failed for index %d: %d\n",
		       dev->index, ret);
		goto err_rings;
	}

	ret = odl_tb5_proto_init(dev);
	if (ret) {
		pr_err("odl_tb5: proto init failed for index %d: %d\n",
		       dev->index, ret);
		goto err_dma;
	}

	mutex_lock(&odl_tb5_devices_lock);
	list_add_tail(&dev->list, &odl_tb5_devices_list);
	mutex_unlock(&odl_tb5_devices_lock);

	tb_service_set_drvdata(svc, dev);

	pr_info("odl_tb5: probed device index %d on xdomain %pUb\n",
		dev->index, dev->xd->remote_uuid);

	return 0;

err_dma:
	odl_tb5_dma_bufs_free(dev);
err_rings:
	odl_tb5_rings_free(dev);
err_chardev:
	odl_tb5_chardev_destroy(dev);
err_free_dev:
	if (odl_max_devices > 0)
		atomic_dec(&odl_bound_count);
	ida_free(&odl_tb5_ida, dev->index);
	kfree(dev);
	return ret;
}

static void odl_tb5_remove(struct tb_service *svc)
{
	struct odl_tb5_device *dev = tb_service_get_drvdata(svc);
	enum odl_tb5_conn_state saved_state;

	if (!dev)
		return;

	atomic_set(&dev->removing, 1);

	mutex_lock(&dev->state_lock);
	saved_state = dev->state;
	dev->state = ODL_TB5_STATE_DISCONNECTED;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	if (saved_state == ODL_TB5_STATE_CONNECTED ||
	    saved_state == ODL_TB5_STATE_READY)
		odl_tb5_proto_send_logout(dev);

	hrtimer_cancel(&dev->rx_poll_timer);

	cancel_work_sync(&dev->verify_work);
	cancel_work_sync(&dev->ctrl_reply_work);
	cancel_work_sync(&dev->restart_work);
	cancel_work_sync(&dev->connect_work);
	cancel_delayed_work_sync(&dev->login_work);
	cancel_work_sync(&dev->tx_drain_work);

	odl_tb5_rings_stop(dev);

	synchronize_rcu();

	mutex_lock(&odl_tb5_devices_lock);
	list_del_rcu(&dev->list);
	mutex_unlock(&odl_tb5_devices_lock);

	odl_tb5_streams_destroy_all(dev);
	ida_destroy(&dev->stream_ida);

	odl_tb5_frame_pool_free(dev);
	odl_tb5_batch_pool_free(dev);
	odl_tb5_dma_bufs_free(dev);
	odl_tb5_rings_free(dev);

	/* BUG1 fix: release regardless of connection state.  The original code
	 * released only from CONNECTED/READY, so removal during HANDSHAKE
	 * (login retrying, peer gone, admin unbind) leaked the hop-ID until
	 * enable_paths returned -ENOMEM on every later load.  Ordering is kept
	 * exactly as upstream: this runs AFTER rings_stop()/bufs_free() above. */
	if (dev->in_hopid_valid) {
		tb_xdomain_disable_paths(dev->xd,
					 dev->local_tx_hopid,
					 dev->tx.ring ? dev->tx.ring->hop : -1,
					 dev->in_hopid,
					 dev->rx.ring ? dev->rx.ring->hop : -1);
		tb_xdomain_release_in_hopid(dev->xd, dev->in_hopid);
		dev->in_hopid_valid = false;
	}

	odl_tb5_chardev_destroy(dev);

	pr_info("odl_tb5: removed device index %d\n", dev->index);

	ida_free(&odl_tb5_ida, dev->index);
	kfree(dev);
}

static struct tb_service_driver odl_tb5_driver = {
	.driver.name	= "odl_tb5",
	.probe		= odl_tb5_probe,
	.remove		= odl_tb5_remove,
	.id_table	= odl_tb5_ids,
};

static int __init odl_tb5_init(void)
{
	int ret;

	if (!is_power_of_2(odl_ring_size) ||
	    odl_ring_size < ODL_TB5_RING_SIZE_MIN ||
	    odl_ring_size > ODL_TB5_RING_SIZE_MAX) {
		pr_err("odl_tb5: invalid ring_size=%u (must be power-of-2, %u-%u)\n",
		       odl_ring_size, ODL_TB5_RING_SIZE_MIN,
		       ODL_TB5_RING_SIZE_MAX);
		return -EINVAL;
	}

	ret = odl_tb5_chardev_init();
	if (ret)
		return ret;

	/* If loopback=1 or more, create software-only devices.
	 * Loopback devices work without Thunderbolt hardware and
	 * don't need property directories or service registration. */
	if (odl_loopback_count > 0) {
		ret = odl_loopback_init();
		if (ret)
			goto err_chardev;
		pr_info("odl_tb5: loopback mode enabled (%d devices)\n",
			odl_loopback_count);
		return 0;
	}

	odl_tb5_property_dir = tb_property_create_dir(&odl_tb5_proto_uuid);
	if (!odl_tb5_property_dir) {
		ret = -ENOMEM;
		goto err_chardev;
	}

	/* Choose protocol ID based on mode */
	u32 protocol_id = ODL_TB5_PROTOCOL_ID;
	if (odl_protocol_mode == 1)
		protocol_id = ODL_TB5_PROTOCOL_ID_APPLE;

	const char *protocol_key = ODL_TB5_PROTOCOL_KEY;
	if (odl_protocol_mode == 1)
		protocol_key = ODL_TB5_PROTOCOL_KEY_APPLE;

	ret = tb_property_add_immediate(odl_tb5_property_dir, "prtcid",
					protocol_id);
	if (ret)
		goto err_dir;

	ret = tb_property_add_immediate(odl_tb5_property_dir, "prtcvers",
					ODL_TB5_PROTOCOL_VER);
	if (ret)
		goto err_dir;

	ret = tb_property_add_immediate(odl_tb5_property_dir, "prtcrevs", 1);
	if (ret)
		goto err_dir;

	ret = tb_property_add_immediate(odl_tb5_property_dir, "prtcstns", 0);
	if (ret)
		goto err_dir;

	ret = tb_register_property_dir(protocol_key,
				       odl_tb5_property_dir);
	if (ret)
		goto err_dir;

	/* In Apple mode, also register under OdinLink's original key so we
	 * can still talk to other OdinLink nodes. Need a separate directory
	 * since the same dir can't be registered under two keys. */
	if (odl_protocol_mode == 1) {
		odl_tb5_apple_property_dir =
			tb_property_create_dir(&odl_tb5_proto_uuid);
		if (!odl_tb5_apple_property_dir) {
			ret = -ENOMEM;
			goto err_dir;
		}
		tb_property_add_immediate(odl_tb5_apple_property_dir,
					  "prtcid", ODL_TB5_PROTOCOL_ID);
		tb_property_add_immediate(odl_tb5_apple_property_dir,
					  "prtcvers", ODL_TB5_PROTOCOL_VER);
		tb_property_add_immediate(odl_tb5_apple_property_dir,
					  "prtcrevs", 1);
		tb_property_add_immediate(odl_tb5_apple_property_dir,
					  "prtcstns", 0);
		ret = tb_register_property_dir(ODL_TB5_PROTOCOL_KEY,
					odl_tb5_apple_property_dir);
		if (ret)
			goto err_dir;
	}

	odl_tb5_proto_register();

	ret = tb_register_service_driver(&odl_tb5_driver);
	if (ret)
		goto err_proto;

	pr_info("odl_tb5: OdinLink TB5 driver loaded (ring_size=%u)\n",
		odl_ring_size);

	return 0;

err_proto:
	odl_tb5_proto_unregister();
err_dir:
	if (odl_tb5_apple_property_dir) {
		tb_property_free_dir(odl_tb5_apple_property_dir);
		odl_tb5_apple_property_dir = NULL;
	}
	tb_property_free_dir(odl_tb5_property_dir);
err_chardev:
	odl_tb5_chardev_exit();
	return ret;
}

static void __exit odl_tb5_exit(void)
{
	struct odl_tb5_device *dev, *tmp;

	/* Clean up software loopback devices first (no NHI/hardware deps) */
	if (odl_loopback_count > 0) {
		odl_loopback_exit();
		goto out;
	}

	/* Unregister first so tb core removes bound services before orphan cleanup. */
	tb_unregister_service_driver(&odl_tb5_driver);

	mutex_lock(&odl_tb5_devices_lock);
	list_for_each_entry_safe(dev, tmp, &odl_tb5_devices_list, list) {
		pr_warn("odl_tb5: cleaning up orphaned device at exit\n");
		list_del_rcu(&dev->list);
		atomic_set(&dev->removing, 1);
		hrtimer_cancel(&dev->rx_poll_timer);
		cancel_work_sync(&dev->verify_work);
		cancel_work_sync(&dev->ctrl_reply_work);
		cancel_work_sync(&dev->restart_work);
		cancel_work_sync(&dev->connect_work);
		cancel_delayed_work_sync(&dev->login_work);
		cancel_work_sync(&dev->tx_drain_work);
		odl_tb5_rings_stop(dev);
		synchronize_rcu();
		odl_tb5_streams_destroy_all(dev);
		ida_destroy(&dev->stream_ida);
		odl_tb5_frame_pool_free(dev);
		odl_tb5_batch_pool_free(dev);
		odl_tb5_dma_bufs_free(dev);
		odl_tb5_rings_free(dev);
		odl_tb5_chardev_destroy(dev);
		ida_free(&odl_tb5_ida, dev->index);
		kfree(dev);
	}
	mutex_unlock(&odl_tb5_devices_lock);

	odl_tb5_proto_unregister();

	/* Unregister property dirs: main dir under its protocol key,
	 * and the Apple backup dir under OdinLink key if dual-mode */
	const char *main_key = (odl_protocol_mode == 1)
		? ODL_TB5_PROTOCOL_KEY_APPLE : ODL_TB5_PROTOCOL_KEY;

	tb_unregister_property_dir(main_key, odl_tb5_property_dir);
	tb_property_free_dir(odl_tb5_property_dir);

	if (odl_tb5_apple_property_dir) {
		tb_unregister_property_dir(ODL_TB5_PROTOCOL_KEY,
					   odl_tb5_apple_property_dir);
		tb_property_free_dir(odl_tb5_apple_property_dir);
	}
out:
	/*
	 * Streams are freed with kfree_rcu(), so a grace period may still be
	 * outstanding here. kfree_rcu() is serviced by the core kernel rather
	 * than by module text, so unloading cannot jump into freed code — but
	 * waiting is cheap at unload and leaves nothing in flight against a
	 * module that is going away.
	 */
	rcu_barrier();

	odl_tb5_chardev_exit();
	ida_destroy(&odl_tb5_ida);
	pr_info("odl_tb5: OdinLink TB5 driver unloaded\n");
}

module_init(odl_tb5_init);
module_exit(odl_tb5_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("OdinLink Team");
MODULE_DESCRIPTION("OdinLink Thunderbolt 5 DMA Ring Driver");
MODULE_IMPORT_NS("DMA_BUF");
