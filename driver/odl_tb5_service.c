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
#if IS_ENABLED(CONFIG_USB4)
#include <linux/pci.h>
#include <linux/device.h>
#endif

static atomic_t odl_bound_count = ATOMIC_INIT(0);

/* Apple protocol uses its own property key and registers as an alternate
 * service so macOS ThunderboltRDMA can discover us via XDomain matching. */
#if IS_ENABLED(CONFIG_USB4)
static struct tb_property_dir *odl_tb5_apple_property_dir;
#endif

static struct tb_property_dir *odl_tb5_property_dir;

#if IS_ENABLED(CONFIG_USB4)
static const struct tb_service_id odl_tb5_ids[] = {
	{ TB_SERVICE(ODL_TB5_PROTOCOL_KEY, ODL_TB5_PROTOCOL_ID) },
	{ TB_SERVICE(ODL_TB5_PROTOCOL_KEY_APPLE, ODL_TB5_PROTOCOL_ID_APPLE) },
	{ }
};
MODULE_DEVICE_TABLE(tbsvc, odl_tb5_ids);
#endif

#if IS_ENABLED(CONFIG_USB4)
extern const struct odl_tb5_transport_ops odl_tb5_nhi_transport;

struct nhi_priv {
	struct tb_xdomain	*xd;
	int			local_tx_hopid;
};

static struct odl_tb5_device *odl_tb5_find_by_xd(struct tb_xdomain *xd)
{
	struct odl_tb5_device *dev;

	if (!xd)
		return NULL;
	list_for_each_entry(dev, &odl_tb5_devices_list, list) {
		if (dev->xd == xd)
			return dev;
	}
	return NULL;
}

static void odl_tb5_destroy(struct odl_tb5_device *dev)
{
	enum odl_tb5_conn_state saved_state;

	if (!dev || atomic_xchg(&dev->removing, 1))
		return;

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

	if (dev->in_hopid_valid && dev->transport) {
		if (dev->transport->path_disable)
			dev->transport->path_disable(dev, dev->in_hopid);
		if (dev->transport->path_release_hopid)
			dev->transport->path_release_hopid(dev, dev->in_hopid);
		dev->in_hopid_valid = false;
	}

	odl_tb5_chardev_destroy(dev);

	if (odl_max_devices > 0)
		atomic_dec(&odl_bound_count);

	pr_info("odl_tb5: removed device index %d\n", dev->index);

	if (dev->xd)
		tb_xdomain_put(dev->xd);

	kfree(dev->transport_priv);
	ida_free(&odl_tb5_ida, dev->index);
	kfree(dev);
}

static void odl_tb5_xd_detach(void *data)
{
	odl_tb5_destroy(data);
}

static int odl_tb5_attach(struct tb_xdomain *xd, struct tb_service *svc,
			  bool bind_any);

static int odl_tb5_bind_xdomain(struct tb_xdomain *xd)
{
	if (!xd || xd->is_unplugged)
		return -ENODEV;

	mutex_lock(&odl_tb5_devices_lock);
	if (odl_tb5_find_by_xd(xd)) {
		mutex_unlock(&odl_tb5_devices_lock);
		return -EEXIST;
	}
	mutex_unlock(&odl_tb5_devices_lock);

	return odl_tb5_attach(xd, NULL, true);
}

static int odl_walk_tb(struct device *dev, void *data)
{
	struct tb_xdomain *xd = tb_to_xdomain(dev);

	(void)data;
	if (xd && odl_bind_any)
		odl_tb5_bind_xdomain(xd);
	device_for_each_child(dev, data, odl_walk_tb);
	return 0;
}

static int odl_scan_nhi(struct device *dev, void *data)
{
	struct pci_dev *pdev = to_pci_dev(dev);

	if (!pdev)
		return 0;
	/* USB4 / Thunderbolt NHI (class 0x0c0340). */
	if (pdev->class != 0x0c0340)
		return 0;
	device_for_each_child(dev, data, odl_walk_tb);
	return 0;
}

static void odl_tb5_scan_xdomains(struct work_struct *work);

static DECLARE_DELAYED_WORK(odl_tb5_scan_work, odl_tb5_scan_xdomains);

static void odl_tb5_scan_xdomains(struct work_struct *work)
{
	(void)work;
	bus_for_each_dev(&pci_bus_type, NULL, NULL, odl_scan_nhi);
	if (odl_bind_any)
		schedule_delayed_work(&odl_tb5_scan_work, 2 * HZ);
}

static int odl_tb5_probe(struct tb_service *svc,
			 const struct tb_service_id *id)
{
	struct tb_xdomain *xd = tb_service_parent(svc);

	(void)id;
	if (!xd)
		return -ENODEV;
	return odl_tb5_attach(xd, svc, false);
}

static int odl_tb5_attach(struct tb_xdomain *xd, struct tb_service *svc,
			  bool bind_any)
{
	if (odl_max_devices > 0 &&
	    atomic_inc_return(&odl_bound_count) > odl_max_devices) {
		atomic_dec(&odl_bound_count);
		pr_info("odl_tb5: max_devices=%d reached, skipping service\n",
			odl_max_devices);
		return -ENODEV;
	}
	struct odl_tb5_device *dev;
	struct nhi_priv *priv;
	int ret;

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev) {
		if (odl_max_devices > 0)
			atomic_dec(&odl_bound_count);
		return -ENOMEM;
	}

	dev->svc = svc;
	dev->xd  = tb_xdomain_get(xd);
	dev->bind_any = bind_any;
	/* skip_login is a Mac-sink escape hatch; honour it on advertised
	 * services too, not only bind_any attachments. */
	if (odl_skip_login)
		dev->skip_handshake = true;

	ret = ida_alloc_max(&odl_tb5_ida, ODL_TB5_MAX_DEVICES - 1, GFP_KERNEL);
	if (ret < 0) {
		if (dev->xd)
			tb_xdomain_put(dev->xd);
		if (odl_max_devices > 0)
			atomic_dec(&odl_bound_count);
		kfree(dev);
		return ret;
	}
	dev->index = ret;

	priv = kzalloc(sizeof(*priv), GFP_KERNEL);
	if (!priv) {
		if (dev->xd)
			tb_xdomain_put(dev->xd);
		if (odl_max_devices > 0)
			atomic_dec(&odl_bound_count);
		ida_free(&odl_tb5_ida, dev->index);
		kfree(dev);
		return -ENOMEM;
	}
	priv->xd = dev->xd;
	priv->local_tx_hopid = -1;

	dev->transport = &odl_tb5_nhi_transport;
	dev->transport_priv = priv;

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
	/* rx_posted_min is a low-water mark: start high so the first real
	 * value wins. The rest are plain counters and kzalloc zeroed them. */
	atomic_set(&dev->rx_posted_min, INT_MAX);
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

	/*
	 * On identity-IOMMU hosts (common with iommu=pt on Proxmox/virt),
	 * dma_alloc_coherent needs physically contiguous memory. The default
	 * odl_ring_size=4096 requests 16 MB batches (order-12) and fails with
	 * -ENOMEM even when RAM is free. Retry with progressively smaller
	 * power-of-two ring sizes so probe still succeeds (issue #23).
	 */
	{
		unsigned int try_rs = odl_ring_size;
		unsigned int requested = odl_ring_size;
		int last_err = -ENOMEM;

		while (try_rs >= ODL_TB5_RING_SIZE_MIN) {
			odl_ring_size = try_rs;
			ret = odl_tb5_rings_alloc(dev);
			if (ret) {
				last_err = ret;
				pr_err("odl_tb5: ring alloc failed for index %d "
				       "(odl_ring_size=%u): %d\n",
				       dev->index, try_rs, ret);
				try_rs >>= 1;
				continue;
			}

			ret = odl_tb5_dma_bufs_alloc(dev);
			if (!ret) {
				if (try_rs != requested)
					pr_warn("odl_tb5: reduced odl_ring_size "
						"%u → %u after alloc failure "
						"(identity IOMMU / iommu=pt often "
						"needs odl_ring_size=1024)\n",
						requested, try_rs);
				/* Keep the working size for any later probes. */
				odl_ring_size = try_rs;
				dev->tx_adaptive.high_watermark =
					min_t(unsigned int, try_rs * 3 / 4,
					      ODL_TB5_FRAME_POOL_SIZE -
					      ODL_TB5_TX_POOL_RESERVE);
				dev->tx_adaptive.low_watermark =
					min_t(unsigned int, try_rs / 4,
					      (ODL_TB5_FRAME_POOL_SIZE -
					       ODL_TB5_TX_POOL_RESERVE) / 2);
				break;
			}

			last_err = ret;
			pr_err("odl_tb5: DMA buf alloc failed for index %d "
			       "(odl_ring_size=%u, %zu bytes/batch): %d\n",
			       dev->index, try_rs,
			       (size_t)ODL_TB5_FRAME_SIZE * try_rs, ret);
			odl_tb5_rings_free(dev);
			try_rs >>= 1;
		}

		if (ret) {
			odl_ring_size = requested;
			pr_err("odl_tb5: could not allocate DMA rings/buffers "
			       "down to odl_ring_size=%u (last err %d). On "
			       "iommu=pt hosts try: modprobe odl_tb5 "
			       "odl_ring_size=1024\n",
			       ODL_TB5_RING_SIZE_MIN, last_err);
			ret = last_err;
			goto err_chardev;
		}
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

	if (svc)
		tb_service_set_drvdata(svc, dev);
	else {
		ret = devm_add_action(&xd->dev, odl_tb5_xd_detach, dev);
		if (ret) {
			pr_err("odl_tb5: devm_add_action failed: %d\n", ret);
			goto err_listed;
		}
	}

	pr_info("odl_tb5: probed device index %d (transport=%s, bind_any=%d)\n",
		dev->index, dev->transport->name, bind_any);

	return 0;

err_listed:
	odl_tb5_destroy(dev);
	return ret;

err_dma:
	odl_tb5_dma_bufs_free(dev);
err_rings:
	odl_tb5_rings_free(dev);
err_chardev:
	odl_tb5_chardev_destroy(dev);
err_free_dev:
	if (dev->xd)
		tb_xdomain_put(dev->xd);
	kfree(dev->transport_priv);
	if (odl_max_devices > 0)
		atomic_dec(&odl_bound_count);
	ida_free(&odl_tb5_ida, dev->index);
	kfree(dev);
	return ret;
}

static void odl_tb5_remove(struct tb_service *svc)
{
	struct odl_tb5_device *dev = tb_service_get_drvdata(svc);

	if (!dev)
		return;
	tb_service_set_drvdata(svc, NULL);
	odl_tb5_destroy(dev);
}

static struct tb_service_driver odl_tb5_driver = {
	.driver.name	= "odl_tb5",
	.probe		= odl_tb5_probe,
	.remove		= odl_tb5_remove,
	.id_table	= odl_tb5_ids,
};

/*
 * Issue #4: "module loaded, no /dev" looks like success. The chardev is
 * created in probe(), and probe() only runs when the other machine
 * advertises the same XDomain protocol. thunderbolt-net / ThunderboltIP
 * is a different protocol and can time out on its own.
 *
 * One delayed line if we are still unbound. Do not walk the thunderbolt
 * bus — those structs are not a stable service-driver API.
 */
#define ODL_TB5_WAIT_PEER_SEC	15

static void odl_tb5_wait_peer_fn(struct work_struct *work)
{
	bool empty;

	mutex_lock(&odl_tb5_devices_lock);
	empty = list_empty(&odl_tb5_devices_list);
	mutex_unlock(&odl_tb5_devices_lock);

	if (!empty)
		return;

	pr_info("odl_tb5: still no peer advertising protocol %s/0x%04x — "
		"/dev/odl_tb5_* will not appear until the other machine "
		"also loads odl_tb5 (thunderbolt-net / ThunderboltIP is a "
		"different protocol)\n",
		odl_protocol_mode == 1 ? ODL_TB5_PROTOCOL_KEY_APPLE
				       : ODL_TB5_PROTOCOL_KEY,
		odl_protocol_mode == 1 ? ODL_TB5_PROTOCOL_ID_APPLE
				       : ODL_TB5_PROTOCOL_ID);
}

static DECLARE_DELAYED_WORK(odl_tb5_wait_peer_work, odl_tb5_wait_peer_fn);
#endif /* CONFIG_USB4 */

static int __init odl_tb5_init(void)
{
	int ret;

	if (!is_power_of_2(odl_ring_size) ||
	    odl_ring_size < ODL_TB5_RING_SIZE_MIN ||
	    odl_ring_size > ODL_TB5_RING_SIZE_MAX) {
		pr_err("odl_tb5: invalid odl_ring_size=%u (must be power-of-2, %u-%u)\n",
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

#if IS_ENABLED(CONFIG_USB4)
	odl_tb5_property_dir = tb_property_create_dir(&odl_tb5_proto_uuid);
	if (!odl_tb5_property_dir) {
		pr_err("odl_tb5: failed to create XDomain property directory\n");
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
	if (ret) {
		pr_err("odl_tb5: tb_register_property_dir(%s) failed: %d\n",
		       protocol_key, ret);
		goto err_dir;
	}

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
		if (ret) {
			pr_err("odl_tb5: tb_register_property_dir(%s) "
			       "failed: %d\n", ODL_TB5_PROTOCOL_KEY, ret);
			goto err_dir;
		}
	}

	ret = odl_tb5_proto_register();
	if (ret) {
		pr_err("odl_tb5: failed to register XDomain protocol handler: %d\n",
		       ret);
		goto err_dir;
	}

	ret = tb_register_service_driver(&odl_tb5_driver);
	if (ret) {
		pr_err("odl_tb5: tb_register_service_driver failed: %d\n", ret);
		goto err_proto;
	}

	schedule_delayed_work(&odl_tb5_wait_peer_work,
			      ODL_TB5_WAIT_PEER_SEC * HZ);
	if (odl_bind_any)
		schedule_delayed_work(&odl_tb5_scan_work, 0);
#endif

	odl_tb5_apple_init();

	pr_info("odl_tb5: OdinLink TB5 driver loaded (odl_ring_size=%u, "
		"protocol=%s/0x%04x v%u)\n",
		odl_ring_size,
		odl_protocol_mode == 1 ? ODL_TB5_PROTOCOL_KEY_APPLE
				       : ODL_TB5_PROTOCOL_KEY,
		odl_protocol_mode == 1 ? ODL_TB5_PROTOCOL_ID_APPLE
				       : ODL_TB5_PROTOCOL_ID,
		ODL_TB5_PROTOCOL_VER);
	pr_info("odl_tb5: /dev/odl_tb5_* appears after probe (peer must "
		"advertise the same protocol). Look for "
		"\"odl_tb5: probed device\", then "
		"\"odl_tb5: entering READY state\"\n");

	return 0;

#if IS_ENABLED(CONFIG_USB4)
err_proto:
	odl_tb5_proto_unregister();
err_dir:
	if (odl_tb5_apple_property_dir) {
		tb_property_free_dir(odl_tb5_apple_property_dir);
		odl_tb5_apple_property_dir = NULL;
	}
	tb_property_free_dir(odl_tb5_property_dir);
#endif
err_chardev:
	odl_tb5_chardev_exit();
	return ret;
}

static void __exit odl_tb5_exit(void)
{
	struct odl_tb5_device *dev, *tmp;

#if IS_ENABLED(CONFIG_USB4)
	cancel_delayed_work_sync(&odl_tb5_wait_peer_work);
	cancel_delayed_work_sync(&odl_tb5_scan_work);
#endif

	/* Clean up software loopback devices first (no NHI/hardware deps) */
	if (odl_loopback_count > 0) {
		odl_loopback_exit();
		goto out;
	}

#if IS_ENABLED(CONFIG_USB4)
	/* Unregister first so tb core removes bound services before orphan cleanup. */
	tb_unregister_service_driver(&odl_tb5_driver);
#endif

	odl_tb5_apple_exit();

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

#if IS_ENABLED(CONFIG_USB4)
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
#endif
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
