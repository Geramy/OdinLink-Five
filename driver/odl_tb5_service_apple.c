// SPDX-License-Identifier: GPL-2.0-only
/*
 * OdinLink — Apple-only Module Entry Point
 *
 * Used when CONFIG_USB4 is not enabled (aarch64 builds
 * for Apple Silicon without the Intel NHI subsystem). Provides
 * module init/exit that only registers the Apple platform driver
 * and loopback support.
 */

#include "odl_tb5_core.h"

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

	if (odl_loopback_count > 0) {
		ret = odl_loopback_init();
		if (ret)
			goto err_chardev;
		pr_info("odl_tb5: loopback mode enabled (%d devices)\n",
			odl_loopback_count);
		return 0;
	}

	odl_tb5_apple_init();

	pr_info("odl_tb5: OdinLink TB5 driver loaded (apple transport, odl_ring_size=%u)\n",
		odl_ring_size);

	return 0;

err_chardev:
	odl_tb5_chardev_exit();
	return ret;
}

static void __exit odl_tb5_exit(void)
{
	struct odl_tb5_device *dev, *tmp;

	if (odl_loopback_count > 0) {
		odl_loopback_exit();
		goto out;
	}

	odl_tb5_apple_exit();

	mutex_lock(&odl_tb5_devices_lock);
	list_for_each_entry_safe(dev, tmp, &odl_tb5_devices_list, list) {
		pr_warn("odl_tb5: cleaning up orphaned device at exit\n");
		list_del_rcu(&dev->list);
		atomic_set(&dev->removing, 1);
		hrtimer_cancel(&dev->rx_poll_timer);
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
out:
	odl_tb5_chardev_exit();
	ida_destroy(&odl_tb5_ida);
	pr_info("odl_tb5: OdinLink TB5 driver unloaded\n");
}

module_init(odl_tb5_init);
module_exit(odl_tb5_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("OdinLink Team");
MODULE_DESCRIPTION("OdinLink Thunderbolt 5 DMA Ring Driver (Apple Transport)");
MODULE_IMPORT_NS("DMA_BUF");
