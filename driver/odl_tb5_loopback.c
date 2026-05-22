// SPDX-License-Identifier: GPL-2.0-only
/*
 * OdinLink Thunderbolt 5 - Software Loopback Device
 *
 * Creates fake OdinLink-Five devices without Thunderbolt hardware.
 * Each device allocates an mmap-able buffer, sets state to READY, and
 * provides a complete ioctl interface. The "peer" is simulated locally.
 *
 * Use: sudo insmod driver/odl_tb5.ko loopback=1
 * Then: /dev/odl_tb5_0 appears, immediately ready.
 *
 * Limitations:
 *   - No real NHI DMA — throughput equals memcpy speed
 *   - Stream sends copy to a software ring buffer
 *   - Completions are immediate (no interrupt latency)
 *   - Use for API/stack testing, not performance measurement
 */

#include "odl_tb5_core.h"
#include <linux/slab.h>
#include <linux/vmalloc.h>

/* ── Loopback per-stream state ──────────────────────────────────────── */

#define LB_STREAM_MAX 256
#define LB_QUEUE_DEPTH 512

struct lb_msg {
	struct list_head  list;
	void             *data;
	uint32_t          len;
	uint8_t           src_id;
};

struct lb_stream {
	bool              active;
	struct list_head  rx_queue;
	spinlock_t        rx_lock;
	wait_queue_head_t rx_wait;
	int               rx_count;
};

/* ── Loopback device instance ───────────────────────────────────────── */

struct lb_device {
	int               index;
	void             *buf;
	size_t            buf_size;
	struct lb_stream  streams[LB_STREAM_MAX];
	struct mutex      stream_lock;
	unsigned long     stream_bitmap[BITS_TO_LONGS(LB_STREAM_MAX)];
};

/* ── Create / Destroy ───────────────────────────────────────────────── */

static struct odl_tb5_device *lb_create(int index)
{
	struct odl_tb5_device *dev;
	struct lb_device *lb;
	int ret;

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return ERR_PTR(-ENOMEM);

	dev->index = index;
	dev->state = ODL_TB5_STATE_DISCONNECTED;
	mutex_init(&dev->state_lock);
	init_waitqueue_head(&dev->state_waitq);
	hash_init(dev->streams);
	ida_init(&dev->stream_ida);
	mutex_init(&dev->stream_lock);
	atomic_set(&dev->open_count, 0);
	atomic_set(&dev->removing, 0);

	/* Char device */
	ret = odl_tb5_chardev_create(dev);
	if (ret)
		goto err_free;

	/* Allocate mmap buffer (4 MB) */
	lb = kzalloc(sizeof(*lb), GFP_KERNEL);
	if (!lb) {
		ret = -ENOMEM;
		goto err_chardev;
	}

	lb->index = index;
	lb->buf_size = ODL_TB5_FRAME_SIZE * 4096; /* 16 MB */
	lb->buf = vmalloc(lb->buf_size);
	if (!lb->buf) {
		ret = -ENOMEM;
		goto err_lb;
	}

	for (int i = 0; i < LB_STREAM_MAX; i++) {
		struct lb_stream *s = &lb->streams[i];
		INIT_LIST_HEAD(&s->rx_queue);
		spin_lock_init(&s->rx_lock);
		init_waitqueue_head(&s->rx_wait);
		s->rx_count = 0;
	}

	mutex_init(&lb->stream_lock);
	dev->loopback_data = lb;

	/* Set READY immediately */
	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_READY;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	pr_info("odl_tb5: loopback device %d ready (buf=%zu MB)\n",
		index, lb->buf_size >> 20);

	return dev;

err_lb:
	kfree(lb);
err_chardev:
	odl_tb5_chardev_destroy(dev);
err_free:
	kfree(dev);
	return ERR_PTR(ret);
}

static void lb_destroy(struct odl_tb5_device *dev)
{
	struct lb_device *lb = dev->loopback_data;
	if (!lb) return;

	vfree(lb->buf);
	mutex_destroy(&lb->stream_lock);
	kfree(lb);
	dev->loopback_data = NULL;

	odl_tb5_chardev_destroy(dev);
	kfree(dev);
}

/* ── Module hooks ───────────────────────────────────────────────────── */

int odl_loopback_init(void)
{
	if (odl_loopback_count <= 0) return 0;
	if (odl_loopback_count > ODL_TB5_MAX_DEVICES)
		odl_loopback_count = ODL_TB5_MAX_DEVICES;

	for (int i = 0; i < odl_loopback_count; i++) {
		struct odl_tb5_device *dev = lb_create(i);
		if (IS_ERR(dev)) continue;
		mutex_lock(&odl_tb5_devices_lock);
		list_add_tail(&dev->list, &odl_tb5_devices_list);
		mutex_unlock(&odl_tb5_devices_lock);
	}
	return 0;
}

void odl_loopback_exit(void)
{
	struct odl_tb5_device *dev, *tmp;
	mutex_lock(&odl_tb5_devices_lock);
	list_for_each_entry_safe(dev, tmp, &odl_tb5_devices_list, list) {
		if (dev->loopback_data) {
			list_del(&dev->list);
			mutex_unlock(&odl_tb5_devices_lock);
			lb_destroy(dev);
			mutex_lock(&odl_tb5_devices_lock);
		}
	}
	mutex_unlock(&odl_tb5_devices_lock);
}
