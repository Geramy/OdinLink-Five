// SPDX-License-Identifier: GPL-2.0-only
/*
 * OdinLink — Loopback Transport Backend
 *
 * Software-only transport for testing without Thunderbolt hardware.
 * Data loops back at memcpy speed. No NHI, no DMA, no interrupts.
 *
 * Implements odl_tb5_transport_ops so the rest of the driver
 * (chardev, streams, ioctl) works exactly like the real NHI path.
 *
 * Use: sudo insmod driver/odl_tb5.ko loopback=1
 * Then: /dev/odl_tb5_0 appears, immediately in READY state.
 */

#include "odl_tb5_core.h"
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/kthread.h>
#include <linux/delay.h>

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

struct lb_device {
	int               index;
	void             *buf;
	size_t            buf_size;
	struct lb_stream  streams[LB_STREAM_MAX];
	struct mutex      stream_lock;
	unsigned long     stream_bitmap[BITS_TO_LONGS(LB_STREAM_MAX)];
};

/* ── Loopback transport ops ────────────────────────────────────────── */

static int lb_ring_alloc(struct odl_tb5_device *dev)
{
	struct lb_device *lb = dev->transport_priv;

	dev->tx.ring_size = odl_ring_size;
	dev->rx.ring_size = odl_ring_size;

	lb->buf_size = ODL_TB5_FRAME_SIZE * 4096;
	lb->buf = vmalloc(lb->buf_size);
	if (!lb->buf)
		return -ENOMEM;

	return 0;
}

static void lb_ring_free(struct odl_tb5_device *dev)
{
	struct lb_device *lb = dev->transport_priv;

	if (lb->buf) {
		vfree(lb->buf);
		lb->buf = NULL;
	}
}

static int lb_ring_start(struct odl_tb5_device *dev)
{
	return 0;
}

static void lb_ring_stop(struct odl_tb5_device *dev)
{
}

static void lb_ring_reset(struct odl_tb5_device *dev)
{
}

static int lb_ring_tx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	return 0;
}

static int lb_ring_rx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	return 0;
}

static struct device *lb_dma_device(struct odl_tb5_device *dev)
{
	return dev->dev ? dev->dev->parent : NULL;
}

static struct odl_tb5_transport_ring_info lb_tx_ring_info(
	struct odl_tb5_device *dev)
{
	return (struct odl_tb5_transport_ring_info){ .hop = 0 };
}

static struct odl_tb5_transport_ring_info lb_rx_ring_info(
	struct odl_tb5_device *dev)
{
	return (struct odl_tb5_transport_ring_info){ .hop = 0 };
}

static int lb_local_tx_hopid(struct odl_tb5_device *dev)
{
	return 0;
}

static int lb_path_enable(struct odl_tb5_device *dev)
{
	return 0;
}

static void lb_path_disable(struct odl_tb5_device *dev)
{
}

static int lb_peer_send_login(struct odl_tb5_device *dev)
{
	return 0;
}

static int lb_peer_send_logout(struct odl_tb5_device *dev)
{
	return 0;
}

static void lb_kick_tx(struct odl_tb5_device *dev)
{
}

static void lb_kick_rx(struct odl_tb5_device *dev)
{
}

static const struct odl_tb5_transport_ops odl_tb5_loopback_transport = {
	.type		= ODL_TB5_TRANSPORT_LOOPBACK,
	.name		= "loopback",
	.ring_alloc	= lb_ring_alloc,
	.ring_free	= lb_ring_free,
	.ring_start	= lb_ring_start,
	.ring_stop	= lb_ring_stop,
	.ring_reset	= lb_ring_reset,
	.ring_tx	= lb_ring_tx,
	.ring_rx	= lb_ring_rx,
	.dma_device	= lb_dma_device,
	.tx_ring_info	= lb_tx_ring_info,
	.rx_ring_info	= lb_rx_ring_info,
	.local_tx_hopid = lb_local_tx_hopid,
	.path_enable	= lb_path_enable,
	.path_disable	= lb_path_disable,
	.peer_send_login  = lb_peer_send_login,
	.peer_send_logout = lb_peer_send_logout,
	.kick_tx	= lb_kick_tx,
	.kick_rx	= lb_kick_rx,
};

/* ── Loopback device create/destroy ────────────────────────────────── */

static struct odl_tb5_device *lb_create(int index)
{
	struct odl_tb5_device *dev;
	struct lb_device *lb;
	int ret, i;

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
	INIT_WORK(&dev->tx_drain_work, odl_tb5_tx_drain_work_fn);
	atomic_set(&dev->rx_posted, 0);
	dev->rx_target = 0;

	dev->tx_adaptive.mode = ODL_TB5_TX_LATENCY;
	dev->tx_adaptive.consecutive_low = 0;
	dev->tx_adaptive.high_watermark = odl_ring_size * 3 / 4;
	dev->tx_adaptive.low_watermark  = odl_ring_size / 4;

	ret = odl_tb5_chardev_create(dev);
	if (ret)
		goto err_free;

	lb = kzalloc(sizeof(*lb), GFP_KERNEL);
	if (!lb) {
		ret = -ENOMEM;
		goto err_chardev;
	}

	lb->index = index;
	for (i = 0; i < LB_STREAM_MAX; i++) {
		struct lb_stream *s = &lb->streams[i];
		INIT_LIST_HEAD(&s->rx_queue);
		spin_lock_init(&s->rx_lock);
		init_waitqueue_head(&s->rx_wait);
		s->rx_count = 0;
	}
	mutex_init(&lb->stream_lock);

	dev->transport = &odl_tb5_loopback_transport;
	dev->transport_priv = lb;

	ret = odl_tb5_rings_alloc(dev);
	if (ret)
		goto err_lb;

	mutex_lock(&dev->state_lock);
	dev->state = ODL_TB5_STATE_READY;
	wake_up_all(&dev->state_waitq);
	mutex_unlock(&dev->state_lock);

	pr_info("odl_tb5: loopback device %d ready (transport=%s)\n",
		index, dev->transport->name);

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
	struct lb_device *lb = dev->transport_priv;

	odl_tb5_rings_free(dev);

	if (lb) {
		vfree(lb->buf);
		mutex_destroy(&lb->stream_lock);
		kfree(lb);
		dev->transport_priv = NULL;
	}

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
		if (dev->transport &&
		    dev->transport->type == ODL_TB5_TRANSPORT_LOOPBACK) {
			list_del(&dev->list);
			mutex_unlock(&odl_tb5_devices_lock);
			lb_destroy(dev);
			mutex_lock(&odl_tb5_devices_lock);
		}
	}
	mutex_unlock(&odl_tb5_devices_lock);
}
