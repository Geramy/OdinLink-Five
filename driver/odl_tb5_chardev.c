// SPDX-License-Identifier: MIT
/*
 * OdinLink Thunderbolt 5 - Character Device Interface
 *
 * Provides the /dev/odl_tb5_N nodes for userspace access via stream ioctls,
 * legacy double-buffer ioctls, and mmap.
 * Part of the odl_tb5.ko multi-file module alongside:
 *   odl_tb5_service.c   - Thunderbolt service probe / remove
 *   odl_tb5_ring_dma.c  - NHI ring allocation, DMA frame pool, TX/RX workers
 *   odl_tb5_proto.c     - OdinLink login/logout handshake protocol
 */

#include <linux/uaccess.h>

#include "odl_tb5_core.h"

static dev_t odl_tb5_devt;
static struct class *odl_tb5_class;

static int odl_tb5_open(struct inode *inode, struct file *filp)
{
	struct odl_tb5_device *dev;
	struct odl_tb5_file_ctx *ctx;

	dev = container_of(inode->i_cdev, struct odl_tb5_device, cdev);

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;

	ctx->dev = dev;
	INIT_LIST_HEAD(&ctx->streams);
	spin_lock_init(&ctx->lock);

	filp->private_data = ctx;
	atomic_inc(&dev->open_count);

	return 0;
}

static int odl_tb5_release(struct inode *inode, struct file *filp)
{
	struct odl_tb5_file_ctx *ctx = filp->private_data;
	struct odl_tb5_device *dev = ctx->dev;
	struct odl_tb5_stream *stream, *tmp;

	/* Crash-safe cleanup: destroy all streams owned by this fd */
	spin_lock(&ctx->lock);
	list_for_each_entry_safe(stream, tmp, &ctx->streams, owner_list) {
		spin_unlock(&ctx->lock);
		odl_tb5_stream_destroy(stream);
		spin_lock(&ctx->lock);
	}
	spin_unlock(&ctx->lock);

	kfree(ctx);
	atomic_dec(&dev->open_count);

	return 0;
}

static long odl_tb5_ioctl(struct file *filp, unsigned int cmd,
			   unsigned long arg)
{
	struct odl_tb5_file_ctx *ctx = filp->private_data;
	struct odl_tb5_device *dev = ctx->dev;
	void __user *uarg = (void __user *)arg;

	switch (cmd) {

	/* ── Stream ioctls ──────────────────────────────────────────── */

	case ODL_TB5_IOCTL_STREAM_OPEN: {
		struct odl_tb5_stream_req req;
		struct odl_tb5_stream *stream;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_create(dev, ctx, req.stream_id);
		if (IS_ERR(stream))
			return PTR_ERR(stream);

		req.stream_id = stream->id;
		if (copy_to_user(uarg, &req, sizeof(req))) {
			odl_tb5_stream_destroy(stream);
			return -EFAULT;
		}

		return 0;
	}

	case ODL_TB5_IOCTL_STREAM_CLOSE: {
		struct odl_tb5_stream_req req;
		struct odl_tb5_stream *stream;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		/* Only owner can close */
		if (stream->owner != ctx) {
			odl_tb5_stream_put(stream);
			return -EPERM;
		}

		odl_tb5_stream_put(stream);
		odl_tb5_stream_destroy(stream);
		return 0;
	}

	case ODL_TB5_IOCTL_STREAM_SEND: {
		struct odl_tb5_stream_xfer req;
		struct odl_tb5_stream *stream;
		int ret;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		ret = odl_tb5_stream_send(stream, req.dst_id,
					  (const void __user *)req.data,
					  req.len);
		odl_tb5_stream_put(stream);
		return ret;
	}

	case ODL_TB5_IOCTL_STREAM_RECV: {
		struct odl_tb5_stream_xfer req;
		struct odl_tb5_stream *stream;
		int ret;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		ret = odl_tb5_stream_recv(stream,
					  (void __user *)req.data,
					  req.len,
					  &req.src_id,
					  &req.actual_len);
		odl_tb5_stream_put(stream);

		if (ret)
			return ret;

		if (copy_to_user(uarg, &req, sizeof(req)))
			return -EFAULT;

		return 0;
	}

	case ODL_TB5_IOCTL_STREAM_WAIT_TX: {
		struct odl_tb5_stream_wait req;
		struct odl_tb5_stream *stream;
		int ret;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		ret = odl_tb5_stream_wait_tx(stream, req.timeout_ms);
		odl_tb5_stream_put(stream);
		return ret;
	}

	case ODL_TB5_IOCTL_STREAM_WAIT_RX: {
		struct odl_tb5_stream_wait req;
		struct odl_tb5_stream *stream;
		int ret;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		ret = odl_tb5_stream_wait_rx(stream, req.timeout_ms);
		odl_tb5_stream_put(stream);
		return ret;
	}

	case ODL_TB5_IOCTL_STREAM_SEND_DMABUF: {
		struct odl_tb5_stream_dmabuf req;
		struct odl_tb5_stream *stream;
		int ret;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		ret = odl_tb5_submit_tx_dmabuf(dev, req.dmabuf_fd,
					       req.offset, req.len);
		odl_tb5_stream_put(stream);
		return ret;
	}

	case ODL_TB5_IOCTL_STREAM_RECV_DMABUF: {
		struct odl_tb5_stream_dmabuf req;
		struct odl_tb5_stream *stream;
		int ret;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;

		stream = odl_tb5_stream_lookup(dev, req.stream_id);
		if (!stream)
			return -ENOENT;

		ret = odl_tb5_submit_rx_dmabuf(dev, req.dmabuf_fd,
					       req.offset, req.len);
		odl_tb5_stream_put(stream);
		return ret;
	}

	/* ── Legacy ioctls ──────────────────────────────────────────── */

	case ODL_TB5_IOCTL_SEND: {
		struct odl_tb5_xfer_request req;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (dev->state != ODL_TB5_STATE_CONNECTED &&
		    dev->state != ODL_TB5_STATE_READY)
			return -ENOTCONN;

		return odl_tb5_submit_tx(dev, req.offset, req.len,
					 !!(req.flags & ODL_TB5_XFER_FLAG_CTRL));
	}

	case ODL_TB5_IOCTL_RECV: {
		struct odl_tb5_xfer_request req;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (dev->state != ODL_TB5_STATE_CONNECTED &&
		    dev->state != ODL_TB5_STATE_READY)
			return -ENOTCONN;

		return odl_tb5_submit_rx(dev, req.offset, req.len);
	}

	case ODL_TB5_IOCTL_SEND_DMABUF: {
		struct odl_tb5_ring_request req;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (dev->state != ODL_TB5_STATE_CONNECTED &&
		    dev->state != ODL_TB5_STATE_READY)
			return -ENOTCONN;

		return odl_tb5_submit_tx_dmabuf(dev, req.dmabuf_fd,
						req.offset, req.len);
	}

	case ODL_TB5_IOCTL_RECV_DMABUF: {
		struct odl_tb5_ring_request req;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (dev->state != ODL_TB5_STATE_CONNECTED &&
		    dev->state != ODL_TB5_STATE_READY)
			return -ENOTCONN;

		return odl_tb5_submit_rx_dmabuf(dev, req.dmabuf_fd,
						req.offset, req.len);
	}

	case ODL_TB5_IOCTL_POLL_COMPLETION: {
		struct odl_tb5_completion comp;

		comp.tx_completed = atomic_read(&dev->tx.completed);
		comp.rx_completed = atomic_read(&dev->rx.completed);
		comp.tx_submitted = atomic_read(&dev->tx.submitted);
		comp.rx_submitted = atomic_read(&dev->rx.submitted);

		if (copy_to_user(uarg, &comp, sizeof(comp)))
			return -EFAULT;

		return 0;
	}

	case ODL_TB5_IOCTL_WAIT_TX: {
		struct odl_tb5_completion comp;
		long ret;

		ret = wait_event_interruptible_timeout(dev->tx.waitq,
			atomic_read(&dev->tx.completed) > 0,
			msecs_to_jiffies(30000));
		if (ret == 0)
			return -ETIMEDOUT;
		if (ret < 0)
			return ret;

		comp.tx_completed = atomic_xchg(&dev->tx.completed, 0);
		comp.rx_completed = atomic_read(&dev->rx.completed);
		comp.tx_submitted = atomic_read(&dev->tx.submitted);
		comp.rx_submitted = atomic_read(&dev->rx.submitted);

		if (copy_to_user(uarg, &comp, sizeof(comp)))
			return -EFAULT;

		return 0;
	}

	case ODL_TB5_IOCTL_WAIT_RX: {
		struct odl_tb5_completion comp;
		long ret;

		ret = wait_event_interruptible_timeout(dev->rx.waitq,
			atomic_read(&dev->rx.completed) > 0,
			msecs_to_jiffies(30000));
		if (ret == 0)
			return -ETIMEDOUT;
		if (ret < 0)
			return ret;

		comp.rx_completed = atomic_xchg(&dev->rx.completed, 0);
		comp.tx_completed = atomic_read(&dev->tx.completed);
		comp.tx_submitted = atomic_read(&dev->tx.submitted);
		comp.rx_submitted = atomic_read(&dev->rx.submitted);

		if (copy_to_user(uarg, &comp, sizeof(comp)))
			return -EFAULT;

		return 0;
	}

	case ODL_TB5_IOCTL_GET_PEER: {
		struct odl_tb5_peer_info info;

		memset(&info, 0, sizeof(info));

		if (dev->xd && dev->xd->remote_uuid)
			memcpy(info.uuid, dev->xd->remote_uuid, 16);

		if (dev->xd) {
			info.link_speed = dev->xd->link_speed;
			info.link_width = dev->xd->link_width;
		}

		info.state = dev->state;

		if (dev->xd && dev->xd->vendor_name)
			strscpy(info.vendor_name, dev->xd->vendor_name,
				sizeof(info.vendor_name));

		if (dev->xd && dev->xd->device_name)
			strscpy(info.device_name, dev->xd->device_name,
				sizeof(info.device_name));

		if (copy_to_user(uarg, &info, sizeof(info)))
			return -EFAULT;

		return 0;
	}

	case ODL_TB5_IOCTL_GET_BUF_INFO: {
		struct odl_tb5_buf_info info;

		info.tx_buf_size  = dev->tx.bufs[0].size;
		info.rx_buf_size  = dev->rx.bufs[0].size;
		info.tx_buf_count = ODL_TB5_NUM_BUFFERS;
		info.rx_buf_count = ODL_TB5_NUM_BUFFERS;

		if (copy_to_user(uarg, &info, sizeof(info)))
			return -EFAULT;

		return 0;
	}

	case ODL_TB5_IOCTL_SWAP_TX_BUF:
		spin_lock(&dev->tx.lock);
		swap(dev->tx.front, dev->tx.back);
		dev->tx.swapped_since_post = true;
		spin_unlock(&dev->tx.lock);
		return 0;

	case ODL_TB5_IOCTL_SWAP_RX_BUF:
		spin_lock(&dev->rx.lock);
		swap(dev->rx.front, dev->rx.back);
		dev->rx.swapped_since_post = true;
		spin_unlock(&dev->rx.lock);
		return 0;

	case ODL_TB5_IOCTL_WAIT_READY: {
		uint32_t timeout_ms;
		long ret;

		if (copy_from_user(&timeout_ms, uarg, sizeof(timeout_ms)))
			return -EFAULT;

		if (timeout_ms == 0) {
			ret = wait_event_interruptible(dev->state_waitq,
				dev->state == ODL_TB5_STATE_READY);
		} else {
			ret = wait_event_interruptible_timeout(dev->state_waitq,
				dev->state == ODL_TB5_STATE_READY,
				msecs_to_jiffies(timeout_ms));
			if (ret == 0)
				return -ETIMEDOUT;
		}

		return ret < 0 ? ret : 0;
	}

	default:
		return -ENOTTY;
	}
}

static int odl_tb5_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct odl_tb5_file_ctx *fctx = filp->private_data;
	struct odl_tb5_device *dev = fctx->dev;
	unsigned long mmap_offset;
	struct odl_tb5_dma_buf *buf;
	struct device *dma_dev;
	size_t vma_size;

	mmap_offset = vma->vm_pgoff << PAGE_SHIFT;

	switch (mmap_offset) {
	case ODL_TB5_MMAP_TX_BUF0:
		buf = &dev->tx.bufs[0];
		break;
	case ODL_TB5_MMAP_TX_BUF1:
		buf = &dev->tx.bufs[1];
		break;
	case ODL_TB5_MMAP_RX_BUF0:
		buf = &dev->rx.bufs[0];
		break;
	case ODL_TB5_MMAP_RX_BUF1:
		buf = &dev->rx.bufs[1];
		break;
	default:
		return -EINVAL;
	}

	vma_size = vma->vm_end - vma->vm_start;
	if (vma_size > buf->size)
		return -EINVAL;

	vma->vm_pgoff = 0;

	dma_dev = tb_ring_dma_device(dev->tx.ring);

	return dma_mmap_coherent(dma_dev, vma, buf->virt, buf->phys,
				 buf->size);
}

static const struct file_operations odl_tb5_fops = {
	.owner		= THIS_MODULE,
	.open		= odl_tb5_open,
	.release	= odl_tb5_release,
	.unlocked_ioctl	= odl_tb5_ioctl,
	.mmap		= odl_tb5_mmap,
};

int odl_tb5_chardev_create(struct odl_tb5_device *dev)
{
	int ret;

	dev->devt = MKDEV(MAJOR(odl_tb5_devt), dev->index);

	cdev_init(&dev->cdev, &odl_tb5_fops);
	dev->cdev.owner = THIS_MODULE;

	ret = cdev_add(&dev->cdev, dev->devt, 1);
	if (ret) {
		pr_err("odl_tb5: cdev_add failed for index %d: %d\n",
		       dev->index, ret);
		return ret;
	}

	dev->dev = device_create(odl_tb5_class, NULL, dev->devt, dev,
				 "%s_%d", ODL_TB5_DEVICE_NAME, dev->index);
	if (IS_ERR(dev->dev)) {
		ret = PTR_ERR(dev->dev);
		pr_err("odl_tb5: device_create failed for index %d: %d\n",
		       dev->index, ret);
		goto err_cdev_del;
	}

	return 0;

err_cdev_del:
	cdev_del(&dev->cdev);
	return ret;
}

void odl_tb5_chardev_destroy(struct odl_tb5_device *dev)
{
	device_destroy(odl_tb5_class, dev->devt);
	cdev_del(&dev->cdev);
}

int odl_tb5_chardev_init(void)
{
	int ret;

	ret = alloc_chrdev_region(&odl_tb5_devt, 0, ODL_TB5_MAX_DEVICES,
				  ODL_TB5_DEVICE_NAME);
	if (ret) {
		pr_err("odl_tb5: alloc_chrdev_region failed: %d\n", ret);
		return ret;
	}

	odl_tb5_class = class_create_compat(ODL_TB5_DEVICE_NAME);
	if (IS_ERR(odl_tb5_class)) {
		ret = PTR_ERR(odl_tb5_class);
		pr_err("odl_tb5: class_create failed: %d\n", ret);
		goto err_unregister;
	}

	return 0;

err_unregister:
	unregister_chrdev_region(odl_tb5_devt, ODL_TB5_MAX_DEVICES);
	return ret;
}

void odl_tb5_chardev_exit(void)
{
	class_destroy(odl_tb5_class);
	unregister_chrdev_region(odl_tb5_devt, ODL_TB5_MAX_DEVICES);
}
