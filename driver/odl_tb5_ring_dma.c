// SPDX-License-Identifier: MIT
/*
 * OdinLink — The DMA Engine: Sending and Receiving Packets
 *
 * The Thunderbolt NHI (Native Host Interface) gives us a ring of fixed-size
 * DMA slots — think of it like a circular conveyor belt of 4KB bins. You
 * drop data into a bin on the TX belt, the hardware ships it across the
 * cable, and the other end picks it up from their RX belt.
 *
 * This file handles:
 *   - Allocating those DMA rings and the buffer memory behind them
 *   - A "frame pool" of reusable 4KB slots (no re-allocation between sends)
 *   - Two send modes: latency (one slot at a time, low delay) and throughput
 *     (big 256KB batches, high bandwidth)
 *   - RX assembly — the other side may split a message across multiple 4KB
 *     frames; this reconstructs them into a single buffer
 *   - Callbacks that fire when the hardware finishes a TX or completes an RX
 *
 * Also handles DMA-buf (GPU memory) transfers for zero-copy GPU→GPU.
 */

#include "odl_tb5_core.h"

#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/math.h>
#include <linux/vmalloc.h>

/* Forward declarations for functions defined later in this file */
static void odl_tb5_stream_free(struct kref *ref);

/*
 * High-resolution fallback poll timer — kicks both TX and RX ring_work
 * every 50 us.
 *
 * NHI MSI-X interrupts DO fire, but ring_work triggered by the ISR
 * sometimes doesn't see completions yet (descriptor write-back delay).
 * This timer ensures completions are processed within ~50 us instead of
 * waiting for the next jiffy tick.
 *
 * schedule_work is idempotent, so ISR-driven and timer-driven kicks
 * are safely additive.
 */

enum hrtimer_restart odl_tb5_rx_poll_timer_fn(struct hrtimer *timer)
{
	struct odl_tb5_device *dev =
		container_of(timer, struct odl_tb5_device, rx_poll_timer);

	if (atomic_read(&dev->removing))
		return HRTIMER_NORESTART;

	if (dev->tx.ring && dev->tx.started)
		schedule_work(&dev->tx.ring->work);

	if (dev->rx.ring && dev->rx.started)
		schedule_work(&dev->rx.ring->work);

	if (dev->state >= ODL_TB5_STATE_CONNECTED &&
	    (dev->tx.started || dev->rx.started)) {
		hrtimer_forward_now(timer,
				    ns_to_ktime(ODL_TB5_POLL_INTERVAL_NS));
		return HRTIMER_RESTART;
	}

	return HRTIMER_NORESTART;
}

/* Find which odl_tb5_device owns a given tb_ring and return its ring_ctx. */
static struct odl_tb5_ring_ctx *
odl_tb5_ring_to_ctx(struct tb_ring *ring)
{
	struct odl_tb5_device *dev;

	list_for_each_entry_rcu(dev, &odl_tb5_devices_list, list) {
		if (dev->tx.ring == ring)
			return &dev->tx;
		if (dev->rx.ring == ring)
			return &dev->rx;
	}

	return NULL;
}

/* Given an RX tb_ring pointer, return the owning odl_tb5_device. */
struct odl_tb5_device *
odl_tb5_rx_ring_to_dev(struct tb_ring *ring)
{
	struct odl_tb5_device *dev;

	list_for_each_entry_rcu(dev, &odl_tb5_devices_list, list) {
		if (dev->rx.ring == ring)
			return dev;
	}

	return NULL;
}

void odl_tb5_tx_callback(struct tb_ring *ring,
			 struct ring_frame *frame, bool canceled)
{
	struct odl_tb5_ring_ctx *ctx;
	struct odl_tb5_frame_slot *slot;
	struct odl_tb5_tx_msg *msg;
	struct odl_tb5_device *dev;

	ctx = odl_tb5_ring_to_ctx(ring);
	if (WARN_ON_ONCE(!ctx))
		return;

	/* Check if this is a frame pool slot (new stream path) */
	dev = container_of(ctx, struct odl_tb5_device, tx);

	if (atomic_read(&dev->removing))
		return;
	slot = container_of(frame, struct odl_tb5_frame_slot, frame);

	if (slot >= dev->frame_pool.slots &&
	    slot < dev->frame_pool.slots + dev->frame_pool.size) {
		msg = slot->tx_msg;
		odl_tb5_frame_pool_put(&dev->frame_pool, slot);

		if (msg) {
			if (atomic_dec_and_test(&msg->frames_pending) &&
			    msg->sent == msg->len) {
				struct odl_tb5_stream *s = msg->stream;

				atomic_inc(&s->tx_completed);
				atomic_dec(&s->tx_in_flight);
				wake_up_interruptible(&s->tx_waitq);
				kfree(msg);
			}
		}

		if (canceled)
			return;
	} else {
		/* Legacy path for proto layer direct ring submissions */
		if (canceled) {
			pr_debug("odl_tb5: TX callback canceled\n");
			return;
		}

		pr_debug("odl_tb5: TX complete frame=%px size=%u\n",
			 frame, frame->size);

		atomic_inc(&ctx->completed);
		wake_up_interruptible(&ctx->waitq);
	}
}

void odl_tb5_tx_batch_callback(struct tb_ring *ring,
			       struct ring_frame *frame, bool canceled)
{
	struct odl_tb5_ring_ctx *ctx;
	struct odl_tb5_device *dev;
	struct odl_tb5_batch_buf *batch = NULL;
	struct odl_tb5_tx_msg *msg;
	int b;

	ctx = odl_tb5_ring_to_ctx(ring);
	if (WARN_ON_ONCE(!ctx))
		return;

	dev = container_of(ctx, struct odl_tb5_device, tx);

	if (atomic_read(&dev->removing))
		return;

	/* Identify which batch buffer owns this frame (8 entries max) */
	for (b = 0; b < ODL_TB5_BATCH_BUF_COUNT; b++) {
		struct odl_tb5_batch_buf *candidate = &dev->batch_pool.bufs[b];

		if (frame >= &candidate->frames[0] &&
		    frame < &candidate->frames[ODL_TB5_BATCH_FRAMES]) {
			batch = candidate;
			break;
		}
	}

	if (WARN_ON_ONCE(!batch))
		return;

	msg = batch->tx_msg;

	/* Return batch buffer to pool when all its frames complete */
	if (atomic_dec_and_test(&batch->frames_pending))
		odl_tb5_batch_pool_put(&dev->batch_pool, batch);

	/* Complete the message when all batches are done */
	if (msg && atomic_dec_and_test(&msg->frames_pending) &&
	    msg->sent == msg->len) {
		struct odl_tb5_stream *s = msg->stream;

		atomic_inc(&s->tx_completed);
		atomic_dec(&s->tx_in_flight);
		wake_up_interruptible(&s->tx_waitq);
		kfree(msg);
	}
}

void odl_tb5_rx_callback(struct tb_ring *ring,
			 struct ring_frame *frame, bool canceled)
{
	struct odl_tb5_ring_ctx *ctx;
	struct odl_tb5_device *dev;
	struct odl_tb5_frame_slot *slot;

	ctx = odl_tb5_ring_to_ctx(ring);
	if (WARN_ON_ONCE(!ctx))
		return;

	dev = odl_tb5_rx_ring_to_dev(ring);

	if (!dev || atomic_read(&dev->removing))
		return;

	/* Check if this is a frame pool slot (new stream path) */
	if (dev && dev->frame_pool.slots) {
		slot = container_of(frame, struct odl_tb5_frame_slot, frame);

		if (slot >= dev->frame_pool.slots &&
		    slot < dev->frame_pool.slots + dev->frame_pool.size) {
			void *data = slot->virt;

			atomic_dec(&dev->rx_posted);

			if (canceled) {
				odl_tb5_frame_pool_put(&dev->frame_pool, slot);
				return;
			}

			/* First check for raw DMA control message
			 * (no stream header — used during verify). */
			if (frame->size >= sizeof(struct odl_tb5_dma_hdr)) {
				__le32 raw_magic;

				memcpy(&raw_magic, data, sizeof(raw_magic));
				if (le32_to_cpu(raw_magic) == ODL_TB5_DMA_MAGIC) {
					struct odl_tb5_dma_hdr *dhdr = data;
					u32 type = le32_to_cpu(dhdr->type);

					if (type == ODL_TB5_DMA_PONG) {
						pr_info("OdinLink: DMA pong received (pool)\n");
						dev->pong_received = true;
						wake_up_interruptible(&dev->verify_waitq);
					} else {
						dev->verify_rx_type = type;
						schedule_work(&dev->ctrl_reply_work);
					}

					odl_tb5_frame_pool_put(&dev->frame_pool, slot);
					odl_tb5_rx_repost(dev);
					return;
				}
			}

			/* Check for stream header */
			if (frame->size >= ODL_TB5_STREAM_HDR_SIZE) {
				struct odl_tb5_stream_hdr *shdr = data;
				u8 dst_id = shdr->dst_id;

				if (dst_id != ODL_TB5_STREAM_ID_CTRL) {
					struct odl_tb5_stream *stream;
					u16 payload_len = le16_to_cpu(shdr->payload_len);

					stream = odl_tb5_stream_lookup(dev, dst_id);
					if (stream && payload_len > 0) {
						const void *payload =
							data + ODL_TB5_STREAM_HDR_SIZE;
						u8 flags = shdr->flags;

						/* Start of new message — reset assembly */
						if (flags & ODL_TB5_SHDR_F_MSG_START) {
							kfree(stream->rx_asm_buf);
							stream->rx_asm_buf = NULL;
							stream->rx_asm_len = 0;
							stream->rx_asm_cap = 0;
							stream->rx_asm_src_id = shdr->src_id;
						}

						/* Append payload to assembly buffer */
						if (stream->rx_asm_len + payload_len >
						    stream->rx_asm_cap) {
							size_t new_cap = max_t(size_t,
								8192,
								max(stream->rx_asm_cap * 2,
								    stream->rx_asm_len +
								    payload_len));
							void *nb = kmalloc(new_cap,
									   GFP_ATOMIC);
							if (nb) {
								if (stream->rx_asm_buf)
									memcpy(nb,
									       stream->rx_asm_buf,
									       stream->rx_asm_len);
								kfree(stream->rx_asm_buf);
								stream->rx_asm_buf = nb;
								stream->rx_asm_cap = new_cap;
							}
						}
						if (stream->rx_asm_buf &&
						    stream->rx_asm_len + payload_len <=
						    stream->rx_asm_cap) {
							memcpy(stream->rx_asm_buf +
							       stream->rx_asm_len,
							       payload, payload_len);
							stream->rx_asm_len += payload_len;
						}

						/* End of message — enqueue complete msg */
						if (flags & ODL_TB5_SHDR_F_MSG_END) {
							struct odl_tb5_rx_msg *rxm;
							unsigned long rxflags;

							rxm = kzalloc(sizeof(*rxm),
								      GFP_ATOMIC);
							if (rxm && stream->rx_asm_buf) {
								rxm->data = stream->rx_asm_buf;
								rxm->len = stream->rx_asm_len;
								rxm->src_id =
									stream->rx_asm_src_id;
								rxm->flags =
									ODL_TB5_SHDR_F_MSG_END;

								spin_lock_irqsave(
									&stream->rx_lock,
									rxflags);
								if (stream->rx_queue_len <
								    stream->rx_queue_max) {
									list_add_tail(
										&rxm->list,
										&stream->rx_queue);
									stream->rx_queue_len++;
									spin_unlock_irqrestore(
										&stream->rx_lock,
										rxflags);
									atomic_inc(
										&stream->rx_complete);
									wake_up_interruptible(
										&stream->rx_waitq);
								} else {
									spin_unlock_irqrestore(
										&stream->rx_lock,
										rxflags);
									kfree(rxm->data);
									kfree(rxm);
								}

								/* Buffer ownership transferred
								 * to rx_msg (or freed above) */
								stream->rx_asm_buf = NULL;
								stream->rx_asm_len = 0;
								stream->rx_asm_cap = 0;
							} else {
								kfree(rxm);
								kfree(stream->rx_asm_buf);
								stream->rx_asm_buf = NULL;
								stream->rx_asm_len = 0;
								stream->rx_asm_cap = 0;
							}
						}

						kref_put(&stream->refcount,
							 odl_tb5_stream_free);
					}
				}
			}

			odl_tb5_frame_pool_put(&dev->frame_pool, slot);
			odl_tb5_rx_repost(dev);
			return;
		}
	}

	/* Legacy path for proto layer direct ring submissions */
	if (canceled) {
		pr_debug("odl_tb5: RX callback canceled\n");
		return;
	}

	if (dev && dev->state >= ODL_TB5_STATE_CONNECTED) {
		int idx = frame - ctx->frames;
		void *data = ctx->bufs[ctx->posted_buf].virt +
			     ((size_t)idx * ODL_TB5_FRAME_SIZE);
		__le32 magic;

		memcpy(&magic, data, sizeof(magic));
		if (le32_to_cpu(magic) == ODL_TB5_DMA_MAGIC) {
			struct odl_tb5_dma_hdr *hdr = data;
			u32 type = le32_to_cpu(hdr->type);

			if (type == ODL_TB5_DMA_PONG) {
				pr_info("OdinLink: DMA pong received\n");
				dev->pong_received = true;
				wake_up_interruptible(&dev->verify_waitq);
			} else {
				dev->verify_rx_type = type;
				schedule_work(&dev->ctrl_reply_work);
			}
			return;
		}
	}

	atomic_inc(&ctx->completed);
	wake_up_interruptible(&ctx->waitq);
}

int odl_tb5_rings_alloc(struct odl_tb5_device *dev)
{
	struct tb_xdomain *xd = dev->xd;
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
	dev->local_tx_hopid = ret;

	unsigned int ring_flags = RING_FLAG_FRAME;
	if (odl_e2e)
		ring_flags |= RING_FLAG_E2E;

	dev->tx.ring = tb_ring_alloc_tx(xd->tb->nhi, -1,
					rs,
					ring_flags);
	if (!dev->tx.ring) {
		pr_err("odl_tb5: failed to allocate TX ring\n");
		ret = -ENOMEM;
		goto err_free_hopid;
	}

	sof_mask = BIT(ODL_TB5_PDF_SOF_DATA);
	eof_mask = BIT(ODL_TB5_PDF_EOF_DATA);

	dev->rx.ring = tb_ring_alloc_rx(xd->tb->nhi, -1,
					rs,
					ring_flags,
					dev->tx.ring->hop,
					sof_mask, eof_mask,
					NULL, NULL);
	if (!dev->rx.ring) {
		pr_err("odl_tb5: failed to allocate RX ring\n");
		ret = -ENOMEM;
		goto err_free_tx_ring;
	}

	pr_info("odl_tb5: rings allocated: TX hop=%d, RX hop=%d, "
		"local_tx_hopid=%d (E2E enabled, e2e_tx_hop=%d)\n",
		dev->tx.ring->hop, dev->rx.ring->hop,
		dev->local_tx_hopid, dev->tx.ring->hop);

	spin_lock_init(&dev->tx.lock);
	spin_lock_init(&dev->rx.lock);
	atomic_set(&dev->tx.completed, 0);
	atomic_set(&dev->tx.submitted, 0);
	atomic_set(&dev->rx.completed, 0);
	atomic_set(&dev->rx.submitted, 0);
	init_waitqueue_head(&dev->tx.waitq);
	init_waitqueue_head(&dev->rx.waitq);

	return 0;

err_free_tx_ring:
	tb_ring_free(dev->tx.ring);
	dev->tx.ring = NULL;
err_free_hopid:
	tb_xdomain_release_out_hopid(xd, dev->local_tx_hopid);
	dev->local_tx_hopid = -1;
err_free_rx_frames:
	kvfree(dev->rx.frames);
	dev->rx.frames = NULL;
err_free_tx_frames:
	kvfree(dev->tx.frames);
	dev->tx.frames = NULL;
	return ret;
}

void odl_tb5_rings_free(struct odl_tb5_device *dev)
{
	if (dev->rx.ring) {
		tb_ring_free(dev->rx.ring);
		dev->rx.ring = NULL;
	}

	if (dev->tx.ring) {
		tb_ring_free(dev->tx.ring);
		dev->tx.ring = NULL;
	}

	if (dev->local_tx_hopid >= 0) {
		tb_xdomain_release_out_hopid(dev->xd, dev->local_tx_hopid);
		dev->local_tx_hopid = -1;
	}

	kvfree(dev->tx.frames);
	dev->tx.frames = NULL;
	kvfree(dev->rx.frames);
	dev->rx.frames = NULL;
}

int odl_tb5_rings_start(struct odl_tb5_device *dev)
{
	tb_ring_start(dev->tx.ring);
	tb_ring_start(dev->rx.ring);
	dev->tx.started = true;
	dev->rx.started = true;
	return 0;
}

void odl_tb5_rings_stop(struct odl_tb5_device *dev)
{
	if (dev->tx.ring && dev->tx.started) {
		tb_ring_stop(dev->tx.ring);
		dev->tx.started = false;
		dev->tx.frames_posted = false;
	}

	if (dev->rx.ring && dev->rx.started) {
		tb_ring_stop(dev->rx.ring);
		dev->rx.started = false;
		dev->rx.frames_posted = false;
	}
}

/* Reset both rings to a clean state after kernel verification. */
void odl_tb5_rings_reset(struct odl_tb5_device *dev)
{
	if (dev->tx.ring && dev->tx.started) {
		tb_ring_stop(dev->tx.ring);
		tb_ring_start(dev->tx.ring);
		dev->tx.frames_posted = false;
		dev->tx.swapped_since_post = false;
		atomic_set(&dev->tx.completed, 0);
		atomic_set(&dev->tx.submitted, 0);
	}

	if (dev->rx.ring && dev->rx.started) {
		tb_ring_stop(dev->rx.ring);
		tb_ring_start(dev->rx.ring);
		dev->rx.frames_posted = false;
		dev->rx.swapped_since_post = false;
		atomic_set(&dev->rx.completed, 0);
		atomic_set(&dev->rx.submitted, 0);
	}
}

int odl_tb5_dma_bufs_alloc(struct odl_tb5_device *dev)
{
	struct device *dma_dev;
	size_t buf_size;
	int i;

	dma_dev = tb_ring_dma_device(dev->tx.ring);
	buf_size = (size_t)ODL_TB5_FRAME_SIZE * dev->tx.ring_size;

	for (i = 0; i < ODL_TB5_NUM_BUFFERS; i++) {
		dev->tx.bufs[i].size = buf_size;
		dev->tx.bufs[i].virt = dma_alloc_coherent(dma_dev, buf_size,
							  &dev->tx.bufs[i].phys,
							  GFP_KERNEL);
		if (!dev->tx.bufs[i].virt) {
			pr_err("odl_tb5: failed to alloc TX DMA buf %d (%zu bytes)\n",
			       i, buf_size);
			goto err_free;
		}

		dev->rx.bufs[i].size = buf_size;
		dev->rx.bufs[i].virt = dma_alloc_coherent(dma_dev, buf_size,
							  &dev->rx.bufs[i].phys,
							  GFP_KERNEL);
		if (!dev->rx.bufs[i].virt) {
			pr_err("odl_tb5: failed to alloc RX DMA buf %d (%zu bytes)\n",
			       i, buf_size);
			goto err_free;
		}
	}

	dev->tx.front = 0;
	dev->tx.back  = 1;
	dev->rx.front = 0;
	dev->rx.back  = 1;
	dev->tx.frames_posted = false;
	dev->tx.swapped_since_post = false;
	dev->rx.frames_posted = false;
	dev->rx.swapped_since_post = false;

	return 0;

err_free:
	odl_tb5_dma_bufs_free(dev);
	return -ENOMEM;
}

void odl_tb5_dma_bufs_free(struct odl_tb5_device *dev)
{
	struct device *dma_dev;
	int i;

	if (dev->tx.ring)
		dma_dev = tb_ring_dma_device(dev->tx.ring);
	else if (dev->rx.ring)
		dma_dev = tb_ring_dma_device(dev->rx.ring);
	else
		return;

	for (i = 0; i < ODL_TB5_NUM_BUFFERS; i++) {
		if (dev->tx.bufs[i].virt) {
			dma_free_coherent(dma_dev,
					  dev->tx.bufs[i].size,
					  dev->tx.bufs[i].virt,
					  dev->tx.bufs[i].phys);
			dev->tx.bufs[i].virt = NULL;
		}

		if (dev->rx.bufs[i].virt) {
			dma_free_coherent(dma_dev,
					  dev->rx.bufs[i].size,
					  dev->rx.bufs[i].virt,
					  dev->rx.bufs[i].phys);
			dev->rx.bufs[i].virt = NULL;
		}
	}
}

int odl_tb5_submit_tx(struct odl_tb5_device *dev,
		      size_t offset, size_t len, bool ctrl)
{
	struct odl_tb5_dma_buf *buf;
	size_t remaining;
	int nframes, i, ret;

	if (dev->state != ODL_TB5_STATE_CONNECTED &&
	    dev->state != ODL_TB5_STATE_READY &&
	    !dev->tx.started)
		return -ENOTCONN;

	if (dev->tx.frames_posted) {
		int sub = atomic_read(&dev->tx.submitted);
		long tw = wait_event_interruptible_timeout(dev->tx.waitq,
				atomic_read(&dev->tx.completed) >= sub,
				msecs_to_jiffies(5000));
		if (tw <= 0) {
			pr_warn("odl_tb5: TX drain timeout (%ld), "
				"resetting ring\n", tw);
			tb_ring_stop(dev->tx.ring);
			tb_ring_start(dev->tx.ring);
		}
		dev->tx.frames_posted = false;
		dev->tx.swapped_since_post = false;
		atomic_set(&dev->tx.completed, 0);
		atomic_set(&dev->tx.submitted, 0);
	}

	buf = &dev->tx.bufs[dev->tx.front];

	if (offset + len > buf->size)
		return -EINVAL;

	nframes = DIV_ROUND_UP(len, ODL_TB5_FRAME_SIZE);
	remaining = len;

	for (i = 0; i < nframes; i++) {
		struct ring_frame *frame = &dev->tx.frames[i];

		frame->buffer_phy = buf->phys + offset +
				    ((size_t)i * ODL_TB5_FRAME_SIZE);
		frame->size = min_t(size_t, ODL_TB5_FRAME_SIZE, remaining);
		frame->callback = odl_tb5_tx_callback;
		frame->sof = ctrl ? ODL_TB5_PDF_SOF_CTRL : ODL_TB5_PDF_SOF_DATA;
		frame->eof = ctrl ? ODL_TB5_PDF_EOF_CTRL : ODL_TB5_PDF_EOF_DATA;

		ret = tb_ring_tx(dev->tx.ring, frame);
		if (ret < 0) {
			pr_err("odl_tb5: tb_ring_tx failed at frame %d: %d\n",
			       i, ret);
			return ret;
		}

		remaining -= frame->size;
	}

	atomic_add(nframes, &dev->tx.submitted);
	dev->tx.frames_posted = true;

	pr_debug("odl_tb5: TX submitted %d frames, offset=%zu len=%zu ctrl=%d "
		"buf_phys=%pad ring_hop=%d\n",
		nframes, offset, len, ctrl,
		&buf->phys, dev->tx.ring->hop);

	return 0;
}

int odl_tb5_submit_rx(struct odl_tb5_device *dev,
		      size_t offset, size_t len)
{
	struct odl_tb5_dma_buf *buf;
	size_t remaining;
	int nframes, i, ret;

	if (dev->state != ODL_TB5_STATE_CONNECTED &&
	    dev->state != ODL_TB5_STATE_READY &&
	    !dev->rx.started)
		return -ENOTCONN;

	if (dev->rx.frames_posted) {
		if (!dev->rx.swapped_since_post)
			return 0;

		tb_ring_stop(dev->rx.ring);
		tb_ring_start(dev->rx.ring);
		dev->rx.frames_posted = false;
		dev->rx.swapped_since_post = false;
		atomic_set(&dev->rx.completed, 0);
		atomic_set(&dev->rx.submitted, 0);
	}

	buf = &dev->rx.bufs[dev->rx.front];

	if (offset + len > buf->size)
		return -EINVAL;

	nframes = DIV_ROUND_UP(len, ODL_TB5_FRAME_SIZE);
	remaining = len;

	for (i = 0; i < nframes; i++) {
		struct ring_frame *frame = &dev->rx.frames[i];

		frame->buffer_phy = buf->phys + offset +
				    ((size_t)i * ODL_TB5_FRAME_SIZE);
		frame->size = min_t(size_t, ODL_TB5_FRAME_SIZE, remaining);
		frame->callback = odl_tb5_rx_callback;
		frame->sof = ODL_TB5_PDF_SOF_DATA;
		frame->eof = ODL_TB5_PDF_EOF_DATA;

		ret = tb_ring_rx(dev->rx.ring, frame);
		if (ret < 0) {
			pr_err("odl_tb5: tb_ring_rx failed at frame %d: %d\n",
			       i, ret);
			return ret;
		}

		remaining -= frame->size;
	}

	atomic_add(nframes, &dev->rx.submitted);
	dev->rx.frames_posted = true;
	dev->rx.posted_buf = dev->rx.front;

	pr_debug("odl_tb5: RX submitted %d frames, offset=%zu len=%zu "
		"buf_phys=%pad ring_hop=%d\n",
		nframes, offset, len,
		&buf->phys, dev->rx.ring->hop);

	return 0;
}

int odl_tb5_submit_tx_dmabuf(struct odl_tb5_device *dev,
			     int dmabuf_fd, loff_t offset, size_t len)
{
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct scatterlist *sg;
	dma_addr_t sg_addr;
	size_t sg_remaining, chunk;
	size_t total_remaining;
	loff_t skip;
	int frame_idx = 0;
	int nents_i;
	int ret = 0;

	if (dev->state != ODL_TB5_STATE_CONNECTED &&
	    dev->state != ODL_TB5_STATE_READY)
		return -ENOTCONN;

	dmabuf = dma_buf_get(dmabuf_fd);
	if (IS_ERR(dmabuf))
		return PTR_ERR(dmabuf);

	if (offset < 0 || len == 0 ||
	    (size_t)offset + len > dmabuf->size) {
		ret = -EINVAL;
		goto err_put;
	}

	attach = dma_buf_attach(dmabuf, dev->dev);
	if (IS_ERR(attach)) {
		ret = PTR_ERR(attach);
		goto err_put;
	}

	sgt = dma_buf_map_attachment(attach, DMA_TO_DEVICE);
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		goto err_detach;
	}

	skip = offset;
	total_remaining = len;

	for_each_sgtable_dma_sg(sgt, sg, nents_i) {
		sg_addr = sg_dma_address(sg);
		sg_remaining = sg_dma_len(sg);

		if (skip > 0) {
			if ((size_t)skip >= sg_remaining) {
				skip -= sg_remaining;
				continue;
			}
			sg_addr += skip;
			sg_remaining -= skip;
			skip = 0;
		}

		while (sg_remaining > 0 && total_remaining > 0) {
			struct ring_frame *frame;

			if (frame_idx >= dev->tx.ring_size) {
				ret = -ENOSPC;
				goto err_unmap;
			}

			frame = &dev->tx.frames[frame_idx];

			chunk = min3((size_t)ODL_TB5_FRAME_SIZE,
				     sg_remaining, total_remaining);

			frame->buffer_phy = sg_addr;
			frame->size = chunk;
			frame->callback = odl_tb5_tx_callback;
			frame->sof = ODL_TB5_PDF_SOF_DATA;
			frame->eof = ODL_TB5_PDF_EOF_DATA;

			ret = tb_ring_tx(dev->tx.ring, frame);
			if (ret < 0)
				goto err_unmap;

			sg_addr += chunk;
			sg_remaining -= chunk;
			total_remaining -= chunk;
			frame_idx++;
		}

		if (total_remaining == 0)
			break;
	}

	atomic_add(frame_idx, &dev->tx.submitted);

	wait_event_interruptible(dev->tx.waitq,
		atomic_read(&dev->tx.completed) >=
		atomic_read(&dev->tx.submitted));

	dma_buf_unmap_attachment(attach, sgt, DMA_TO_DEVICE);
	dma_buf_detach(dmabuf, attach);
	dma_buf_put(dmabuf);
	return 0;

err_unmap:
	dma_buf_unmap_attachment(attach, sgt, DMA_TO_DEVICE);
err_detach:
	dma_buf_detach(dmabuf, attach);
err_put:
	dma_buf_put(dmabuf);
	return ret;
}

int odl_tb5_submit_rx_dmabuf(struct odl_tb5_device *dev,
			     int dmabuf_fd, loff_t offset, size_t len)
{
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct scatterlist *sg;
	dma_addr_t sg_addr;
	size_t sg_remaining, chunk;
	size_t total_remaining;
	loff_t skip;
	int frame_idx = 0;
	int nents_i;
	int ret = 0;

	if (dev->state != ODL_TB5_STATE_CONNECTED &&
	    dev->state != ODL_TB5_STATE_READY)
		return -ENOTCONN;

	dmabuf = dma_buf_get(dmabuf_fd);
	if (IS_ERR(dmabuf))
		return PTR_ERR(dmabuf);

	if (offset < 0 || len == 0 ||
	    (size_t)offset + len > dmabuf->size) {
		ret = -EINVAL;
		goto err_put;
	}

	attach = dma_buf_attach(dmabuf, dev->dev);
	if (IS_ERR(attach)) {
		ret = PTR_ERR(attach);
		goto err_put;
	}

	sgt = dma_buf_map_attachment(attach, DMA_FROM_DEVICE);
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		goto err_detach;
	}

	skip = offset;
	total_remaining = len;

	for_each_sgtable_dma_sg(sgt, sg, nents_i) {
		sg_addr = sg_dma_address(sg);
		sg_remaining = sg_dma_len(sg);

		if (skip > 0) {
			if ((size_t)skip >= sg_remaining) {
				skip -= sg_remaining;
				continue;
			}
			sg_addr += skip;
			sg_remaining -= skip;
			skip = 0;
		}

		while (sg_remaining > 0 && total_remaining > 0) {
			struct ring_frame *frame;

			if (frame_idx >= dev->rx.ring_size) {
				ret = -ENOSPC;
				goto err_unmap;
			}

			frame = &dev->rx.frames[frame_idx];

			chunk = min3((size_t)ODL_TB5_FRAME_SIZE,
				     sg_remaining, total_remaining);

			frame->buffer_phy = sg_addr;
			frame->size = chunk;
			frame->callback = odl_tb5_rx_callback;
			frame->sof = ODL_TB5_PDF_SOF_DATA;
			frame->eof = ODL_TB5_PDF_EOF_DATA;

			ret = tb_ring_rx(dev->rx.ring, frame);
			if (ret < 0)
				goto err_unmap;

			sg_addr += chunk;
			sg_remaining -= chunk;
			total_remaining -= chunk;
			frame_idx++;
		}

		if (total_remaining == 0)
			break;
	}

	atomic_add(frame_idx, &dev->rx.submitted);

	wait_event_interruptible(dev->rx.waitq,
		atomic_read(&dev->rx.completed) >=
		atomic_read(&dev->rx.submitted));

	dma_buf_unmap_attachment(attach, sgt, DMA_FROM_DEVICE);
	dma_buf_detach(dmabuf, attach);
	dma_buf_put(dmabuf);
	return 0;

err_unmap:
	dma_buf_unmap_attachment(attach, sgt, DMA_FROM_DEVICE);
err_detach:
	dma_buf_detach(dmabuf, attach);
err_put:
	dma_buf_put(dmabuf);
	return ret;
}

/* ══════════════════════════════════════════════════════════════════════
 * DMA Frame Pool
 * ══════════════════════════════════════════════════════════════════════ */

int odl_tb5_frame_pool_alloc(struct odl_tb5_device *dev)
{
	struct odl_tb5_frame_pool *pool = &dev->frame_pool;
	struct device *dma_dev = tb_ring_dma_device(dev->tx.ring);
	int i;

	pool->size = ODL_TB5_FRAME_POOL_SIZE;
	pool->free_count = pool->size;
	spin_lock_init(&pool->lock);
	init_waitqueue_head(&pool->avail_waitq);

	pool->slots = kvzalloc(pool->size * sizeof(*pool->slots), GFP_KERNEL);
	if (!pool->slots)
		return -ENOMEM;

	pool->bitmap = bitmap_zalloc(pool->size, GFP_KERNEL);
	if (!pool->bitmap) {
		kvfree(pool->slots);
		pool->slots = NULL;
		return -ENOMEM;
	}

	for (i = 0; i < pool->size; i++) {
		struct odl_tb5_frame_slot *slot = &pool->slots[i];

		slot->virt = dma_alloc_coherent(dma_dev, ODL_TB5_FRAME_SIZE,
						&slot->phys, GFP_KERNEL);
		if (!slot->virt) {
			pr_err("odl_tb5: frame pool alloc failed at slot %d\n", i);
			goto err_free;
		}
		slot->slot_idx = i;
		slot->in_use = false;
		slot->frame.buffer_phy = slot->phys;
	}

	pr_info("odl_tb5: frame pool allocated: %d x %d bytes (%d KB)\n",
		pool->size, ODL_TB5_FRAME_SIZE,
		(pool->size * ODL_TB5_FRAME_SIZE) >> 10);
	return 0;

err_free:
	while (--i >= 0) {
		dma_free_coherent(dma_dev, ODL_TB5_FRAME_SIZE,
				  pool->slots[i].virt, pool->slots[i].phys);
	}
	bitmap_free(pool->bitmap);
	pool->bitmap = NULL;
	kvfree(pool->slots);
	pool->slots = NULL;
	return -ENOMEM;
}

void odl_tb5_frame_pool_free(struct odl_tb5_device *dev)
{
	struct odl_tb5_frame_pool *pool = &dev->frame_pool;
	struct device *dma_dev;
	int i;

	if (!pool->slots)
		return;

	dma_dev = tb_ring_dma_device(dev->tx.ring);

	for (i = 0; i < pool->size; i++) {
		if (pool->slots[i].virt)
			dma_free_coherent(dma_dev, ODL_TB5_FRAME_SIZE,
					  pool->slots[i].virt,
					  pool->slots[i].phys);
	}

	bitmap_free(pool->bitmap);
	pool->bitmap = NULL;
	kvfree(pool->slots);
	pool->slots = NULL;
}

struct odl_tb5_frame_slot *odl_tb5_frame_pool_get(struct odl_tb5_frame_pool *pool)
{
	struct odl_tb5_frame_slot *slot;
	unsigned long flags;
	int idx;

	spin_lock_irqsave(&pool->lock, flags);

	idx = find_first_zero_bit(pool->bitmap, pool->size);
	if (idx >= pool->size) {
		spin_unlock_irqrestore(&pool->lock, flags);
		return NULL;
	}

	set_bit(idx, pool->bitmap);
	slot = &pool->slots[idx];
	slot->in_use = true;
	slot->tx_msg = NULL;
	pool->free_count--;

	spin_unlock_irqrestore(&pool->lock, flags);
	return slot;
}

void odl_tb5_frame_pool_put(struct odl_tb5_frame_pool *pool,
			    struct odl_tb5_frame_slot *slot)
{
	unsigned long flags;

	spin_lock_irqsave(&pool->lock, flags);

	clear_bit(slot->slot_idx, pool->bitmap);
	slot->in_use = false;
	slot->tx_msg = NULL;
	pool->free_count++;

	spin_unlock_irqrestore(&pool->lock, flags);
	wake_up_interruptible(&pool->avail_waitq);
}

/* Batch allocation: get up to @requested slots with a single spinlock. */
int odl_tb5_frame_pool_get_batch(struct odl_tb5_frame_pool *pool,
				 struct odl_tb5_frame_slot **slots,
				 int requested)
{
	unsigned long flags;
	int allocated = 0;
	int idx = 0;

	spin_lock_irqsave(&pool->lock, flags);

	while (allocated < requested &&
	       pool->free_count > ODL_TB5_TX_POOL_RESERVE) {
		idx = find_next_zero_bit(pool->bitmap, pool->size, idx);
		if (idx >= pool->size)
			break;

		set_bit(idx, pool->bitmap);
		pool->free_count--;
		pool->slots[idx].in_use = true;
		pool->slots[idx].tx_msg = NULL;
		slots[allocated++] = &pool->slots[idx];
		idx++;
	}

	spin_unlock_irqrestore(&pool->lock, flags);
	return allocated;
}

/* ══════════════════════════════════════════════════════════════════════
 * SG Batch Buffer Pool (throughput mode)
 * ══════════════════════════════════════════════════════════════════════ */

int odl_tb5_batch_pool_alloc(struct odl_tb5_device *dev)
{
	struct odl_tb5_batch_pool *pool = &dev->batch_pool;
	struct device *dma_dev = tb_ring_dma_device(dev->tx.ring);
	int i;

	INIT_LIST_HEAD(&pool->free_list);
	spin_lock_init(&pool->lock);
	init_waitqueue_head(&pool->avail_waitq);
	pool->free_count = 0;

	for (i = 0; i < ODL_TB5_BATCH_BUF_COUNT; i++) {
		struct odl_tb5_batch_buf *buf = &pool->bufs[i];

		buf->virt = dma_alloc_coherent(dma_dev,
					       ODL_TB5_BATCH_BUF_SIZE,
					       &buf->phys, GFP_KERNEL);
		if (!buf->virt) {
			pr_err("odl_tb5: batch buf alloc failed at %d\n", i);
			goto err_free;
		}

		buf->in_use = false;
		buf->tx_msg = NULL;
		atomic_set(&buf->frames_pending, 0);
		INIT_LIST_HEAD(&buf->list);
		list_add_tail(&buf->list, &pool->free_list);
		pool->free_count++;
	}

	pr_info("odl_tb5: batch pool allocated: %d x %d KB (%d MB total)\n",
		ODL_TB5_BATCH_BUF_COUNT,
		ODL_TB5_BATCH_BUF_SIZE >> 10,
		(ODL_TB5_BATCH_BUF_COUNT * ODL_TB5_BATCH_BUF_SIZE) >> 20);
	return 0;

err_free:
	while (--i >= 0) {
		dma_free_coherent(dma_dev, ODL_TB5_BATCH_BUF_SIZE,
				  pool->bufs[i].virt, pool->bufs[i].phys);
		pool->bufs[i].virt = NULL;
	}
	INIT_LIST_HEAD(&pool->free_list);
	pool->free_count = 0;
	return -ENOMEM;
}

void odl_tb5_batch_pool_free(struct odl_tb5_device *dev)
{
	struct odl_tb5_batch_pool *pool = &dev->batch_pool;
	struct device *dma_dev;
	int i;

	if (!pool->bufs[0].virt)
		return;

	if (dev->tx.ring)
		dma_dev = tb_ring_dma_device(dev->tx.ring);
	else if (dev->rx.ring)
		dma_dev = tb_ring_dma_device(dev->rx.ring);
	else
		return;

	for (i = 0; i < ODL_TB5_BATCH_BUF_COUNT; i++) {
		if (pool->bufs[i].virt)
			dma_free_coherent(dma_dev, ODL_TB5_BATCH_BUF_SIZE,
					  pool->bufs[i].virt,
					  pool->bufs[i].phys);
		pool->bufs[i].virt = NULL;
	}

	INIT_LIST_HEAD(&pool->free_list);
	pool->free_count = 0;
}

struct odl_tb5_batch_buf *
odl_tb5_batch_pool_get(struct odl_tb5_batch_pool *pool)
{
	struct odl_tb5_batch_buf *buf;
	unsigned long flags;

	spin_lock_irqsave(&pool->lock, flags);

	if (list_empty(&pool->free_list)) {
		spin_unlock_irqrestore(&pool->lock, flags);
		return NULL;
	}

	buf = list_first_entry(&pool->free_list,
			       struct odl_tb5_batch_buf, list);
	list_del_init(&buf->list);
	buf->in_use = true;
	pool->free_count--;

	spin_unlock_irqrestore(&pool->lock, flags);
	return buf;
}

void odl_tb5_batch_pool_put(struct odl_tb5_batch_pool *pool,
			    struct odl_tb5_batch_buf *buf)
{
	unsigned long flags;

	spin_lock_irqsave(&pool->lock, flags);

	buf->in_use = false;
	buf->tx_msg = NULL;
	list_add_tail(&buf->list, &pool->free_list);
	pool->free_count++;

	spin_unlock_irqrestore(&pool->lock, flags);
	wake_up_interruptible(&pool->avail_waitq);
}

/* ══════════════════════════════════════════════════════════════════════
 * Stream Lifecycle
 * ══════════════════════════════════════════════════════════════════════ */

static void odl_tb5_stream_free(struct kref *ref)
{
	struct odl_tb5_stream *stream =
		container_of(ref, struct odl_tb5_stream, refcount);
	struct odl_tb5_tx_msg *tx, *tx_tmp;
	struct odl_tb5_rx_msg *rx, *rx_tmp;

	list_for_each_entry_safe(tx, tx_tmp, &stream->tx_queue, list) {
		list_del(&tx->list);
		kvfree(tx->data);
		kfree(tx);
	}

	list_for_each_entry_safe(rx, rx_tmp, &stream->rx_queue, list) {
		list_del(&rx->list);
		kfree(rx->data);
		kfree(rx);
	}

	kfree(stream->rx_asm_buf);
	kfree(stream);
}

struct odl_tb5_stream *odl_tb5_stream_create(struct odl_tb5_device *dev,
					     struct odl_tb5_file_ctx *owner,
					     u8 filter_id)
{
	struct odl_tb5_stream *stream;
	int id;

	stream = kzalloc(sizeof(*stream), GFP_KERNEL);
	if (!stream)
		return ERR_PTR(-ENOMEM);

	if (filter_id == 0) {
		id = ida_alloc_range(&dev->stream_ida, 20, 255, GFP_KERNEL);
		if (id < 0) {
			kfree(stream);
			return ERR_PTR(id);
		}
		stream->id = (u8)id;
	} else {
		if (filter_id < 1) {
			kfree(stream);
			return ERR_PTR(-EINVAL);
		}
		id = ida_alloc_range(&dev->stream_ida,
				     filter_id, filter_id, GFP_KERNEL);
		if (id < 0) {
			kfree(stream);
			return ERR_PTR(id);
		}
		stream->id = filter_id;
	}

	stream->dev = dev;
	stream->owner = owner;
	INIT_LIST_HEAD(&stream->tx_queue);
	spin_lock_init(&stream->tx_lock);
	stream->tx_queue_len = 0;
	stream->tx_queue_max = 64;
	atomic_set(&stream->tx_completed, 0);
	atomic_set(&stream->tx_in_flight, 0);
	init_waitqueue_head(&stream->tx_waitq);

	INIT_LIST_HEAD(&stream->rx_queue);
	spin_lock_init(&stream->rx_lock);
	stream->rx_queue_len = 0;
	stream->rx_queue_max = 65536;   /* was 256: a single ~1MB msg = ~264 frames;
					 * under load the sender runs >256 frames ahead
					 * and the old cap silently DROPPED frames ->
					 * plugin framing desync -> vLLM hang. */
	atomic_set(&stream->rx_complete, 0);
	init_waitqueue_head(&stream->rx_waitq);

	kref_init(&stream->refcount);

	mutex_lock(&dev->stream_lock);
	hash_add_rcu(dev->streams, &stream->node, stream->id);
	mutex_unlock(&dev->stream_lock);

	if (owner) {
		spin_lock(&owner->lock);
		list_add(&stream->owner_list, &owner->streams);
		spin_unlock(&owner->lock);
	}

	/* Start RX repost on first stream open */
	if (dev->rx_target == 0 && dev->frame_pool.slots) {
		dev->rx_target = dev->frame_pool.size / 2;
		odl_tb5_rx_repost(dev);
		pr_info("odl_tb5: RX repost started (target=%d)\n",
			dev->rx_target);
	}

	pr_info("odl_tb5: stream %u created (owner=%px)\n",
		stream->id, owner);
	return stream;
}

void odl_tb5_stream_destroy(struct odl_tb5_stream *stream)
{
	struct odl_tb5_device *dev = stream->dev;

	pr_info("odl_tb5: stream %u destroying\n", stream->id);

	mutex_lock(&dev->stream_lock);

	hash_del_rcu(&stream->node);
	mutex_unlock(&dev->stream_lock);

	if (stream->owner) {
		spin_lock(&stream->owner->lock);
		list_del(&stream->owner_list);
		spin_unlock(&stream->owner->lock);
	}

	ida_free(&dev->stream_ida, stream->id);
	kref_put(&stream->refcount, odl_tb5_stream_free);
}

void odl_tb5_stream_put(struct odl_tb5_stream *stream)
{
	kref_put(&stream->refcount, odl_tb5_stream_free);
}

void odl_tb5_streams_destroy_all(struct odl_tb5_device *dev)
{
	struct odl_tb5_stream *stream;
	struct hlist_node *tmp;
	int bkt;

	mutex_lock(&dev->stream_lock);
	hash_for_each_safe(dev->streams, bkt, tmp, stream, node) {

		hash_del_rcu(&stream->node);

		if (stream->owner) {
			spin_lock(&stream->owner->lock);
			list_del(&stream->owner_list);
			spin_unlock(&stream->owner->lock);
		}
		ida_free(&dev->stream_ida, stream->id);
		kref_put(&stream->refcount, odl_tb5_stream_free);
	}
	mutex_unlock(&dev->stream_lock);
}

struct odl_tb5_stream *odl_tb5_stream_lookup(struct odl_tb5_device *dev,
					     u8 stream_id)
{
	struct odl_tb5_stream *stream;

	rcu_read_lock();
	hash_for_each_possible_rcu(dev->streams, stream, node, stream_id) {
		if (stream->id == stream_id) {
			/* Stream may be concurrently freed (hash_del_rcu in
			 * stream_destroy); only take a ref if still alive. */
			if (!kref_get_unless_zero(&stream->refcount))
				break;
			rcu_read_unlock();
			return stream;
		}
	}
	rcu_read_unlock();
	return NULL;
}

/* ══════════════════════════════════════════════════════════════════════
 * Stream TX Path — Adaptive Latency / Throughput Mode
 * ══════════════════════════════════════════════════════════════════════ */

/*
 * Evaluate whether to use latency or throughput TX mode.
 *
 * Throughput mode uses pre-allocated 256KB batch buffers for reduced
 * per-frame overhead (fewer pool locks, larger copy_from_user calls).
 * Latency mode uses per-frame pool slots for minimal single-frame delay.
 */
static enum odl_tb5_tx_mode
odl_tb5_evaluate_tx_mode(struct odl_tb5_device *dev, size_t msg_len)
{
	unsigned int pool_used;
	unsigned int nframes;

	/* Batch pool not available — latency only */
	if (!dev->batch_pool.bufs[0].virt)
		return ODL_TB5_TX_LATENCY;

	/* Large messages always use throughput mode */
	if (msg_len > ODL_TB5_THROUGHPUT_THRESH) {
		dev->tx_adaptive.consecutive_low = 0;
		dev->tx_adaptive.mode = ODL_TB5_TX_THROUGHPUT;
		return ODL_TB5_TX_THROUGHPUT;
	}

	pool_used = dev->frame_pool.size - dev->frame_pool.free_count;
	nframes = DIV_ROUND_UP(msg_len, ODL_TB5_STREAM_PAYLOAD_MAX);

	if (dev->tx_adaptive.mode == ODL_TB5_TX_LATENCY) {
		/* Transition up: load + new msg exceeds high watermark */
		if (pool_used + nframes > dev->tx_adaptive.high_watermark) {
			dev->tx_adaptive.consecutive_low = 0;
			dev->tx_adaptive.mode = ODL_TB5_TX_THROUGHPUT;
			pr_debug("odl_tb5: TX mode → THROUGHPUT "
				 "(pool_used=%u nframes=%u)\n",
				 pool_used, nframes);
		}
	} else {
		/* Transition down: load below low watermark for N sends */
		if (pool_used < dev->tx_adaptive.low_watermark) {
			if (++dev->tx_adaptive.consecutive_low >=
			    ODL_TB5_MODE_HYSTERESIS) {
				dev->tx_adaptive.mode = ODL_TB5_TX_LATENCY;
				dev->tx_adaptive.consecutive_low = 0;
				pr_debug("odl_tb5: TX mode → LATENCY "
					 "(pool_used=%u)\n", pool_used);
			}
		} else {
			dev->tx_adaptive.consecutive_low = 0;
		}
	}

	return dev->tx_adaptive.mode;
}

/*
 * Latency-mode send — per-frame pool slots with backpressure.
 *
 * Each frame-sized chunk is copied directly from userspace into a DMA
 * pool slot, submitted to the NHI TX ring, and the slot is recycled by
 * the TX callback.  If the pool runs low, we block until a TX callback
 * frees a slot — this provides natural flow control.
 */
static int odl_tb5_stream_send_latency(struct odl_tb5_stream *stream,
					u8 dst_id,
					const void __user *data,
					size_t len)
{
	struct odl_tb5_device *dev = stream->dev;
	struct odl_tb5_frame_pool *pool = &dev->frame_pool;
	struct odl_tb5_tx_msg *msg;
	struct odl_tb5_frame_slot *slot;
	struct odl_tb5_stream_hdr *hdr;
	long ret;

	msg = kzalloc(sizeof(*msg), GFP_KERNEL);
	if (!msg)
		return -ENOMEM;

	msg->data = NULL;
	msg->dst_id = dst_id;
	msg->len = len;
	msg->sent = 0;
	atomic_set(&msg->frames_pending, 0);
	msg->done = false;
	msg->stream = stream;
	INIT_LIST_HEAD(&msg->list);

	atomic_inc(&stream->tx_in_flight);

	while (msg->sent < len) {
		size_t remain = len - msg->sent;
		size_t payload = min_t(size_t, remain,
				       ODL_TB5_STREAM_PAYLOAD_MAX);
		bool first = (msg->sent == 0);
		bool last  = (msg->sent + payload == len);

		ret = wait_event_interruptible_timeout(pool->avail_waitq,
			pool->free_count > ODL_TB5_TX_POOL_RESERVE,
			msecs_to_jiffies(5000));
		if (ret <= 0) {
			if (ret == 0)
				ret = -ETIMEDOUT;
			goto wait_pending;
		}

		slot = odl_tb5_frame_pool_get(pool);
		if (!slot)
			continue;

		hdr = slot->virt;
		hdr->src_id = stream->id;
		hdr->dst_id = dst_id;
		if (first && last)
			hdr->flags = ODL_TB5_SHDR_F_SINGLE;
		else if (first)
			hdr->flags = ODL_TB5_SHDR_F_MSG_START;
		else if (last)
			hdr->flags = ODL_TB5_SHDR_F_MSG_END;
		else
			hdr->flags = 0;
		hdr->payload_len = cpu_to_le16(payload);

		if (copy_from_user(slot->virt + ODL_TB5_STREAM_HDR_SIZE,
				   data + msg->sent, payload)) {
			odl_tb5_frame_pool_put(pool, slot);
			ret = -EFAULT;
			goto wait_pending;
		}

		slot->frame.size = ODL_TB5_STREAM_HDR_SIZE + payload;
		slot->frame.sof  = ODL_TB5_PDF_SOF_DATA;
		slot->frame.eof  = ODL_TB5_PDF_EOF_DATA;
		slot->frame.callback = odl_tb5_tx_callback;
		slot->tx_msg = msg;

		atomic_inc(&msg->frames_pending);
		msg->sent += payload;

		if (tb_ring_tx(dev->tx.ring, &slot->frame) < 0) {
			atomic_dec(&msg->frames_pending);
			msg->sent -= payload;
			odl_tb5_frame_pool_put(pool, slot);
			ret = -EIO;
			goto wait_pending;
		}
	}

	return 0;

wait_pending:
	if (atomic_read(&msg->frames_pending) > 0) {
		wait_event_interruptible_timeout(stream->tx_waitq,
			atomic_read(&msg->frames_pending) == 0,
			msecs_to_jiffies(1000));
	}
	if (atomic_read(&msg->frames_pending) == 0) {
		atomic_dec(&stream->tx_in_flight);
		wake_up_interruptible(&stream->tx_waitq);
		kfree(msg);
	}
	return (int)ret;
}

/*
 * Throughput-mode send — uses pre-allocated 256KB contiguous DMA batch
 * buffers to reduce per-frame overhead.  Each batch holds up to 64
 * frames (64 × 4091 = 261,824 bytes of payload).  Benefits:
 *   - 1 batch buffer allocation vs. 64 pool slot allocations
 *   - Copy data in larger chunks (vs. 4KB per frame)
 *   - Frames are pre-staged before submission
 */
static int odl_tb5_stream_send_throughput(struct odl_tb5_stream *stream,
					   u8 dst_id,
					   const void __user *data,
					   size_t len)
{
	struct odl_tb5_device *dev = stream->dev;
	struct odl_tb5_batch_pool *bpool = &dev->batch_pool;
	struct odl_tb5_tx_msg *msg;
	size_t total_sent = 0;
	long ret;

	msg = kzalloc(sizeof(*msg), GFP_KERNEL);
	if (!msg)
		return -ENOMEM;

	msg->data = NULL;
	msg->dst_id = dst_id;
	msg->len = len;
	msg->sent = 0;
	atomic_set(&msg->frames_pending, 0);
	msg->done = false;
	msg->stream = stream;
	INIT_LIST_HEAD(&msg->list);

	atomic_inc(&stream->tx_in_flight);

	while (total_sent < len) {
		struct odl_tb5_batch_buf *batch;
		size_t batch_payload_cap;
		size_t batch_payload;
		int nframes, i;

		/* Wait for a free batch buffer */
		ret = wait_event_interruptible_timeout(bpool->avail_waitq,
			bpool->free_count > 0,
			msecs_to_jiffies(5000));
		if (ret <= 0) {
			if (ret == 0)
				ret = -ETIMEDOUT;
			goto wait_pending;
		}

		batch = odl_tb5_batch_pool_get(bpool);
		if (!batch)
			continue; /* spurious wakeup */

		/* How much user data fits in one batch buffer? */
		batch_payload_cap = (size_t)ODL_TB5_BATCH_FRAMES *
				    ODL_TB5_STREAM_PAYLOAD_MAX;
		batch_payload = min_t(size_t, len - total_sent,
				      batch_payload_cap);
		nframes = DIV_ROUND_UP(batch_payload,
				       ODL_TB5_STREAM_PAYLOAD_MAX);

		/* Fill each frame within the batch buffer */
		for (i = 0; i < nframes; i++) {
			void *frame_base = batch->virt +
					   ((size_t)i * ODL_TB5_FRAME_SIZE);
			struct odl_tb5_stream_hdr *hdr = frame_base;
			size_t offset = (size_t)i * ODL_TB5_STREAM_PAYLOAD_MAX;
			size_t payload = min_t(size_t,
					       batch_payload - offset,
					       ODL_TB5_STREAM_PAYLOAD_MAX);
			bool first = (total_sent == 0 && i == 0);
			bool last  = (total_sent + offset + payload == len);

			/* Stream header */
			hdr->src_id = stream->id;
			hdr->dst_id = dst_id;
			if (first && last)
				hdr->flags = ODL_TB5_SHDR_F_SINGLE;
			else if (first)
				hdr->flags = ODL_TB5_SHDR_F_MSG_START;
			else if (last)
				hdr->flags = ODL_TB5_SHDR_F_MSG_END;
			else
				hdr->flags = 0;
			hdr->payload_len = cpu_to_le16(payload);

			/* Copy payload from userspace */
			if (copy_from_user(frame_base +
					   ODL_TB5_STREAM_HDR_SIZE,
					   data + total_sent + offset,
					   payload)) {
				odl_tb5_batch_pool_put(bpool, batch);
				ret = -EFAULT;
				goto wait_pending;
			}

			/* Set up ring descriptor */
			batch->frames[i].buffer_phy = batch->phys +
				((size_t)i * ODL_TB5_FRAME_SIZE);
			batch->frames[i].size = ODL_TB5_STREAM_HDR_SIZE +
						payload;
			batch->frames[i].sof = ODL_TB5_PDF_SOF_DATA;
			batch->frames[i].eof = ODL_TB5_PDF_EOF_DATA;
			batch->frames[i].callback =
				odl_tb5_tx_batch_callback;
		}

		/* Arm completion tracking before submission */
		batch->tx_msg = msg;
		batch->total_frames = nframes;
		atomic_set(&batch->frames_pending, nframes);
		atomic_add(nframes, &msg->frames_pending);
		msg->sent += batch_payload;

		/* Submit all frames in this batch to the ring */
		for (i = 0; i < nframes; i++) {
			if (tb_ring_tx(dev->tx.ring,
				       &batch->frames[i]) < 0) {
				int unsub = nframes - i;

				atomic_sub(unsub, &batch->frames_pending);
				atomic_sub(unsub, &msg->frames_pending);
				msg->sent -= (size_t)unsub *
					     ODL_TB5_STREAM_PAYLOAD_MAX;
				batch->total_frames = i;
				if (i == 0)
					odl_tb5_batch_pool_put(bpool,
							       batch);
				ret = -EIO;
				goto wait_pending;
			}
		}

		total_sent += batch_payload;
	}

	return 0;

wait_pending:
	if (atomic_read(&msg->frames_pending) > 0) {
		wait_event_interruptible_timeout(stream->tx_waitq,
			atomic_read(&msg->frames_pending) == 0,
			msecs_to_jiffies(1000));
	}
	if (atomic_read(&msg->frames_pending) == 0) {
		atomic_dec(&stream->tx_in_flight);
		wake_up_interruptible(&stream->tx_waitq);
		kfree(msg);
	}
	return (int)ret;
}

/*
 * Non-blocking availability checks (for poll/epoll + async ioctl).
 * Returns true if a send/recv would not block.
 */
bool odl_tb5_stream_can_send(struct odl_tb5_stream *stream)
{
	struct odl_tb5_device *dev = stream->dev;
	struct odl_tb5_frame_pool *pool = &dev->frame_pool;

	/* Can send if we have frames available and state is ready */
	return dev->state == ODL_TB5_STATE_READY &&
	       pool && pool->free_count > ODL_TB5_TX_POOL_RESERVE;
}

bool odl_tb5_stream_can_recv(struct odl_tb5_stream *stream)
{
	/* Can recv if data is already in the RX queue */
	return atomic_read(&stream->rx_complete) > 0;
}

/*
 * Stream send — adaptive dispatcher.
 *
 * Evaluates TX mode (latency vs throughput) based on message size and
 * current load, then dispatches to the appropriate send path.
 */
int odl_tb5_stream_send(struct odl_tb5_stream *stream,
			u8 dst_id, const void __user *data, size_t len)
{
	struct odl_tb5_device *dev = stream->dev;
	enum odl_tb5_tx_mode mode;

	if (dev->state != ODL_TB5_STATE_READY)
		return -ENOTCONN;

	if (len == 0 || len > (size_t)ODL_TB5_STREAM_PAYLOAD_MAX * 4096)
		return -EINVAL;

	mode = odl_tb5_evaluate_tx_mode(dev, len);

	if (mode == ODL_TB5_TX_THROUGHPUT)
		return odl_tb5_stream_send_throughput(stream, dst_id,
						      data, len);
	return odl_tb5_stream_send_latency(stream, dst_id, data, len);
}

void odl_tb5_tx_drain_work_fn(struct work_struct *work)
{
	struct odl_tb5_device *dev =
		container_of(work, struct odl_tb5_device, tx_drain_work);
	struct odl_tb5_stream *stream;
	struct odl_tb5_tx_msg *msg;
	struct odl_tb5_frame_slot *slot;
	struct odl_tb5_stream_hdr *hdr;
	unsigned long flags;
	int bkt;
	bool did_work;

	do {
		did_work = false;

		rcu_read_lock();
		hash_for_each(dev->streams, bkt, stream, node) {
			spin_lock_irqsave(&stream->tx_lock, flags);
			msg = list_first_entry_or_null(&stream->tx_queue,
						       struct odl_tb5_tx_msg,
						       list);
			if (!msg || msg->sent >= msg->len) {
				spin_unlock_irqrestore(&stream->tx_lock, flags);
				continue;
			}
			spin_unlock_irqrestore(&stream->tx_lock, flags);

			slot = odl_tb5_frame_pool_get(&dev->frame_pool);
			if (!slot)
				goto out;

			/* Build frame: 5-byte stream header + payload */
			hdr = slot->virt;
			hdr->src_id = stream->id;
			hdr->dst_id = msg->dst_id;

			{
				size_t remain = msg->len - msg->sent;
				size_t payload = min_t(size_t, remain,
						       ODL_TB5_STREAM_PAYLOAD_MAX);
				bool first = (msg->sent == 0);
				bool last = (msg->sent + payload == msg->len);

				if (first && last)
					hdr->flags = ODL_TB5_SHDR_F_SINGLE;
				else if (first)
					hdr->flags = ODL_TB5_SHDR_F_MSG_START;
				else if (last)
					hdr->flags = ODL_TB5_SHDR_F_MSG_END;
				else
					hdr->flags = 0;

				hdr->payload_len = cpu_to_le16(payload);
				memcpy(slot->virt + ODL_TB5_STREAM_HDR_SIZE,
				       msg->data + msg->sent, payload);

				slot->frame.size = ODL_TB5_STREAM_HDR_SIZE + payload;
				slot->frame.sof = ODL_TB5_PDF_SOF_DATA;
				slot->frame.eof = ODL_TB5_PDF_EOF_DATA;
				slot->frame.callback = odl_tb5_tx_callback;
				slot->tx_msg = msg;

				msg->sent += payload;
				atomic_inc(&msg->frames_pending);
			}

			if (tb_ring_tx(dev->tx.ring, &slot->frame) < 0) {
				pr_warn("odl_tb5: tb_ring_tx failed for "
					"stream %u (sent=%zu/%zu)\n",
					stream->id, msg->sent, msg->len);
				msg->sent -= le16_to_cpu(hdr->payload_len);
				atomic_dec(&msg->frames_pending);
				odl_tb5_frame_pool_put(&dev->frame_pool, slot);
				/* Wake waiter so timeout can fire */
				wake_up_interruptible(&stream->tx_waitq);
				goto out;
			}

			did_work = true;
		}
out:
		rcu_read_unlock();
	} while (did_work);
}

int odl_tb5_stream_wait_tx(struct odl_tb5_stream *stream, u32 timeout_ms)
{
	long ret;

	if (timeout_ms == 0) {
		ret = wait_event_interruptible(stream->tx_waitq,
			atomic_read(&stream->tx_in_flight) == 0);
	} else {
		ret = wait_event_interruptible_timeout(stream->tx_waitq,
			atomic_read(&stream->tx_in_flight) == 0,
			msecs_to_jiffies(timeout_ms));
		if (ret == 0)
			return -ETIMEDOUT;
	}

	return ret < 0 ? -EINTR : 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * Stream RX Path
 * ══════════════════════════════════════════════════════════════════════ */

/* Optional bounded busy-poll for an RX completion before sleeping.  The RX
 * softirq (odl_tb5_rx_callback) increments rx_complete on another CPU, so a
 * spinning reader sees it within cache-coherency latency and skips the
 * context-switch wake (~10-15 us on this box).  Bounded + falls back to
 * wait_event, so it never hangs.  Off unless odl_busy_poll_us > 0. */
static inline void odl_tb5_rx_busy_poll(struct odl_tb5_stream *stream)
{
	ktime_t deadline;

	if (!odl_busy_poll_us || atomic_read(&stream->rx_complete) > 0)
		return;
	deadline = ktime_add_ns(ktime_get(), (u64)odl_busy_poll_us * 1000);
	while (atomic_read(&stream->rx_complete) == 0) {
		if (ktime_after(ktime_get(), deadline))
			break;
		cpu_relax();
	}
}

int odl_tb5_stream_recv(struct odl_tb5_stream *stream,
			void __user *buf, size_t buf_len,
			u8 *src_id, u32 *actual_len)
{
	struct odl_tb5_rx_msg *msg;
	unsigned long flags;
	int ret = 0;

	/* Wait for a complete assembled message */
	odl_tb5_rx_busy_poll(stream);
	ret = wait_event_interruptible(stream->rx_waitq,
		atomic_read(&stream->rx_complete) > 0);
	if (ret)
		return -EINTR;

	/* Dequeue one complete message */
	spin_lock_irqsave(&stream->rx_lock, flags);
	msg = list_first_entry_or_null(&stream->rx_queue,
				       struct odl_tb5_rx_msg, list);
	if (msg) {
		list_del(&msg->list);
		stream->rx_queue_len--;
	}
	spin_unlock_irqrestore(&stream->rx_lock, flags);

	if (!msg)
		return -EAGAIN;

	atomic_dec(&stream->rx_complete);

	*src_id = msg->src_id;
	*actual_len = min_t(size_t, msg->len, buf_len);
	if (copy_to_user(buf, msg->data, *actual_len))
		ret = -EFAULT;

	kfree(msg->data);
	kfree(msg);
	return ret;
}

int odl_tb5_stream_wait_rx(struct odl_tb5_stream *stream, u32 timeout_ms)
{
	long ret;

	odl_tb5_rx_busy_poll(stream);
	if (timeout_ms == 0) {
		ret = wait_event_interruptible(stream->rx_waitq,
			atomic_read(&stream->rx_complete) > 0);
	} else {
		ret = wait_event_interruptible_timeout(stream->rx_waitq,
			atomic_read(&stream->rx_complete) > 0,
			msecs_to_jiffies(timeout_ms));
		if (ret == 0)
			return -ETIMEDOUT;
	}

	return ret < 0 ? -EINTR : 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * RX Repost — keep RX ring filled with pool frames
 * ══════════════════════════════════════════════════════════════════════ */

void odl_tb5_rx_repost(struct odl_tb5_device *dev)
{
	int posted = atomic_read(&dev->rx_posted);
	int target = dev->rx_target;

	while (posted < target) {
		struct odl_tb5_frame_slot *slot;

		slot = odl_tb5_frame_pool_get(&dev->frame_pool);
		if (!slot)
			break;

		slot->frame.buffer_phy = slot->phys;
		slot->frame.size = 0; /* NHI fills this on RX completion */
		slot->frame.callback = odl_tb5_rx_callback;
		slot->frame.sof = ODL_TB5_PDF_SOF_DATA;
		slot->frame.eof = ODL_TB5_PDF_EOF_DATA;

		if (tb_ring_rx(dev->rx.ring, &slot->frame) < 0) {
			odl_tb5_frame_pool_put(&dev->frame_pool, slot);
			break;
		}

		atomic_inc(&dev->rx_posted);
		posted++;
	}
}
