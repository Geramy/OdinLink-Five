#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/dma-buf.h>
#include <linux/thunderbolt.h>
#include <linux/slab.h>
#include <linux/wait.h>
#include <linux/kthread.h>
#include <linux/list.h>
#include <linux/types.h>
#include <linux/stddef.h>
#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/pci.h>
#include <linux/uaccess.h>
#include <linux/atomic.h>
#include <linux/wait.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/scatterlist.h>
#include <linux/dma-mapping.h>

#define DEVICE_NAME "tb5_ol_ring"

#define TB5_RING_IOCTL_ENQUEUE_SEND _IOW('T', 1, struct tb5_ring_request)
#define TB5_RING_IOCTL_ENQUEUE_RECV _IOW('T', 2, struct tb5_ring_request)
#define TB5_RING_IOCTL_TEST_COMPLETION _IOR('T', 3, int)

struct tb5_ring_request {
    int dmabuf_fd;
    loff_t offset;
    size_t len;
};

struct tb5_ring_device {
    struct cdev cdev;
    dev_t devt;
    struct class *class;
    struct device *device;
    wait_queue_head_t waitq;
    atomic_t completion_count;
    int index;
    struct list_head list;
    // Mock Thunderbolt state
    bool tb_connected;
};

static struct class *tb5_class;
static dev_t tb5_devt;
static LIST_HEAD(tb5_devices_list);
static int tb5_device_count = 0;

static long tb5_ring_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct tb5_ring_device *dev = file->private_data;
    struct tb5_ring_request req;
    struct dma_buf *dmabuf;
    struct dma_buf_attachment *attach;
    struct sg_table *sgt;
    int ret = 0;

    pr_debug("TB5: ioctl called with cmd=0x%x, arg=%lx\n", cmd, arg);

    switch (cmd) {
    case TB5_RING_IOCTL_ENQUEUE_SEND:
    case TB5_RING_IOCTL_ENQUEUE_RECV:
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;

        // Validate DMA buffer file descriptor
        if (req.dmabuf_fd < 0) {
            pr_err("TB5: Invalid DMA buffer file descriptor: %d\n", req.dmabuf_fd);
            return -EINVAL;
        }

        dmabuf = dma_buf_get(req.dmabuf_fd);
        if (IS_ERR(dmabuf)) {
            pr_err("TB5: Failed to get DMA buffer: %ld\n", PTR_ERR(dmabuf));
            return PTR_ERR(dmabuf);
        }

        // Validate buffer size and offset
        if (req.offset < 0 || req.len == 0 || req.offset + req.len > dmabuf->size) {
            pr_err("TB5: Invalid buffer parameters: offset=%lld, len=%zu, buf_size=%zu\n",
                   req.offset, req.len, dmabuf->size);
            dma_buf_put(dmabuf);
            return -EINVAL;
        }

        attach = dma_buf_attach(dmabuf, dev->device);
        if (IS_ERR(attach)) {
            pr_err("TB5: Failed to attach DMA buffer: %ld\n", PTR_ERR(attach));
            dma_buf_put(dmabuf);
            return PTR_ERR(attach);
        }

        sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
        if (IS_ERR(sgt)) {
            pr_err("TB5: Failed to map DMA buffer: %ld\n", PTR_ERR(sgt));
            dma_buf_detach(dmabuf, attach);
            dma_buf_put(dmabuf);
            return PTR_ERR(sgt);
        }

        // Validate scatter-gather list
        if (!sgt->sgl || !sgt->nents) {
            pr_err("TB5: Invalid scatter-gather list\n");
            dma_buf_unmap_attachment(attach, sgt, DMA_BIDIRECTIONAL);
            dma_buf_detach(dmabuf, attach);
            dma_buf_put(dmabuf);
            return -EINVAL;
        }

        // Process the command
        if (cmd == TB5_RING_IOCTL_ENQUEUE_SEND) {
            // TODO: Implement actual Thunderbolt ring transmission
            pr_debug("TB5: Enqueueing send: fd=%d, offset=%lld, len=%zu\n",
                    req.dmabuf_fd, req.offset, req.len);

            // Check if Thunderbolt connection is available
            if (!dev->tb_connected) {
                pr_err("TB5: Thunderbolt connection not available\n");
                ret = -ENODEV;
            } else {
                // Here would be the actual tb_ring_tx implementation
                atomic_inc(&dev->completion_count);
            }
        } else { // ENQUEUE_RECV
            pr_debug("TB5: Enqueueing recv: fd=%d, offset=%lld, len=%zu\n",
                    req.dmabuf_fd, req.offset, req.len);

            if (!dev->tb_connected) {
                pr_err("TB5: Thunderbolt connection not available\n");
                ret = -ENODEV;
            } else {
                // Here would be the actual tb_ring_rx implementation
                atomic_inc(&dev->completion_count);
            }
        }

        dma_buf_unmap_attachment(attach, sgt, DMA_BIDIRECTIONAL);
        dma_buf_detach(dmabuf, attach);
        dma_buf_put(dmabuf);
        break;

    case TB5_RING_IOCTL_TEST_COMPLETION:
        // Return completion count
        ret = atomic_read(&dev->completion_count);
        if (copy_to_user((void __user *)arg, &ret, sizeof(int)))
            ret = -EFAULT;
        break;

    default:
        pr_err("TB5: Unknown ioctl command: 0x%x\n", cmd);
        return -EINVAL;
    }

    return ret;
}

static int tb5_ring_open(struct inode *inode, struct file *file)
{
    struct tb5_ring_device *dev;

    dev = container_of(inode->i_cdev, struct tb5_ring_device, cdev);
    file->private_data = dev;

    return 0;
}

static int tb5_ring_release(struct inode *inode, struct file *file)
{
    file->private_data = NULL;
    return 0;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = tb5_ring_open,
    .release = tb5_ring_release,
    .unlocked_ioctl = tb5_ring_ioctl,
};

static struct tb5_ring_device *tb5_ring_device_find_by_minor(int minor)
{
    struct tb5_ring_device *dev;

    list_for_each_entry(dev, &tb5_devices_list, list) {
        if (dev->index == minor)
            return dev;
    }
    return NULL;
}

static int tb5_ring_create_device(int index)
{
    struct tb5_ring_device *dev;
    int ret;

    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    // Initialize device state
    dev->index = index;
    dev->tb_connected = true; // Mock Thunderbolt connection for testing
    atomic_set(&dev->completion_count, 0);
    init_waitqueue_head(&dev->waitq);

    // Register character device
    dev->devt = MKDEV(MAJOR(tb5_devt), index);
    dev->device = device_create(tb5_class, NULL, dev->devt, NULL, DEVICE_NAME "%d", index);
    if (IS_ERR(dev->device)) {
        kfree(dev);
        return PTR_ERR(dev->device);
    }

    cdev_init(&dev->cdev, &fops);
    ret = cdev_add(&dev->cdev, dev->devt, 1);
    if (ret < 0) {
        device_destroy(tb5_class, dev->devt);
        kfree(dev);
        return ret;
    }

    list_add_tail(&dev->list, &tb5_devices_list);
    tb5_device_count++;

    pr_info("Thunderbolt 5 DMA ring device %d created\n", index);
    return 0;
}

static void tb5_ring_destroy_device(struct tb5_ring_device *dev)
{
    cdev_del(&dev->cdev);
    device_destroy(tb5_class, dev->devt);
    list_del(&dev->list);
    kfree(dev);
    tb5_device_count--;
}

static int __init tb5_ring_init(void)
{
    int ret;

    tb5_class = class_create(DEVICE_NAME);
    if (IS_ERR(tb5_class)) {
        return PTR_ERR(tb5_class);
    }

    ret = alloc_chrdev_region(&tb5_devt, 0, 256, DEVICE_NAME);
    if (ret < 0) {
        class_destroy(tb5_class);
        return ret;
    }

    // Create a single device for testing (index 0)
    ret = tb5_ring_create_device(0);
    if (ret < 0) {
        unregister_chrdev_region(tb5_devt, 256);
        class_destroy(tb5_class);
        return ret;
    }

    pr_info("Thunderbolt 5 DMA ring module loaded\n");
    return 0;
}

static void __exit tb5_ring_exit(void)
{
    struct tb5_ring_device *dev, *tmp;

    list_for_each_entry_safe(dev, tmp, &tb5_devices_list, list) {
        tb5_ring_destroy_device(dev);
    }

    unregister_chrdev_region(tb5_devt, 256);
    class_destroy(tb5_class);

    pr_info("Thunderbolt 5 DMA ring module unloaded\n");
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("TB5 NCCL Plugin Team");
MODULE_DESCRIPTION("Thunderbolt 5 DMA Ring Kernel Driver");
MODULE_INFO(import_ns, "DMA_BUF");

module_init(tb5_ring_init);
module_exit(tb5_ring_exit);
