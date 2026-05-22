/*
 * OdinLink Verbs Provider — Device Discovery & ibv_open_device Interposition
 *
 * Two operating modes:
 *
 * 1. Standalone shared library (primary):
 *    libodl_tb5_verbs.so provides ibv_open_device via symbol interposition.
 *    For OdinLink-Five devices, creates a fully functional ibv_context with
 *    all standard ibv_* operations. For non-ODL devices, chains to the real
 *    libibverbs.
 *
 *    Usage: LD_PRELOAD=libodl_tb5_verbs.so <any verbs app>
 *           or: gcc ... -lodl_tb5_verbs (resolves ibv_open_device at link time)
 *
 * 2. rdma-core provider plugin (requires private headers):
 *    libodl_tb5-rdmav34.so registers via PROVIDER_DRIVER for full
 *    rdma-core integration (ibv_devinfo, etc.). Build with
 *    -DHAVE_RDMA_CORE_DRIVER_H.
 *
 * Device discovery: scans /dev/odl_tb5_N entries.
 * Zero-copy GPU:   ibv_reg_dmabuf_mr passes dmabuf fd to kernel driver.
 * Async model:     ibv_post_send → workqueue → worker thread → completion CQ.
 */

#include "odl_tb5_verbs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <errno.h>

int odl_verbs_debug_level = -1;

/* ── Device Scan ────────────────────────────────────────────────────── */

#define ODL_MAX_DEVICES 16

static struct odl_verbs_device *odl_device_list[ODL_MAX_DEVICES];
static int odl_device_count = 0;
static pthread_once_t odl_scan_once = PTHREAD_ONCE_INIT;

static void odl_scan_devices(void)
{
    DIR *dir = opendir("/dev");
    if (!dir) return;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && odl_device_count < ODL_MAX_DEVICES) {
        if (strncmp(entry->d_name, "odl_tb5_", 8) != 0)
            continue;

        int idx = atoi(entry->d_name + 8);
        char path[64];
        snprintf(path, sizeof(path), "/dev/%s", entry->d_name);

        int fd = open(path, O_RDWR);
        if (fd < 0) continue;
        close(fd);

        struct odl_verbs_device *dev = calloc(1, sizeof(*dev));
        if (!dev) continue;

        dev->dev_index = idx;
        strncpy(dev->dev_path, path, sizeof(dev->dev_path) - 1);
        snprintf(dev->dev_name, sizeof(dev->dev_name), "odl_tb5_%d", idx);

        /* Set up the ibv_device fields */
        strncpy((char *)dev->base.name,     dev->dev_name, sizeof(dev->base.name) - 1);
        strncpy((char *)dev->base.dev_name, dev->dev_name, sizeof(dev->base.dev_name) - 1);
        strncpy((char *)dev->base.dev_path, dev->dev_path, sizeof(dev->base.dev_path) - 1);
        dev->base.node_type      = IBV_NODE_RNIC;
        dev->base.transport_type = IBV_TRANSPORT_IB;

        odl_device_list[odl_device_count++] = dev;
    }
    closedir(dir);
    odl_loginfo("scan complete: %d device(s) found", odl_device_count);
}

/* ── Wrapper API ────────────────────────────────────────────────────── */

struct ibv_device *odl_find_tb5_device(int dev_index)
{
    pthread_once(&odl_scan_once, odl_scan_devices);
    for (int i = 0; i < odl_device_count; i++) {
        if (odl_device_list[i]->dev_index == dev_index)
            return &odl_device_list[i]->base;
    }
    return NULL;
}

int odl_num_tb5_devices(void)
{
    pthread_once(&odl_scan_once, odl_scan_devices);
    return odl_device_count;
}

bool odl_is_tb5_device(struct ibv_device *dev)
{
    return dev && dev->name &&
           strncmp(dev->name, "odl_tb5_", 8) == 0;
}

int odl_tb5_device_index(struct ibv_device *dev)
{
    struct odl_verbs_device *odl_dev = odl_dev_from_ibv(dev);
    return odl_dev->dev_index;
}

void odl_tb5_verbs_set_debug(int level)
{
    odl_verbs_debug_level = level;
}

/* ── ibv_open_device Symbol Interposition ───────────────────────────── */

struct ibv_context *ibv_open_device(struct ibv_device *device)
{
    ODL_TRACE_ENTRY();

    if (!device) { errno = EINVAL; return NULL; }

    /* Handle OdinLink-Five devices directly */
    if (device->name && strncmp(device->name, "odl_tb5_", 8) == 0) {
        struct ibv_context *ctx = odl_ibv_open_device(device);
        ODL_TRACE_EXIT();
        return ctx;
    }

    /* Chain to real libibverbs for all other devices */
    static struct ibv_context *(*real_ibv_open_device)(struct ibv_device *, int);
    if (!real_ibv_open_device) {
        real_ibv_open_device = dlsym(RTLD_NEXT, "ibv_open_device");
        if (!real_ibv_open_device) {
            odl_logerr("dlsym(RTLD_NEXT, ibv_open_device) failed: %s",
                        dlerror());
            errno = ENOSYS;
            return NULL;
        }
    }

    odl_loginfo("forwarding %s to real libibverbs", device->name);
    struct ibv_context *ctx = real_ibv_open_device(device, -1);
    ODL_TRACE_EXIT();
    return ctx;
}
