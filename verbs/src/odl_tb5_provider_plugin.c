/*
 * OdinLink — Verbs: rdma-core Provider Plugin (Auto-Discovery)
 *
 * When you install libodl_tb5-rdmav34.so to the libibverbs provider
 * directory, rdma-core loads it automatically. From then on, every
 * rdma-core app (ibv_devinfo, ibv_rc_pingpong, NCCL's IB transport)
 * sees OdinLink as just another RDMA device alongside your Mellanox
 * or SoftiWARP cards. No LD_PRELOAD needed.
 *
 * This file manually replicates the provider struct layouts from
 * rdma-core's private driver.h to avoid depending on unreleased
 * header packages.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <odl_tb5/odl_tb5.h>

/* ── Private provider structs (matching rdma-core driver.h layouts) ─── */

struct verbs_device {
    struct ibv_device          device;    /* must be first */
    const void                *ops;       /* const struct verbs_device_ops * */
    int                        refcount;
    void                      *entry;     /* list_node */
    void                      *sysfs;     /* struct verbs_sysfs_dev * */
    uint64_t                   core_support;
};

struct verbs_device_ops {
    const char                *name;
    uint32_t                   match_min_abi_version;
    uint32_t                   match_max_abi_version;
    const void                *match_table;
    const struct verbs_device_ops **static_providers;
    int                      (*match_device)(void *sysfs_dev);
    struct verbs_context     *(*alloc_context)(struct ibv_device *, int, void *);
    struct verbs_context     *(*import_context)(struct ibv_device *, int);
    struct verbs_device      *(*alloc_device)(void *sysfs_dev);
    void                     (*uninit_device)(struct verbs_device *);
};

/* ── Private libibverbs symbols (resolved at runtime) ───────────────── */

typedef void (*register_driver_fn_t)(const struct verbs_device_ops *);

static register_driver_fn_t verbs_register_driver = NULL;

static int resolve_symbols(void)
{
    static int done = 0;
    if (done) return 0;

    void *h = dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return -1;
    verbs_register_driver = dlsym(h, "verbs_register_driver_34");
    dlclose(h);
    done = (verbs_register_driver != NULL);
    return done ? 0 : -1;
}

/* ── Device ops implementations ────────────────────────────────────────
 *
 * alloc_device and alloc_context are called by libibverbs when it
 * discovers an OdinLink-Five device via sysfs or /dev/ scan.
 *
 * The actual device lifecycle is handled by the verbs core library
 * (libodl_tb5_verbs.so). This plugin just wires the rdma-core
 * discovery into that library's device and context management.
 */

static struct verbs_device *odl_alloc_device(void *sysfs_dev)
{
    (void)sysfs_dev;

    struct verbs_device *vdev = calloc(1, sizeof(*vdev));
    if (!vdev) return NULL;

    vdev->device.node_type      = IBV_NODE_RNIC;
    vdev->device.transport_type = IBV_TRANSPORT_IB;
    vdev->ops = NULL;

    return vdev;
}

static void odl_uninit_device(struct verbs_device *vdev)
{
    free(vdev);
}

static struct verbs_context *odl_alloc_context(struct ibv_device *ibdev,
                                                int cmd_fd,
                                                void *private_data)
{
    (void)cmd_fd;
    (void)private_data;

    if (!ibdev) {
        errno = EINVAL;
        return NULL;
    }

    /* The device name is set by the verbs core during /dev/ scanning.
     * Format: "odl_tb5_0", "odl_tb5_1", etc. */
    int dev_index = 0;
    if (ibdev->name) {
        const char *n = strrchr(ibdev->name, '_');
        if (n) dev_index = atoi(n + 1);
    }

    /* Open the OdinLink-Five device */
    odl_tb5_t handle;
    int ret = odl_tb5_open(&handle, dev_index);
    if (ret != 0) {
        errno = ENODEV;
        return NULL;
    }

    /* Now we need to create a proper ibv_context. The verbs core
     * expects this function to return a struct verbs_context * that
     * embeds struct ibv_context at its end.
     *
     * However, the verbs_context from the public header uses a
     * different layout than what verbs_init_and_alloc_context
     * creates. Since we can't call the private init function
     * (it's not available in all rdma-core versions), we
     * allocate the context ourselves.
     *
     * Our approach: return a verbs_context where we set up the
     * query_device_ex and query_port function pointers directly.
     * ibv_query_device and ibv_query_port here are handled by
     * symbol interposition (ibv_query_device in ops.c).
     *
     * For the verbs_open_device path, the verbs core needs:
     *   ibv_context.cmd_fd  = -1 (we don't have a uverbs fd)
     *   ibv_context.device  = ibdev (from our device)
     *
     * The verbs_context functions like query_device_ex are used
     * by ibv_query_device_ex, not ibv_query_device (which we
     * intercept via dlsym). We don't need to set them here
     * since ibv_query_device is intercepted.
     */

    /* Create an ibv_context using the public API */
    struct ibv_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        odl_tb5_close(handle);
        errno = ENOMEM;
        return NULL;
    }

    ctx->device            = ibdev;
    ctx->cmd_fd            = -1;
    ctx->async_fd          = -1;
    ctx->num_comp_vectors  = 1;

    /* Initialize the ops table that poll_cq, post_send, etc. use */
    extern void odl_init_context_ops(struct ibv_context *);
    odl_init_context_ops(ctx);

    /* Store the ODL handle in the mutex field (hack for compatibility).
     * The handle will be used by the odl_free_context function which
     * calls odl_tb5_close. We need it accessible from the context.
     * We use the abi_compat field for this. */
    ctx->abi_compat = (void *)(intptr_t)handle;

    /* Return a verbs_context wrapping this ibv_context.
     * We don't use the verbs_context extended ops — we handle
     * everything through the ibv_context_ops + symbol interposition. */
    struct verbs_context *vctx = verbs_get_ctx(ctx);

    return vctx;
}

/* ── Device ops table ────────────────────────────────────────────────── */

static const struct verbs_device_ops odl_verbs_ops = {
    .name          = "odl_tb5",
    .match_min_abi_version = 34,
    .match_max_abi_version = 34,
    .match_table   = NULL,
    .static_providers = NULL,
    .match_device  = NULL,
    .alloc_context = odl_alloc_context,
    .import_context = NULL,
    .alloc_device  = odl_alloc_device,
    .uninit_device = odl_uninit_device,
};

/* ── Constructor (called by dlopen at libibverbs init time) ─────────── */

static void __attribute__((constructor)) odl_provider_register(void)
{
    if (resolve_symbols() != 0) {
        fprintf(stderr, "odl_tb5: cannot register provider (no verbs_register_driver_34)\n");
        return;
    }
    verbs_register_driver(&odl_verbs_ops);
}
