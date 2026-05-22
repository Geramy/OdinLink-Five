/*
 * OdinLink Verbs Provider — Memory Region Registration
 *
 * ibv_reg_mr:        Register host memory for DMA. The kernel driver
 *                    handles host memory via its mmap'd buffers.
 *
 * ibv_reg_dmabuf_mr: Register a DMA-buf file descriptor for zero-copy
 *                    GPU memory. The fd is passed to the kernel driver's
 *                    send_dmabuf/recv_dmabuf ioctls. This is the same
 *                    approach as Apple's ibv_reg_dmabuf_mr on macOS.
 */

#include "odl_tb5_verbs.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

struct ibv_mr *odl_reg_mr(struct ibv_pd *pd, void *addr,
                           size_t length, uint64_t hca_va,
                           int access)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(pd->context);

    struct odl_verbs_mr *mr = calloc(1, sizeof(*mr));
    if (!mr) { errno = ENOMEM; return NULL; }

    mr->base.addr    = addr;
    mr->base.length  = length;
    mr->base.handle  = 0;
    mr->base.lkey    = 0;
    mr->base.rkey    = 0;
    mr->base.context = pd->context;
    mr->mr_type      = 0; /* host */
    mr->access_flags = access;
    mr->host_addr    = addr;
    mr->host_length  = length;
    mr->dmabuf_fd    = -1;
    mr->iova         = hca_va ? hca_va : (uint64_t)(uintptr_t)addr;

    pthread_mutex_lock(&ctx->mr_lock);
    if (ctx->nmrs >= ODL_VERBS_MAX_MRS) {
        pthread_mutex_unlock(&ctx->mr_lock);
        free(mr);
        errno = ENOMEM;
        return NULL;
    }
    ctx->mrs[ctx->nmrs++] = mr;
    pthread_mutex_unlock(&ctx->mr_lock);

    odl_loginfo("reg_mr: addr=%p len=%zu lkey=%06x rkey=%06x",
                 addr, length, mr->base.lkey, mr->base.rkey);
    ODL_TRACE_EXIT();
    return &mr->base;
}

struct ibv_mr *odl_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
                                  size_t length, uint64_t iova,
                                  int fd, int access)
{
    ODL_TRACE_ENTRY();
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(pd->context);

    struct odl_verbs_mr *mr = calloc(1, sizeof(*mr));
    if (!mr) { errno = ENOMEM; return NULL; }

    /* Duplicate the dmabuf fd so we own it */
    mr->dmabuf_fd = fcntl(fd, F_DUPFD_CLOEXEC, 3);
    if (mr->dmabuf_fd < 0) {
        odl_logerr("fcntl(F_DUPFD_CLOEXEC) failed: %s", strerror(errno));
        free(mr);
        return NULL;
    }

    mr->base.addr        = NULL;
    mr->base.length      = length;
    mr->base.handle      = 0;
    mr->base.context     = pd->context;
    mr->mr_type          = 1; /* dmabuf */
    mr->access_flags     = access;
    mr->dmabuf_offset    = offset;
    mr->iova             = iova;
    mr->host_addr        = NULL;
    mr->host_length      = 0;

    /* Use a unique handle for lookup during send/recv */
    mr->base.lkey = (uint32_t)(uintptr_t)mr;
    mr->base.rkey = mr->base.lkey;

    pthread_mutex_lock(&ctx->mr_lock);
    if (ctx->nmrs >= ODL_VERBS_MAX_MRS) {
        pthread_mutex_unlock(&ctx->mr_lock);
        close(mr->dmabuf_fd);
        free(mr);
        errno = ENOMEM;
        return NULL;
    }
    ctx->mrs[ctx->nmrs++] = mr;
    pthread_mutex_unlock(&ctx->mr_lock);

    odl_loginfo("reg_dmabuf_mr: fd=%d offset=%llu len=%zu iova=%llx "
                 "lkey=%06x",
                 fd, (unsigned long long)offset, length,
                 (unsigned long long)iova, mr->base.lkey);
    ODL_TRACE_EXIT();
    return &mr->base;
}

int odl_dereg_mr(struct ibv_mr *mr)
{
    ODL_TRACE_ENTRY();
    if (!mr) return -EINVAL;

    struct odl_verbs_mr *omr = odl_mr_from_ibv(mr);
    struct odl_verbs_context *ctx = odl_ctx_from_ibv(mr->context);

    pthread_mutex_lock(&ctx->mr_lock);
    for (int i = 0; i < ctx->nmrs; i++) {
        if (ctx->mrs[i] == omr) {
            ctx->mrs[i] = ctx->mrs[--ctx->nmrs];
            break;
        }
    }
    pthread_mutex_unlock(&ctx->mr_lock);

    if (omr->dmabuf_fd >= 0) {
        close(omr->dmabuf_fd);
        odl_loginfo("dereg_mr: dmabuf fd=%d closed", omr->dmabuf_fd);
    }

    free(omr);
    ODL_TRACE_EXIT_VAL(0);
}
