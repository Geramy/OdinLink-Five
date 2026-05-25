/*
 * OdinLink — RCCL-Style DMA-buf Send/Recv Test (AMD Parity)
 *
 * Exercises the same code path the RCCL plugin uses:
 *   odl_tb5_send_dmabuf(handle, dmabuf_fd, 0, size)
 *   odl_tb5_recv_dmabuf(handle, dmabuf_fd, 0, size)
 *
 * The fd is passed directly (not through ibv_reg_dmabuf_mr), matching
 * how RCCL's net_v7 plugin casts the data pointer as an fd.
 *
 * DMA-buf source priority:
 *   1. /dev/dma_heap/system  (real dmabuf, no GPU needed)
 *   2. /dev/dri/renderD*     (AMDGPU dmabuf export, needs AMD GPU)
 *   3. memfd                 (sealed, tests library plumbing only)
 *
 * Build:
 *   gcc -o test_rccl_dmabuf odl_tb5_test_rccl_dmabuf.c \
 *       -I../lib/include -I../driver/uapi -lodl_tb5 -lpthread
 *
 * Run:
 *   sudo ./test_rccl_dmabuf
 *   (loopback=1 mode: sudo insmod driver/odl_tb5.ko loopback=1)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>
#include <pthread.h>
#include <errno.h>

#include <odl_tb5/odl_tb5.h>

#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif
#ifndef F_SEAL_SEAL
#define F_SEAL_SEAL     0x0001
#define F_SEAL_SHRINK   0x0002
#define F_SEAL_GROW     0x0004
#endif

/* ── DMA-buf fd creation ──────────────────────────────────────────── */

static int try_dma_heap_fd(size_t size)
{
    int heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0)
        return -1;

    struct dma_heap_allocation_data data = {
        .len        = size,
        .fd_flags   = O_CLOEXEC | O_RDWR,
        .heap_flags = 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
        close(heap_fd);
        return -1;
    }
    close(heap_fd);
    printf("  [dma_heap] fd=%d size=%zu\n", data.fd, size);
    return data.fd;
}

static int try_amdgpu_dmabuf_fd(size_t size)
{
    /* Try to find an AMDGPU render node and export a dmabuf from it.
     * This requires an AMD GPU with the amdgpu driver loaded.  We open
     * /dev/dri/renderD128+ and use the AMDGPU_CTX ioctl to allocate
     * and export. */
    for (int node = 128; node < 140; node++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", node);

        int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0)
            continue;

        /* Check if this is amdgpu via the version ioctl */
        struct drm_version {
            int version_major, version_minor, version_patchlevel;
            size_t name_len, date_len, desc_len;
            char *name, *date, *desc;
        };
        struct drm_version v = {0};
        if (ioctl(fd, 0x00 /* DRM_IOCTL_VERSION */, &v) < 0) {
            close(fd);
            continue;
        }

        /* For simplicity: just close and note the path.
         * Real AMDGPU dmabuf export would use amdgpu_bo_alloc +
         * amdgpu_bo_export_to_dmabuf from libdrm_amdgpu.
         * For now we fall through to memfd. */
        close(fd);
        printf("  [amdgpu] found %s — needs libdrm_amdgpu for real export\n",
               path);
        return -1;
    }
    return -1;
}

static int make_sealed_memfd(size_t size)
{
    int fd = memfd_create("odl-rccl-test", MFD_ALLOW_SEALING);
    if (fd < 0)
        return -1;

    ftruncate(fd, (off_t)size);
    fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW);
    printf("  [memfd] fd=%d size=%zu (plumbing test only)\n", fd, size);
    return fd;
}

static int make_dmabuf_fd(size_t size, int *is_real)
{
    int fd;

    fd = try_dma_heap_fd(size);
    if (fd >= 0) { *is_real = 1; return fd; }

    fd = try_amdgpu_dmabuf_fd(size);
    if (fd >= 0) { *is_real = 1; return fd; }

    fd = make_sealed_memfd(size);
    if (fd >= 0) { *is_real = 0; return fd; }

    fprintf(stderr, "FAIL: no dmabuf source available\n");
    return -1;
}

/* ── Write a pattern into a dmabuf fd (if mappable) ─────────────── */
static int write_pattern(int fd, size_t size, unsigned char byte)
{
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED)
        return -1;
    memset(addr, byte, size);
    munmap(addr, size);
    return 0;
}

static int check_pattern(int fd, size_t size, unsigned char expected)
{
    void *addr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED)
        return -1;

    unsigned char *buf = addr;
    for (size_t i = 0; i < size; i++) {
        if (buf[i] != expected) {
            printf("  mismatch at offset %zu: got %02x expected %02x\n",
                   i, buf[i], expected);
            munmap(addr, size);
            return -1;
        }
    }
    munmap(addr, size);
    return 0;
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    int failures = 0;
    odl_tb5_t handle = NULL;
    int ret;

    printf("=================================\n");
    printf("OdinLink RCCL DMA-buf Test\n");
    printf("=================================\n");

    /* ── open device (loopback=1 works fine) ──────────────────── */
    ret = odl_tb5_open(&handle, 0);
    if (ret < 0) {
        printf("FAIL: odl_tb5_open(0) returned %d\n", ret);
        printf("SKIP: Load kernel module (loopback=1) first\n");
        return 77;
    }
    printf("1. Device opened\n");

    ret = odl_tb5_wait_peer(handle, 2000);
    if (ret < 0) {
        printf("FAIL: odl_tb5_wait_peer returned %d "
               "(no peer connected?)\n", ret);
        odl_tb5_close(handle);
        return 77;
    }
    printf("2. Peer ready\n");

    /* ── create two dmabuf fds: one for send, one for recv ───── */
    size_t buf_size = 65536;
    int is_real_send = 0, is_real_recv = 0;

    int send_fd = make_dmabuf_fd(buf_size, &is_real_send);
    assert(send_fd >= 0);
    printf("3a. Send dmabuf fd=%d (real=%d)\n", send_fd, is_real_send);

    int recv_fd = make_dmabuf_fd(buf_size, &is_real_recv);
    assert(recv_fd >= 0);
    printf("3b. Recv dmabuf fd=%d (real=%d)\n", recv_fd, is_real_recv);

    /* ── fill send buffer with pattern ────────────────────────── */
    if (is_real_send) {
        if (write_pattern(send_fd, buf_size, 0xAB) == 0)
            printf("4. Send buffer filled with 0xAB\n");
        else
            printf("4. Send buffer: mmap not available\n");
    } else {
        printf("4. Send buffer: memfd, skipping pattern write\n");
    }

    /* ── send via odl_tb5_send_dmabuf (RCCL-style) ───────────── */
    ret = odl_tb5_send_dmabuf(handle, send_fd, 0, buf_size);
    if (ret == 0) {
        printf("5. odl_tb5_send_dmabuf(handle, fd=%d, 0, %zu) → OK\n",
               send_fd, buf_size);
    } else {
        printf("5. odl_tb5_send_dmabuf returned %d (%s)\n",
               ret, strerror(-ret));
        if (!is_real_send)
            printf("   (expected with memfd — kernel rejects fake dmabuf)\n");
        failures++;
    }

    /* ── recv via odl_tb5_recv_dmabuf (RCCL-style) ───────────── */
    ret = odl_tb5_recv_dmabuf(handle, recv_fd, 0, buf_size);
    if (ret == 0) {
        printf("6. odl_tb5_recv_dmabuf(handle, fd=%d, 0, %zu) → OK\n",
               recv_fd, buf_size);
    } else {
        printf("6. odl_tb5_recv_dmabuf returned %d (%s)\n",
               ret, strerror(-ret));
        if (!is_real_recv)
            printf("   (expected with memfd — kernel rejects fake dmabuf)\n");
        failures++;
    }

    /* ── check recv buffer has data ───────────────────────────── */
    if (ret == 0 && is_real_recv) {
        if (check_pattern(recv_fd, buf_size, 0xAB) == 0)
            printf("7. Recv buffer pattern verified\n");
        else {
            printf("7. Recv buffer pattern MISMATCH\n");
            failures++;
        }
    } else {
        printf("7. Recv buffer: skipped pattern check\n");
    }

    /* ── stream-based dmabuf test ─────────────────────────────── */
    uint8_t stream_id = 0;
    ret = odl_tb5_stream_open(handle, 0, &stream_id);
    if (ret == 0) {
        printf("8. Stream opened: id=%u\n", stream_id);

        ret = odl_tb5_stream_send_dmabuf(handle, stream_id,
                                           stream_id, send_fd, 0,
                                           buf_size);
        printf("9. stream_send_dmabuf → %d (%s)\n",
               ret, ret == 0 ? "OK" : strerror(-ret));
        if (ret != 0) failures++;

        ret = odl_tb5_stream_recv_dmabuf(handle, stream_id,
                                           recv_fd, 0, buf_size);
        printf("10. stream_recv_dmabuf → %d (%s)\n",
               ret, ret == 0 ? "OK" : strerror(-ret));
        if (ret != 0) failures++;

        odl_tb5_stream_close(handle, stream_id);
        printf("11. Stream closed\n");
    } else {
        printf("8. stream_open → %d (skip stream dmabuf tests)\n", ret);
    }

    /* ── cleanup ──────────────────────────────────────────────── */
    close(send_fd);
    close(recv_fd);
    odl_tb5_close(handle);
    printf("12. Cleanup done\n");

    /* ── summary ──────────────────────────────────────────────── */
    printf("\n=================================\n");
    if (failures == 0) {
        printf("ALL RCCL DMABUF TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", failures);
    }
    printf("=================================\n");
    return failures > 0 ? 1 : 0;
}
