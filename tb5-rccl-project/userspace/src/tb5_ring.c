#include <tb5/tb5_ring.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#define TB5_RING_IOCTL_ENQUEUE_SEND _IOW('T', 1, struct tb5_ring_request)
#define TB5_RING_IOCTL_ENQUEUE_RECV _IOW('T', 2, struct tb5_ring_request)
#define TB5_RING_IOCTL_TEST_COMPLETION _IOR('T', 3, int)

struct tb5_ring_handle {
    int fd;
};

int tb5_ring_open(tb5_ring_t *handle) {
    int fd = open("/dev/tb5_ol_ring0", O_RDWR);
    if (fd < 0) {
        return errno;
    }
    struct tb5_ring_handle *h = malloc(sizeof(struct tb5_ring_handle));
    h->fd = fd;
    *handle = h;
    return 0;
}

void tb5_ring_close(tb5_ring_t handle) {
    if (handle) {
        close(handle->fd);
        free(handle);
    }
}

int tb5_ring_enqueue_send(tb5_ring_t handle, int dmabuf_fd, off_t offset, size_t len) {
    struct tb5_ring_request req = {dmabuf_fd, offset, len};
    return ioctl(handle->fd, TB5_RING_IOCTL_ENQUEUE_SEND, &req) < 0 ? errno : 0;
}

int tb5_ring_enqueue_recv(tb5_ring_t handle, int dmabuf_fd, off_t offset, size_t len) {
    struct tb5_ring_request req = {dmabuf_fd, offset, len};
    return ioctl(handle->fd, TB5_RING_IOCTL_ENQUEUE_RECV, &req) < 0 ? errno : 0;
}

int tb5_ring_test_completion(tb5_ring_t handle, int *completed) {
    int ret = ioctl(handle->fd, TB5_RING_IOCTL_TEST_COMPLETION, completed);
    return ret < 0 ? errno : 0;
}
