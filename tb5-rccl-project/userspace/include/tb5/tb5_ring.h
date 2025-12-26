#ifndef TB5_RING_H
#define TB5_RING_H

#include <sys/types.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tb5_ring_request {
    int dmabuf_fd;
    off_t offset;
    size_t len;
};

typedef struct tb5_ring_handle *tb5_ring_t;

int tb5_ring_open(tb5_ring_t *handle);
void tb5_ring_close(tb5_ring_t handle);

int tb5_ring_enqueue_send(tb5_ring_t handle, int dmabuf_fd, off_t offset, size_t len);
int tb5_ring_enqueue_recv(tb5_ring_t handle, int dmabuf_fd, off_t offset, size_t len);
int tb5_ring_test_completion(tb5_ring_t handle, int *completed);

#ifdef __cplusplus
}
#endif

#endif // TB5_RING_H
