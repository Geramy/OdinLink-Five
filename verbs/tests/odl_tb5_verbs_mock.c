/*
 * OdinLink-Five User-Space Mock Device
 *
 * Simulates two OdinLink-Five "peers" connected via shared memory,
 * without any kernel module or Thunderbolt hardware.
 *
 * How it works:
 *   Creates two simulated device contexts connected by a shared-memory
 *   ring buffer. Each context exposes odl_tb5_t handles that can be
 *   opened and used with the standard odl_tb5_* API.
 *
 *   Data sent on one side is immediately available for receive on the
 *   other side. The mock supports both the legacy double-buffer and
 *   stream-based APIs.
 *
 * Build: gcc -shared -fPIC -o libodl_tb5_mock.so \
 *            odl_tb5_verbs_mock.c -lpthread
 *
 * Usage:
 *   LD_PRELOAD=libodl_tb5_mock.so <any verbs app>
 *   This intercepts both libodl_tb5 and provides simulated devices.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>

/* We need the real verbs types so our mock ibv_device matches the
 * layout that the verbs provider expects. */
#include <infiniband/verbs.h>

/* Minimal peer_info struct matching driver/uapi/odl_tb5_uapi.h layout */
struct mock_peer_info {
    uint8_t  uuid[16];
    uint32_t link_speed;
    uint32_t link_width;
    uint32_t state;
    uint32_t reserved;
    char     vendor_name[64];
    char     device_name[64];
};

/* ── Mock device constants ──────────────────────────────────────────── */

#define MOCK_MAX_DEVICES      2
#define MOCK_BUF_SIZE         (4UL * 1024 * 1024)  /* 4 MB per buffer */
#define MOCK_NUM_BUFFERS      2
#define MOCK_STREAM_MAX       64
#define MOCK_STREAM_QUEUE_DEPTH 256
#define MOCK_FRAME_SIZE       4096

/* ── Stream queue entry ─────────────────────────────────────────────── */

struct mock_msg {
    uint8_t  src_id;
    uint8_t  dst_id;
    uint32_t len;
    uint8_t  data[MOCK_FRAME_SIZE];
};

/* ── Shared memory ring ─────────────────────────────────────────────── */

struct mock_ring {
    pthread_mutex_t  lock;
    pthread_cond_t   cond;
    int              head;
    int              tail;
    int              count;
    struct mock_msg  msgs[MOCK_STREAM_QUEUE_DEPTH];
    bool             peer_connected;
};

/* ── Per-stream state ───────────────────────────────────────────────── */

struct mock_stream {
    bool               in_use;
    uint8_t            stream_id;
    struct mock_ring   tx_ring;
    struct mock_ring   rx_ring;
};

/* ── Shared memory segment (one per device index) ───────────────────── */

struct mock_shm {
    pthread_mutex_t    global_lock;
    pthread_cond_t     global_cond;
    bool               side_a_connected;
    bool               side_b_connected;

    struct mock_stream streams_a_to_b[MOCK_STREAM_MAX];
    struct mock_stream streams_b_to_a[MOCK_STREAM_MAX];

    /* Legacy double buffers */
    uint8_t            legacy_tx_a[MOCK_BUF_SIZE];
    uint8_t            legacy_rx_a[MOCK_BUF_SIZE];
    uint8_t            legacy_tx_b[MOCK_BUF_SIZE];
    uint8_t            legacy_rx_b[MOCK_BUF_SIZE];

    /* Simulated peer info */
    uint32_t           link_speed;
    uint32_t           link_width;
    uint32_t           state; /* odl_conn_state */
};

/* ── Per-side handle ────────────────────────────────────────────────── */

struct mock_handle {
    int                side;         /* 0 = A, 1 = B */
    int                dev_index;
    struct mock_shm   *shm;

    /* Legacy buffer tracking */
    int                tx_back;
    int                rx_back;

    /* Stream allocation */
    uint8_t            next_stream_id;
    pthread_mutex_t    stream_lock;
    bool               streams_in_use[MOCK_STREAM_MAX];
};

/* The mock intercepts only odl_tb5_* API functions.
 * Device discovery uses the real verbs provider's scan (need /dev/odl_tb5_N).
 * The ibv_open_device path goes through the verbs provider which calls
 * odl_tb5_open — which we intercept.
 *
 * Note: we do NOT intercept odl_ibv_open_device — the verbs provider's
 * version receives a valid struct odl_verbs_device* from its own scan,
 * so it can safely cast it. */

/* ── Shared memory management ───────────────────────────────────────── */

/* ── Shared memory management ───────────────────────────────────────── */

#include <sys/stat.h>

static pthread_mutex_t shm_lock = PTHREAD_MUTEX_INITIALIZER;
static int shm_count = 0;

static struct mock_shm *mock_create_shm(int dev_index)
{
    /* Use malloc'd memory instead of shm_open for portability */
    struct mock_shm *shm = calloc(1, sizeof(*shm));
    if (!shm) return NULL;

    /* Initialize locks (no PTHREAD_PROCESS_SHARED needed for single-process) */
    pthread_mutex_init(&shm->global_lock, NULL);
    pthread_cond_init(&shm->global_cond, NULL);

    for (int i = 0; i < MOCK_STREAM_MAX; i++) {
        pthread_mutex_init(&shm->streams_a_to_b[i].tx_ring.lock, NULL);
        pthread_mutex_init(&shm->streams_a_to_b[i].rx_ring.lock, NULL);
        pthread_cond_init(&shm->streams_a_to_b[i].tx_ring.cond, NULL);
        pthread_cond_init(&shm->streams_a_to_b[i].rx_ring.cond, NULL);
        pthread_mutex_init(&shm->streams_b_to_a[i].tx_ring.lock, NULL);
        pthread_mutex_init(&shm->streams_b_to_a[i].rx_ring.lock, NULL);
        pthread_cond_init(&shm->streams_b_to_a[i].tx_ring.cond, NULL);
        pthread_cond_init(&shm->streams_b_to_a[i].rx_ring.cond, NULL);
    }

    shm->link_speed = 40;
    shm->link_width = 4;
    shm->state = 4;

    return shm;
}

/* ── Mock ring operations ───────────────────────────────────────────── */

static int mock_ring_push(struct mock_ring *ring,
                           const uint8_t *data, uint32_t len,
                           uint8_t src_id, uint8_t dst_id)
{
    pthread_mutex_lock(&ring->lock);

    while (ring->count >= MOCK_STREAM_QUEUE_DEPTH) {
        pthread_cond_wait(&ring->cond, &ring->lock);
    }

    int slot = ring->tail;
    ring->msgs[slot].len = len > MOCK_FRAME_SIZE ? MOCK_FRAME_SIZE : len;
    ring->msgs[slot].src_id = src_id;
    ring->msgs[slot].dst_id = dst_id;
    memcpy(ring->msgs[slot].data, data, ring->msgs[slot].len);

    ring->tail = (ring->tail + 1) % MOCK_STREAM_QUEUE_DEPTH;
    ring->count++;

    pthread_cond_signal(&ring->cond);
    pthread_mutex_unlock(&ring->lock);
    return 0;
}

static int mock_ring_pop(struct mock_ring *ring,
                          uint8_t *data, uint32_t buf_len,
                          uint8_t *src_id, uint32_t *actual_len)
{
    pthread_mutex_lock(&ring->lock);

    while (ring->count == 0) {
        pthread_cond_wait(&ring->cond, &ring->lock);
    }

    int slot = ring->head;
    uint32_t to_copy = ring->msgs[slot].len > buf_len ?
                        buf_len : ring->msgs[slot].len;
    memcpy(data, ring->msgs[slot].data, to_copy);
    if (src_id) *src_id = ring->msgs[slot].src_id;
    if (actual_len) *actual_len = to_copy;

    ring->head = (ring->head + 1) % MOCK_STREAM_QUEUE_DEPTH;
    ring->count--;

    pthread_cond_signal(&ring->cond);
    pthread_mutex_unlock(&ring->lock);
    return 0;
}

/* ── Mock odl_tb5 API implementation ────────────────────────────────── */

/* These symbols override libodl_tb5.so when LD_PRELOADed */

int odl_tb5_open(void **handle, int index)
{
    if (index < 0 || index >= MOCK_MAX_DEVICES)
        return -ENODEV;

    pthread_mutex_lock(&shm_lock);

    struct mock_shm *shm = mock_create_shm(index);
    if (!shm) {
        pthread_mutex_unlock(&shm_lock);
        return -ENOMEM;
    }

    /* Determine which side we are */
    int side = -1;
    pthread_mutex_lock(&shm->global_lock);
    if (!shm->side_a_connected) {
        shm->side_a_connected = true;
        side = 0;
    } else if (!shm->side_b_connected) {
        shm->side_b_connected = true;
        side = 1;
    }
    shm->state = 4; /* READY */
    pthread_cond_broadcast(&shm->global_cond);
    pthread_mutex_unlock(&shm->global_lock);

    if (side < 0) {
        pthread_mutex_unlock(&shm_lock);
        return -EBUSY;
    }

    struct mock_handle *h = calloc(1, sizeof(*h));
    if (!h) {
        pthread_mutex_unlock(&shm_lock);
        return -ENOMEM;
    }

    h->side = side;
    h->dev_index = index;
    h->shm = shm;
    h->tx_back = 0;
    h->rx_back = 0;
    h->next_stream_id = 1;
    pthread_mutex_init(&h->stream_lock, NULL);

    *handle = h;

    pthread_mutex_unlock(&shm_lock);
    return 0;
}

int odl_tb5_open_path(void **handle, const char *path)
{
    int index = 0;
    if (sscanf(path, "/dev/odl_tb5_%d", &index) >= 1)
        return odl_tb5_open(handle, index);
    return -ENODEV;
}

void odl_tb5_close(void *handle)
{
    struct mock_handle *h = handle;
    if (!h) return;

    pthread_mutex_lock(&h->shm->global_lock);
    if (h->side == 0)
        h->shm->side_a_connected = false;
    else
        h->shm->side_b_connected = false;
    h->shm->state = 0;
    pthread_mutex_unlock(&h->shm->global_lock);

    pthread_mutex_destroy(&h->stream_lock);
    free(h);
}

int odl_tb5_get_peer(void *handle, void *info)
{
    struct mock_handle *h = handle;
    struct mock_peer_info *p = info;

    memset(p, 0, sizeof(*p));
    p->link_speed = h->shm->link_speed;
    p->link_width = h->shm->link_width;
    p->state = h->shm->state;
    return 0;
}

int odl_tb5_wait_peer(void *handle, int timeout_ms)
{
    struct mock_handle *h = handle;

    pthread_mutex_lock(&h->shm->global_lock);
    if (h->shm->state >= 4) {
        pthread_mutex_unlock(&h->shm->global_lock);
        return 0;
    }

    struct timespec ts;
    if (timeout_ms > 0) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&h->shm->global_cond,
                                &h->shm->global_lock, &ts);
    } else {
        pthread_cond_wait(&h->shm->global_cond, &h->shm->global_lock);
    }

    int state = h->shm->state;
    pthread_mutex_unlock(&h->shm->global_lock);
    return state >= 4 ? 0 : -ETIMEDOUT;
}

/* Legacy double-buffer API */
void *odl_tb5_tx_buffer(void *handle, size_t *size)
{
    struct mock_handle *h = handle;
    if (size) *size = MOCK_BUF_SIZE;
    return h->side == 0 ? h->shm->legacy_tx_a : h->shm->legacy_tx_b;
}

void *odl_tb5_rx_buffer(void *handle, size_t *size)
{
    struct mock_handle *h = handle;
    if (size) *size = MOCK_BUF_SIZE;
    return h->side == 0 ? h->shm->legacy_rx_a : h->shm->legacy_rx_b;
}

int odl_tb5_send(void *handle, size_t offset, size_t len)
{
    struct mock_handle *h = handle;

    uint8_t *tx = h->side == 0 ? h->shm->legacy_tx_a : h->shm->legacy_tx_b;
    uint8_t *rx = h->side == 0 ? h->shm->legacy_rx_b : h->shm->legacy_rx_a;

    if (offset + len > MOCK_BUF_SIZE) return -EINVAL;
    memcpy(rx + offset, tx + offset, len);

    return 0;
}

int odl_tb5_send_ctrl(void *handle, size_t offset, size_t len)
{
    return odl_tb5_send(handle, offset, len);
}

int odl_tb5_recv(void *handle, size_t offset, size_t len)
{
    (void)handle; (void)offset; (void)len;
    return 0;
}

int odl_tb5_swap_tx(void *handle)
{
    struct mock_handle *h = handle;
    h->tx_back = !h->tx_back;
    return 0;
}

int odl_tb5_swap_rx(void *handle)
{
    struct mock_handle *h = handle;
    h->rx_back = !h->rx_back;
    return 0;
}

int odl_tb5_send_dmabuf(void *handle, int dmabuf_fd,
                          off_t offset, size_t len)
{
    (void)dmabuf_fd;
    /* In mock mode, dmabuf sends are same as regular sends */
    return odl_tb5_send(handle, (size_t)offset, len);
}

int odl_tb5_recv_dmabuf(void *handle, int dmabuf_fd,
                          off_t offset, size_t len)
{
    (void)dmabuf_fd;
    return odl_tb5_recv(handle, (size_t)offset, len);
}

int odl_tb5_poll(void *handle, void *comp)
{
    (void)handle;
    memset(comp, 0, 16); /* odl_tb5_completion is 16 bytes */
    return 0;
}

int odl_tb5_wait_tx(void *handle, void *comp)
{
    return odl_tb5_poll(handle, comp);
}

int odl_tb5_wait_rx(void *handle, void *comp)
{
    return odl_tb5_poll(handle, comp);
}

int odl_tb5_get_buf_info(void *handle, uint64_t *tx_size,
                          uint64_t *rx_size)
{
    (void)handle;
    if (tx_size) *tx_size = MOCK_BUF_SIZE;
    if (rx_size) *rx_size = MOCK_BUF_SIZE;
    return 0;
}

int odl_tb5_get_fd(void *handle)
{
    (void)handle;
    return -1; /* No real fd in mock mode */
}

/* Stream API */

int odl_tb5_stream_open(void *handle, uint8_t filter_id,
                         uint8_t *stream_id_out)
{
    struct mock_handle *h = handle;
    (void)filter_id;

    pthread_mutex_lock(&h->stream_lock);

    uint8_t sid = 0;
    for (int i = 1; i < MOCK_STREAM_MAX; i++) {
        if (!h->streams_in_use[i]) {
            h->streams_in_use[i] = true;
            sid = (uint8_t)i;
            break;
        }
    }

    if (sid == 0) {
        pthread_mutex_unlock(&h->stream_lock);
        return -ENOSPC;
    }

    /* Initialize the stream's rings */
    struct mock_stream *streams = h->side == 0 ?
        h->shm->streams_a_to_b : h->shm->streams_b_to_a;
    struct mock_stream *s = &streams[sid];
    s->in_use = true;
    s->stream_id = sid;

    pthread_mutex_unlock(&h->stream_lock);

    if (stream_id_out) *stream_id_out = sid;
    return 0;
}

int odl_tb5_stream_close(void *handle, uint8_t stream_id)
{
    struct mock_handle *h = handle;

    pthread_mutex_lock(&h->stream_lock);
    h->streams_in_use[stream_id] = false;
    pthread_mutex_unlock(&h->stream_lock);

    return 0;
}

int odl_tb5_stream_send(void *handle, uint8_t stream_id,
                         uint8_t dst_id, const void *data,
                         uint32_t len)
{
    struct mock_handle *h = handle;
    (void)dst_id;

    /* Route to the other side's RX ring */
    struct mock_stream *streams = h->side == 0 ?
        h->shm->streams_b_to_a : h->shm->streams_a_to_b;

    if (stream_id >= MOCK_STREAM_MAX)
        return -ENOENT;

    /* Auto-activate the target stream if not yet active.
     * This handles the case where the send side opened the stream
     * but the receive side hasn't. */
    if (!streams[stream_id].in_use) {
        streams[stream_id].in_use = true;
        streams[stream_id].stream_id = stream_id;
    }

    return mock_ring_push(&streams[stream_id].rx_ring,
                           data, len, stream_id, 0);
}

int odl_tb5_stream_recv(void *handle, uint8_t stream_id,
                         void *buf, uint32_t buf_len,
                         uint8_t *src_id, uint32_t *actual_len)
{
    struct mock_handle *h = handle;

    struct mock_stream *streams = h->side == 0 ?
        h->shm->streams_a_to_b : h->shm->streams_b_to_a;

    if (stream_id >= MOCK_STREAM_MAX || !streams[stream_id].in_use)
        return -ENOENT;

    return mock_ring_pop(&streams[stream_id].rx_ring,
                          buf, buf_len, src_id, actual_len);
}

int odl_tb5_stream_wait_tx(void *handle, uint8_t stream_id,
                            uint32_t timeout_ms)
{
    (void)handle; (void)stream_id; (void)timeout_ms;
    return 0;
}

int odl_tb5_stream_wait_rx(void *handle, uint8_t stream_id,
                            uint32_t timeout_ms)
{
    (void)handle; (void)stream_id; (void)timeout_ms;
    return 0;
}

int odl_tb5_stream_send_dmabuf(void *handle, uint8_t stream_id,
                                uint8_t dst_id, int dmabuf_fd,
                                uint64_t offset, uint64_t len)
{
    (void)dmabuf_fd;
    return odl_tb5_stream_send(handle, stream_id, dst_id,
                                NULL, (uint32_t)len);
}

int odl_tb5_stream_recv_dmabuf(void *handle, uint8_t stream_id,
                                int dmabuf_fd, uint64_t offset,
                                uint64_t len)
{
    (void)dmabuf_fd;
    /* Can't receive to dmabuf in mock mode */
    return 0;
}
