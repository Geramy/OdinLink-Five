/*
 * OdinLink TB5 — RCCL/NCCL network plugin, real v7 ABI.
 *
 * Replaces rccl/src/odl_tb5_plugin.c, which (a) exported the wrong symbol
 * name, (b) used a struct layout matching no real NCCL/RCCL version, and
 * (c) had a stub data path (cast the data pointer to a dmabuf FD and never
 * polled for completion).  See ODINLINK-FINDINGS.md, BUG 6/7/8.
 *
 * Structs/signatures here come from NVIDIA nccl plugins/net/example/nccl/net_v7.h.
 *
 * Transport model: OdinLink exposes one point-to-point device per TB link with
 * a TX front buffer and an RX front buffer, plus monotonic completion counters
 * (tx_completed/rx_completed vs tx_submitted/rx_submitted).  We therefore:
 *   - share one odl handle per device between the send and recv comm
 *     (TX and RX directions are independent),
 *   - stage payloads through the TX/RX buffers (host memory; NCCL_PTR_HOST),
 *   - implement test() by comparing the completion counter captured at submit
 *     time against the current completed counter.
 *
 * Because a device has a single TX and single RX buffer, run NCCL with one
 * channel:  NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>

#include "nccl/err.h"
#include "nccl/common.h"
#include "nccl/net_device.h"
#include "nccl/net_v7.h"

#include <odl_tb5/odl_tb5.h>

#ifndef NCCL_NET_HANDLE_MAXSIZE
#define NCCL_NET_HANDLE_MAXSIZE 128
#endif
#ifndef NCCL_PTR_HOST
#define NCCL_PTR_HOST 0x1
#endif

#define ODL_MAX_DEVICES 16
#define ODL_MAX_REQUESTS 256

static ncclDebugLogger_t odl_log;
#define ODLLOG(fmt, ...)                                                       \
    do {                                                                       \
        if (odl_log)                                                           \
            odl_log(NCCL_LOG_INFO, NCCL_NET, __FILE__, __LINE__,               \
                    "ODL_TB5: " fmt, ##__VA_ARGS__);                           \
    } while (0)

/* ---- device table: one shared handle per physical device ---------------- */
struct odl_dev {
    char        path[64];
    odl_tb5_t   h;
    int         refs;
    pthread_mutex_t lock;
};
static struct odl_dev devs[ODL_MAX_DEVICES];
static int  ndevs;
static pthread_mutex_t devs_lock = PTHREAD_MUTEX_INITIALIZER;

struct odl_comm {
    int  dev;
    int  is_send;
};

struct odl_request {
    int       used;
    int       is_send;
    int       size;      /* bytes expected/sent                     */
    uint32_t  seq;       /* completion counter value we wait for    */
    void     *dst;       /* recv only: where to copy on completion  */
    int       dev;
    int       consumed;
};
static struct odl_request reqs[ODL_MAX_REQUESTS];
static pthread_mutex_t reqs_lock = PTHREAD_MUTEX_INITIALIZER;

static struct odl_request *req_alloc(void)
{
    pthread_mutex_lock(&reqs_lock);
    for (int i = 0; i < ODL_MAX_REQUESTS; i++) {
        if (!reqs[i].used) {
            memset(&reqs[i], 0, sizeof(reqs[i]));
            reqs[i].used = 1;
            pthread_mutex_unlock(&reqs_lock);
            return &reqs[i];
        }
    }
    pthread_mutex_unlock(&reqs_lock);
    return NULL;
}
static void req_free(struct odl_request *r)
{
    pthread_mutex_lock(&reqs_lock);
    r->used = 0;
    pthread_mutex_unlock(&reqs_lock);
}

/* ---- device open/close with refcount ----------------------------------- */
static ncclResult_t dev_get(int dev, odl_tb5_t *out)
{
    if (dev < 0 || dev >= ndevs)
        return ncclInvalidArgument;
    pthread_mutex_lock(&devs_lock);
    if (devs[dev].refs == 0) {
        int rc = odl_tb5_open_path(&devs[dev].h, devs[dev].path);
        if (rc < 0) {
            pthread_mutex_unlock(&devs_lock);
            ODLLOG("open(%s) failed rc=%d", devs[dev].path, rc);
            return ncclSystemError;
        }
        /* link must be READY before any traffic */
        if (odl_tb5_wait_peer(devs[dev].h, 10000) < 0) {
            odl_tb5_close(devs[dev].h);
            pthread_mutex_unlock(&devs_lock);
            ODLLOG("peer not ready on %s", devs[dev].path);
            return ncclSystemError;
        }
    }
    devs[dev].refs++;
    *out = devs[dev].h;
    pthread_mutex_unlock(&devs_lock);
    return ncclSuccess;
}
static void dev_put(int dev)
{
    pthread_mutex_lock(&devs_lock);
    if (dev >= 0 && dev < ndevs && devs[dev].refs > 0) {
        if (--devs[dev].refs == 0)
            odl_tb5_close(devs[dev].h);
    }
    pthread_mutex_unlock(&devs_lock);
}

/* ---- plugin entry points ----------------------------------------------- */
static ncclResult_t odlInit(ncclDebugLogger_t logFunction)
{
    odl_log = logFunction;
    ndevs = 0;
    for (int i = 0; i < ODL_MAX_DEVICES && ndevs < ODL_MAX_DEVICES; i++) {
        char p[64];
        snprintf(p, sizeof(p), "/dev/odl_tb5_%d", i);
        if (access(p, R_OK | W_OK) == 0) {
            snprintf(devs[ndevs].path, sizeof(devs[ndevs].path), "%s", p);
            devs[ndevs].refs = 0;
            pthread_mutex_init(&devs[ndevs].lock, NULL);
            ndevs++;
        }
    }
    ODLLOG("init: %d device(s)", ndevs);
    return ndevs > 0 ? ncclSuccess : ncclSystemError;
}

static ncclResult_t odlDevices(int *ndev) { *ndev = ndevs; return ncclSuccess; }

static ncclResult_t odlGetProperties(int dev, ncclNetProperties_v7_t *props)
{
    if (dev < 0 || dev >= ndevs) return ncclInvalidArgument;
    static char nbuf[ODL_MAX_DEVICES][64];
    snprintf(nbuf[dev], sizeof(nbuf[dev]), "odl_tb5_%d", dev);

    int speed_mbps = 20000;             /* real link: 10 Gb/s x2 lanes       */
    odl_tb5_t h;
    if (dev_get(dev, &h) == ncclSuccess) {
        struct odl_tb5_peer_info info;
        if (odl_tb5_get_peer(h, &info) == 0 && info.link_speed > 0)
            speed_mbps = (int)(info.link_speed * info.link_width * 1000);
        dev_put(dev);
    }

    props->name             = nbuf[dev];
    props->pciPath          = NULL;
    props->guid             = (uint64_t)dev;
    props->ptrSupport       = NCCL_PTR_HOST;
    props->speed            = speed_mbps;
    props->port             = 0;
    props->latency          = 22.0f;    /* measured median, microseconds     */
    props->maxComms         = 4;
    props->maxRecvs         = 1;
    props->netDeviceType    = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    return ncclSuccess;
}

struct odl_handle { int dev; };

static ncclResult_t odlListen(int dev, void *handle, void **listenComm)
{
    if (dev < 0 || dev >= ndevs) return ncclInvalidArgument;
    struct odl_handle *hd = (struct odl_handle *)handle;
    struct odl_comm *c = calloc(1, sizeof(*c));
    if (!c) return ncclSystemError;
    hd->dev = dev;                       /* peer connects to the same link   */
    c->dev = dev;
    *listenComm = c;
    return ncclSuccess;
}

static ncclResult_t odlConnect(int dev, void *handle, void **sendComm,
                               ncclNetDeviceHandle_v7_t **sendDevComm)
{
    (void)handle;
    odl_tb5_t h;
    ncclResult_t r = dev_get(dev, &h);
    if (r != ncclSuccess) { *sendComm = NULL; return r; }
    struct odl_comm *c = calloc(1, sizeof(*c));
    if (!c) { dev_put(dev); return ncclSystemError; }
    c->dev = dev; c->is_send = 1;
    *sendComm = c;
    if (sendDevComm) *sendDevComm = NULL;   /* no device offload */
    return ncclSuccess;
}

static ncclResult_t odlAccept(void *listenComm, void **recvComm,
                              ncclNetDeviceHandle_v7_t **recvDevComm)
{
    struct odl_comm *lc = listenComm;
    odl_tb5_t h;
    ncclResult_t r = dev_get(lc->dev, &h);
    if (r != ncclSuccess) { *recvComm = NULL; return r; }
    struct odl_comm *c = calloc(1, sizeof(*c));
    if (!c) { dev_put(lc->dev); return ncclSystemError; }
    c->dev = lc->dev; c->is_send = 0;
    *recvComm = c;
    if (recvDevComm) *recvDevComm = NULL;
    return ncclSuccess;
}

/* Host-memory staging: registration is a no-op, handle is the pointer. */
static ncclResult_t odlRegMr(void *comm, void *data, int size, int type,
                             void **mhandle)
{
    (void)comm; (void)size;
    if (type != NCCL_PTR_HOST) return ncclInternalError;
    *mhandle = data;
    return ncclSuccess;
}
static ncclResult_t odlRegMrDmaBuf(void *comm, void *data, size_t size, int type,
                                   uint64_t offset, int fd, void **mhandle)
{
    (void)comm; (void)data; (void)size; (void)type; (void)offset; (void)fd;
    (void)mhandle;
    return ncclInternalError;           /* not supported -> NCCL uses regMr  */
}
static ncclResult_t odlDeregMr(void *comm, void *mhandle)
{
    (void)comm; (void)mhandle; return ncclSuccess;
}

static ncclResult_t odlIsend(void *sendComm, void *data, int size, int tag,
                             void *mhandle, void **request)
{
    (void)tag; (void)mhandle;
    struct odl_comm *c = sendComm;
    odl_tb5_t h = devs[c->dev].h;
    size_t cap = 0;
    void *tx = odl_tb5_tx_buffer(h, &cap);
    if (!tx || (size_t)size > cap) { *request = NULL; return ncclInternalError; }

    struct odl_tb5_completion pre;
    if (odl_tb5_poll(h, &pre) < 0) { *request = NULL; return ncclSystemError; }

    memcpy(tx, data, size);
    if (odl_tb5_send(h, 0, (size_t)size) < 0) { *request = NULL; return ncclSystemError; }

    struct odl_request *rq = req_alloc();
    if (!rq) { *request = NULL; return ncclInternalError; }
    rq->is_send = 1;
    rq->size    = size;
    rq->dev     = c->dev;
    rq->seq     = pre.tx_completed + 1;   /* this submission's completion    */
    *request = rq;
    return ncclSuccess;
}

static ncclResult_t odlIrecv(void *recvComm, int n, void **data, int *sizes,
                             int *tags, void **mhandles, void **request)
{
    (void)tags; (void)mhandles;
    struct odl_comm *c = recvComm;
    if (n != 1) return ncclInternalError;   /* maxRecvs = 1 */
    odl_tb5_t h = devs[c->dev].h;

    struct odl_tb5_completion pre;
    if (odl_tb5_poll(h, &pre) < 0) { *request = NULL; return ncclSystemError; }

    if (odl_tb5_recv(h, 0, (size_t)sizes[0]) < 0) { *request = NULL; return ncclSystemError; }

    struct odl_request *rq = req_alloc();
    if (!rq) { *request = NULL; return ncclInternalError; }
    rq->is_send = 0;
    rq->size    = sizes[0];
    rq->dst     = data[0];
    rq->dev     = c->dev;
    rq->seq     = pre.rx_completed + 1;
    *request = rq;
    return ncclSuccess;
}

static ncclResult_t odlIflush(void *recvComm, int n, void **data, int *sizes,
                              void **mhandles, void **request)
{
    (void)recvComm; (void)n; (void)data; (void)sizes; (void)mhandles;
    *request = NULL;                    /* host memory: nothing to flush     */
    return ncclSuccess;
}

static ncclResult_t odlTest(void *request, int *done, int *sizes)
{
    struct odl_request *rq = request;
    if (!rq) { *done = 1; return ncclSuccess; }

    struct odl_tb5_completion c;
    if (odl_tb5_poll(devs[rq->dev].h, &c) < 0) return ncclSystemError;

    uint32_t cur = rq->is_send ? c.tx_completed : c.rx_completed;
    if (cur < rq->seq) { *done = 0; return ncclSuccess; }

    if (!rq->is_send && rq->dst) {
        size_t cap = 0;
        void *rx = odl_tb5_rx_buffer(devs[rq->dev].h, &cap);
        if (rx) memcpy(rq->dst, rx, (size_t)rq->size);
    }
    *done = 1;
    if (sizes) sizes[0] = rq->size;
    req_free(rq);
    return ncclSuccess;
}

static ncclResult_t odlCloseSend(void *sendComm)
{
    struct odl_comm *c = sendComm;
    if (c) { dev_put(c->dev); free(c); }
    return ncclSuccess;
}
static ncclResult_t odlCloseRecv(void *recvComm)
{
    struct odl_comm *c = recvComm;
    if (c) { dev_put(c->dev); free(c); }
    return ncclSuccess;
}
static ncclResult_t odlCloseListen(void *listenComm)
{
    free(listenComm);
    return ncclSuccess;
}
static ncclResult_t odlGetDeviceMr(void *comm, void *mhandle, void **dptr_mhandle)
{
    (void)comm; (void)mhandle; (void)dptr_mhandle;
    return ncclInternalError;
}
static ncclResult_t odlIrecvConsumed(void *recvComm, int n, void *request)
{
    (void)recvComm; (void)n; (void)request;
    return ncclSuccess;
}

/* Exported with the name RCCL/NCCL actually looks for. */
ncclNet_v7_t ncclNetPlugin_v7 = {
    .name          = "ODL_TB5",
    .init          = odlInit,
    .devices       = odlDevices,
    .getProperties = odlGetProperties,
    .listen        = odlListen,
    .connect       = odlConnect,
    .accept        = odlAccept,
    .regMr         = odlRegMr,
    .regMrDmaBuf   = odlRegMrDmaBuf,
    .deregMr       = odlDeregMr,
    .isend         = odlIsend,
    .irecv         = odlIrecv,
    .iflush        = odlIflush,
    .test          = odlTest,
    .closeSend     = odlCloseSend,
    .closeRecv     = odlCloseRecv,
    .closeListen   = odlCloseListen,
    .getDeviceMr   = odlGetDeviceMr,
    .irecvConsumed = odlIrecvConsumed,
};
