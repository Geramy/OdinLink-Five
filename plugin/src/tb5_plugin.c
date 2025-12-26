#include <nccl/net_v7.h>
#include <tb5/tb5_ring.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

struct tb5_comm {
    tb5_ring_t ring;
    int refcount; // Reference count for send/recv comms
};

struct tb5_request {
    bool done;
    int size;
};

static ncclDebugLogger_t logger = NULL;

static ncclResult_t tb5_init(ncclDebugLogger_t logFunction) {
    logger = logFunction;
    return ncclSuccess;
}

static ncclResult_t tb5_devices(int* ndev) {
    // Read /sys/bus/thunderbolt/devices for XDomain links
    DIR *dir = opendir("/sys/bus/thunderbolt/devices");
    if (!dir) {
        *ndev = 0;
        return ncclSuccess;
    }
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir))) {
        if (strstr(entry->d_name, "domain")) {
            count++;
        }
    }
    closedir(dir);
    *ndev = count;
    return ncclSuccess;
}

static ncclResult_t tb5_getProperties(int dev, ncclNetProperties_v7_t* props) {
    strcpy(props->name, "Thunderbolt 5 DMA Ring");
    strcpy(props->pciPath, "/sys/bus/thunderbolt");
    props->guid = dev;
    props->ptrSupport = 1; // dmabuf
    props->speed = 80000; // 80 Gbps
    props->port = dev;
    props->maxComm = 1;
    props->maxRecvs = 1;
    return ncclSuccess;
}

static ncclResult_t tb5_listen(int dev, void* handle, void** listenComm) {
    // For simplicity, handle is dev id
    int *dev_ptr = malloc(sizeof(int));
    *dev_ptr = dev;
    *listenComm = dev_ptr;
    return ncclSuccess;
}

static ncclResult_t tb5_connect(int dev, void* handle, void** sendComm, void** recvComm) {
    int remote_dev = handle ? *(int*)handle : 0;  // Default to device 0 if no handle (for accept)
    printf("Connecting from device %d to device %d\n", dev, remote_dev);

    struct tb5_comm* comm = malloc(sizeof(struct tb5_comm));
    if (tb5_ring_open(&comm->ring) != 0) {
        free(comm);
        return ncclSystemError;
    }
    comm->refcount = 2; // Both send and recv reference this comm
    *sendComm = comm;
    *recvComm = comm;
    return ncclSuccess;
}

static ncclResult_t tb5_accept(void* listenComm, void** recvComm, void** sendComm) {
    int dev = *(int*)listenComm;
    return tb5_connect(dev, NULL, sendComm, recvComm);
}

static ncclResult_t tb5_closeListen(void* listenComm) {
    free(listenComm);
    return ncclSuccess;
}

static ncclResult_t tb5_isend(void* sendComm, void* data, int size, int tag, void** request) {
    struct tb5_comm* comm = (struct tb5_comm*)sendComm;
    // Assume data is dmabuf fd, but actually need ncclMemPoolGetDmabufFd
    // For now, assume data is int fd
    int fd = *(int*)data;
    if (tb5_ring_enqueue_send(comm->ring, fd, 0, size) != 0) {
        return ncclSystemError;
    }
    struct tb5_request* req = malloc(sizeof(struct tb5_request));
    req->done = false;
    req->size = size;
    *request = req;
    return ncclSuccess;
}

static ncclResult_t tb5_irecv(void* recvComm, void* data, int size, int tag, void** request) {
    struct tb5_comm* comm = (struct tb5_comm*)recvComm;
    int fd = *(int*)data;
    if (tb5_ring_enqueue_recv(comm->ring, fd, 0, size) != 0) {
        return ncclSystemError;
    }
    struct tb5_request* req = malloc(sizeof(struct tb5_request));
    req->done = false;
    req->size = size;
    *request = req;
    return ncclSuccess;
}

static ncclResult_t tb5_iflush(void* sendComm, int dev, void** request) {
    // For simplicity
    struct tb5_request* req = malloc(sizeof(struct tb5_request));
    req->done = true;
    req->size = 0;
    *request = req;
    return ncclSuccess;
}

static ncclResult_t tb5_test(void* request, int* done, int* size) {
    struct tb5_request* req = (struct tb5_request*)request;
    // Note: In a real implementation, we'd need to track the comm per request
    // For now, assume completion is immediate since we're using sync operations
    req->done = true; // Mock completion
    *done = req->done;
    *size = req->size;
    return ncclSuccess;
}

static ncclResult_t tb5_closeSend(void* sendComm) {
    struct tb5_comm* comm = (struct tb5_comm*)sendComm;
    comm->refcount--;
    if (comm->refcount == 0) {
        tb5_ring_close(comm->ring);
        free(comm);
    }
    return ncclSuccess;
}

static ncclResult_t tb5_closeRecv(void* recvComm) {
    // Same as send - both decrement the same refcount
    return tb5_closeSend(recvComm);
}

ncclNet_v7_t ncclNetPlugin_v7 = {
    .name = "TB5_OL",
    .init = tb5_init,
    .devices = tb5_devices,
    .getProperties = tb5_getProperties,
    .listen = tb5_listen,
    .connect = tb5_connect,
    .accept = tb5_accept,
    .closeListen = tb5_closeListen,
    .isend = tb5_isend,
    .irecv = tb5_irecv,
    .iflush = tb5_iflush,
    .test = tb5_test,
    .closeSend = tb5_closeSend,
    .closeRecv = tb5_closeRecv,
};
