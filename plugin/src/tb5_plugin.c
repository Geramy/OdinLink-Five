#include <net_v7.h>
#include <tb5/tb5_ring.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

struct tb5_comm {
    tb5_ring_t ring;
    int refcount; // Reference count for send/recv comms
};

struct tb5_request {
    bool done;
    int size;
};

struct tb5_handle {
    int dev_id;
    char hw_id[64];
};

// Global array to store hardware IDs for each device
#define MAX_DEVICES 16
static char hardware_ids[MAX_DEVICES][64];
static int num_devices = 0;

static rcclDebugLogger_t logger = NULL;

static rcclResult_t tb5_init(rcclDebugLogger_t logFunction) {
    logger = logFunction;
    return rcclSuccess;
}

static rcclResult_t tb5_devices(int* ndev) {
    // Read /sys/bus/thunderbolt/devices for XDomain links
    DIR *dir = opendir("/sys/bus/thunderbolt/devices");
    if (!dir) {
        *ndev = 0;
        num_devices = 0;
        return rcclSuccess;
    }
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir)) && count < MAX_DEVICES) {
        if (strstr(entry->d_name, "domain")) {
            // Read the hardware ID (UUID) from the domain's root device
            char path[256];
            char root_dev[32];
            snprintf(root_dev, sizeof(root_dev), "%s-0", entry->d_name + 6); // Remove "domain" prefix, add "-0"
            snprintf(path, sizeof(path), "/sys/bus/thunderbolt/devices/%s/%s/unique_id", entry->d_name, root_dev);
            FILE *fp = fopen(path, "r");
            if (fp) {
                if (fgets(hardware_ids[count], sizeof(hardware_ids[count]), fp)) {
                    // Remove newline
                    char *newline = strchr(hardware_ids[count], '\n');
                    if (newline) *newline = '\0';
                }
                fclose(fp);
            } else {
                // Fallback: use domain name as ID
                strcpy(hardware_ids[count], entry->d_name);
            }
            count++;
        }
    }
    closedir(dir);
    *ndev = count;
    num_devices = count;
    return rcclSuccess;
}

static rcclResult_t tb5_getProperties(int dev, rcclNetProperties_v7_t* props) {
    strcpy(props->name, "Thunderbolt 5 DMA Ring");
    strcpy(props->pciPath, "/sys/bus/thunderbolt");
    props->guid = dev;
    props->ptrSupport = 1; // dmabuf
    props->speed = 80000; // 80 Gbps
    props->port = dev;
    props->maxComm = 1;
    props->maxRecvs = 1;
    return rcclSuccess;
}

static rcclResult_t tb5_listen(int dev, void* handle, void** listenComm) {
    // Create handle with device ID and hardware ID
    struct tb5_handle *handle_ptr = malloc(sizeof(struct tb5_handle));
    handle_ptr->dev_id = dev;
    if (dev < num_devices) {
        strcpy(handle_ptr->hw_id, hardware_ids[dev]);
    } else {
        strcpy(handle_ptr->hw_id, "unknown");
    }
    *listenComm = handle_ptr;
    return rcclSuccess;
}

static rcclResult_t tb5_connect(int dev, void* handle, void** sendComm, void** recvComm) {
    struct tb5_handle* remote_handle = (struct tb5_handle*)handle;
    int remote_dev = remote_handle ? remote_handle->dev_id : 0;  // Default to device 0 if no handle (for accept)

    // Get hardware IDs - remote HW ID comes from handle (provided by remote device)
    const char* local_hw_id = (dev < num_devices) ? hardware_ids[dev] : "unknown";
    const char* remote_hw_id = remote_handle ? remote_handle->hw_id : "unknown";
    printf("Connected: Local hardware ID: %s, Remote hardware ID: %s (provided by remote device via handle)\n", local_hw_id, remote_hw_id);

    struct tb5_comm* comm = malloc(sizeof(struct tb5_comm));
    if (tb5_ring_open(&comm->ring) != 0) {
        free(comm);
        return rcclSystemError;
    }
    comm->refcount = 2; // Both send and recv reference this comm
    *sendComm = comm;
    *recvComm = comm;
    return rcclSuccess;
}

static rcclResult_t tb5_accept(void* listenComm, void** recvComm, void** sendComm) {
    struct tb5_handle* listen_handle = (struct tb5_handle*)listenComm;
    int dev = listen_handle->dev_id;
    return tb5_connect(dev, NULL, sendComm, recvComm);
}

static rcclResult_t tb5_closeListen(void* listenComm) {
    free(listenComm);
    return rcclSuccess;
}

static rcclResult_t tb5_isend(void* sendComm, void* data, int size, int tag, void** request) {
    struct tb5_comm* comm = (struct tb5_comm*)sendComm;
    // Assume data is dmabuf fd, but actually need rcclMemPoolGetDmabufFd
    // For now, assume data is int fd
    int fd = *(int*)data;
    if (tb5_ring_enqueue_send(comm->ring, fd, 0, size) != 0) {
        return rcclSystemError;
    }
    struct tb5_request* req = malloc(sizeof(struct tb5_request));
    req->done = false;
    req->size = size;
    *request = req;
    return rcclSuccess;
}

static rcclResult_t tb5_irecv(void* recvComm, void* data, int size, int tag, void** request) {
    struct tb5_comm* comm = (struct tb5_comm*)recvComm;
    int fd = *(int*)data;
    if (tb5_ring_enqueue_recv(comm->ring, fd, 0, size) != 0) {
        return rcclSystemError;
    }
    struct tb5_request* req = malloc(sizeof(struct tb5_request));
    req->done = false;
    req->size = size;
    *request = req;
    return rcclSuccess;
}

static rcclResult_t tb5_iflush(void* sendComm, int dev, void** request) {
    // For simplicity
    struct tb5_request* req = malloc(sizeof(struct tb5_request));
    req->done = true;
    req->size = 0;
    *request = req;
    return rcclSuccess;
}

static rcclResult_t tb5_test(void* request, int* done, int* size) {
    struct tb5_request* req = (struct tb5_request*)request;
    // Note: In a real implementation, we'd need to track the comm per request
    // For now, assume completion is immediate since we're using sync operations
    req->done = true; // Mock completion
    *done = req->done;
    *size = req->size;
    return rcclSuccess;
}

static rcclResult_t tb5_closeSend(void* sendComm) {
    struct tb5_comm* comm = (struct tb5_comm*)sendComm;
    comm->refcount--;
    if (comm->refcount == 0) {
        tb5_ring_close(comm->ring);
        free(comm);
    }
    return rcclSuccess;
}

static rcclResult_t tb5_closeRecv(void* recvComm) {
    // Same as send - both decrement the same refcount
    return tb5_closeSend(recvComm);
}

rcclNet_v7_t rcclNetPlugin_v7 = {
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
