/*
 * OdinLink Thunderbolt 5 - RCCL Net v7 Plugin
 *
 * Implements the RCCL network plugin interface using the OdinLink
 * TB5 driver. Uses the external buffer path (DMA-buf) - RCCL provides
 * its own GPU buffers which are passed through directly to the driver.
 *
 * Exposes shared-memory stats at /run/odl_tb5/rccl_stats so the
 * daemon can report TX/RX counters, op counts, and uptime.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "net_v7.h"
#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_rccl_stats.h>

#define ODL_TB5_MAX_RCCL_DEVICES 16

static rcclDebugLogger_t odl_logger;
static char hw_ids[ODL_TB5_MAX_RCCL_DEVICES][64];
static int num_devices;

static struct odl_rccl_stats *stats_map;
static int stats_fd = -1;

/* Communication handle - shared by send and recv */
struct odl_tb5_comm {
	odl_tb5_t handle;
	int refcount;
};

/* Async request tracking */
struct odl_tb5_request {
	bool done;
	int  size;
};

/* Listen handle */
struct odl_tb5_listen_handle {
	int  dev_id;
	char hw_id[64];
};

static uint64_t clock_mono_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void stats_init(void)
{
	mkdir(ODL_RCCL_STATS_DIR, 0755);

	stats_fd = open(ODL_RCCL_STATS_PATH, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (stats_fd < 0)
		return;

	if (ftruncate(stats_fd, sizeof(struct odl_rccl_stats)) < 0) {
		close(stats_fd);
		stats_fd = -1;
		return;
	}

	stats_map = mmap(NULL, sizeof(struct odl_rccl_stats),
			 PROT_READ | PROT_WRITE, MAP_SHARED, stats_fd, 0);
	if (stats_map == MAP_FAILED) {
		stats_map = NULL;
		close(stats_fd);
		stats_fd = -1;
		return;
	}

	memset(stats_map, 0, sizeof(*stats_map));
	stats_map->magic = ODL_RCCL_STATS_MAGIC;
	stats_map->version = ODL_RCCL_STATS_VERSION;
	stats_map->start_time_ns = clock_mono_ns();
	__atomic_store_n(&stats_map->active, 1, __ATOMIC_RELEASE);
}

static void stats_cleanup(void)
{
	if (stats_map) {
		__atomic_store_n(&stats_map->active, 0, __ATOMIC_RELEASE);
		munmap(stats_map, sizeof(struct odl_rccl_stats));
		stats_map = NULL;
	}
	if (stats_fd >= 0) {
		close(stats_fd);
		stats_fd = -1;
	}
}

static inline void stats_record_tx(int size)
{
	if (!stats_map)
		return;
	__atomic_add_fetch(&stats_map->tx_bytes, (uint64_t)size, __ATOMIC_RELAXED);
	__atomic_add_fetch(&stats_map->tx_ops, 1, __ATOMIC_RELAXED);
	__atomic_store_n(&stats_map->last_update_ns, clock_mono_ns(), __ATOMIC_RELAXED);
}

static inline void stats_record_rx(int size)
{
	if (!stats_map)
		return;
	__atomic_add_fetch(&stats_map->rx_bytes, (uint64_t)size, __ATOMIC_RELAXED);
	__atomic_add_fetch(&stats_map->rx_ops, 1, __ATOMIC_RELAXED);
	__atomic_store_n(&stats_map->last_update_ns, clock_mono_ns(), __ATOMIC_RELAXED);
}

static rcclResult_t odl_tb5_init(rcclDebugLogger_t logFunction)
{
	odl_logger = logFunction;
	stats_init();
	atexit(stats_cleanup);
	return rcclSuccess;
}

static rcclResult_t odl_tb5_devices(int *ndev)
{
	DIR *dir;
	struct dirent *entry;
	char path[256], uuid_buf[64];
	FILE *fp;

	num_devices = 0;
	dir = opendir("/sys/bus/thunderbolt/devices");
	if (!dir) {
		*ndev = 0;
		return rcclSuccess;
	}

	while ((entry = readdir(dir)) != NULL && num_devices < ODL_TB5_MAX_RCCL_DEVICES) {
		if (strncmp(entry->d_name, "domain", 6) != 0)
			continue;

		snprintf(path, sizeof(path),
			 "/sys/bus/thunderbolt/devices/%s/unique_id",
			 entry->d_name);
		fp = fopen(path, "r");
		if (!fp) {
			snprintf(hw_ids[num_devices], sizeof(hw_ids[0]),
				 "%s", entry->d_name);
		} else {
			if (fgets(uuid_buf, sizeof(uuid_buf), fp)) {
				uuid_buf[strcspn(uuid_buf, "\n")] = '\0';
				snprintf(hw_ids[num_devices], sizeof(hw_ids[0]),
					 "%s", uuid_buf);
			}
			fclose(fp);
		}
		num_devices++;
	}

	closedir(dir);
	*ndev = num_devices;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_getProperties(int dev, rcclNetProperties_v7_t *props)
{
	if (dev < 0 || dev >= num_devices)
		return rcclInvalidArgument;

	memset(props, 0, sizeof(*props));
	snprintf(props->name, sizeof(props->name),
		 "OdinLink TB5 DMA Ring");
	snprintf(props->pciPath, sizeof(props->pciPath),
		 "/sys/bus/thunderbolt");
	props->guid = (uint64_t)dev;
	props->ptrSupport = 1;
	props->speed = 80000;
	props->port = dev;
	props->maxComm = 1;
	props->maxRecvs = 1;

	return rcclSuccess;
}

static rcclResult_t odl_tb5_listen(int dev, void *handle,
				   void **listenComm)
{
	struct odl_tb5_listen_handle *lh;

	if (dev < 0 || dev >= num_devices)
		return rcclInvalidArgument;

	lh = calloc(1, sizeof(*lh));
	if (!lh)
		return rcclSystemError;

	lh->dev_id = dev;
	snprintf(lh->hw_id, sizeof(lh->hw_id), "%s", hw_ids[dev]);

	if (handle)
		memcpy(handle, lh->hw_id, sizeof(lh->hw_id));

	*listenComm = lh;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_connect(int dev, void *handle,
				    void **sendComm, void **recvComm)
{
	struct odl_tb5_comm *comm;
	int ret;

	comm = calloc(1, sizeof(*comm));
	if (!comm)
		return rcclSystemError;

	ret = odl_tb5_open(&comm->handle, dev);
	if (ret < 0) {
		free(comm);
		return rcclSystemError;
	}

	ret = odl_tb5_wait_peer(comm->handle, 5000);
	if (ret < 0) {
		odl_tb5_close(comm->handle);
		free(comm);
		return rcclSystemError;
	}

	comm->refcount = 2;
	*sendComm = comm;
	*recvComm = comm;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_accept(void *listenComm,
				   void **recvComm, void **sendComm)
{
	struct odl_tb5_listen_handle *lh = listenComm;
	struct odl_tb5_comm *comm;
	int ret;

	comm = calloc(1, sizeof(*comm));
	if (!comm)
		return rcclSystemError;

	ret = odl_tb5_open(&comm->handle, lh->dev_id);
	if (ret < 0) {
		free(comm);
		return rcclSystemError;
	}

	ret = odl_tb5_wait_peer(comm->handle, 5000);
	if (ret < 0) {
		odl_tb5_close(comm->handle);
		free(comm);
		return rcclSystemError;
	}

	comm->refcount = 2;
	*sendComm = comm;
	*recvComm = comm;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_closeListen(void *listenComm)
{
	free(listenComm);
	return rcclSuccess;
}

static rcclResult_t odl_tb5_isend(void *sendComm, void *data,
				  int size, int tag, void **request)
{
	struct odl_tb5_comm *comm = sendComm;
	struct odl_tb5_request *req;
	int ret;

	req = calloc(1, sizeof(*req));
	if (!req)
		return rcclSystemError;

	ret = odl_tb5_send_dmabuf(comm->handle, (int)(intptr_t)data, 0, size);
	if (ret < 0) {
		free(req);
		return rcclSystemError;
	}

	stats_record_tx(size);

	req->size = size;
	req->done = false;
	*request = req;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_irecv(void *recvComm, void *data,
				  int size, int tag, void **request)
{
	struct odl_tb5_comm *comm = recvComm;
	struct odl_tb5_request *req;
	int ret;

	req = calloc(1, sizeof(*req));
	if (!req)
		return rcclSystemError;

	ret = odl_tb5_recv_dmabuf(comm->handle, (int)(intptr_t)data, 0, size);
	if (ret < 0) {
		free(req);
		return rcclSystemError;
	}

	stats_record_rx(size);

	req->size = size;
	req->done = false;
	*request = req;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_iflush(void *sendComm, int dev,
				   void **request)
{
	return rcclSuccess;
}

static rcclResult_t odl_tb5_test(void *request, int *done, int *size)
{
	struct odl_tb5_request *req = request;

	if (!req) {
		*done = 1;
		return rcclSuccess;
	}

	*done = 1;
	if (size)
		*size = req->size;

	if (*done) {
		free(req);
	}

	return rcclSuccess;
}

static rcclResult_t odl_tb5_closeSend(void *sendComm)
{
	struct odl_tb5_comm *comm = sendComm;

	if (!comm)
		return rcclSuccess;

	if (__atomic_sub_fetch(&comm->refcount, 1, __ATOMIC_SEQ_CST) == 0) {
		odl_tb5_close(comm->handle);
		free(comm);
	}

	return rcclSuccess;
}

static rcclResult_t odl_tb5_closeRecv(void *recvComm)
{
	return odl_tb5_closeSend(recvComm);
}

rcclNet_v7_t rcclNetPlugin_v7 = {
	.name        = "ODL_TB5",
	.init        = odl_tb5_init,
	.devices     = odl_tb5_devices,
	.getProperties = odl_tb5_getProperties,
	.listen      = odl_tb5_listen,
	.connect     = odl_tb5_connect,
	.accept      = odl_tb5_accept,
	.closeListen = odl_tb5_closeListen,
	.isend       = odl_tb5_isend,
	.irecv       = odl_tb5_irecv,
	.iflush      = odl_tb5_iflush,
	.test        = odl_tb5_test,
	.closeSend   = odl_tb5_closeSend,
	.closeRecv   = odl_tb5_closeRecv,
};
