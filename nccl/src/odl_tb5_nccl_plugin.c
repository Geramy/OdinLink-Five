#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>

#include "net.h"
#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_nccl_stats.h>
#include <odl_tb5/odl_compress.h>

#define ODL_NCCL_MAX_DEVICES 16
#define ODL_NCCL_MAX_MR      64
#define ODL_NCCL_MAX_COMMS   16

/* ── Minimal CUDA driver types for dlopen ────────────────────────── */

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemRangeHandleType;
#define CUDA_SUCCESS 0
#define CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 1ULL

typedef CUresult (*cuMemGetHandleForAddressRange_fn_t)(
    void *handle, CUdeviceptr dptr, size_t size,
    CUmemRangeHandleType handleType, unsigned long long flags);

/* ── Plugin state ────────────────────────────────────────────────── */

static ncclDebugLogger_t odl_logger;
static bool cuda_available = false;
static void *cuda_lib = NULL;
static cuMemGetHandleForAddressRange_fn_t cuMemGetHandleForAddressRange_fn = NULL;

static char hw_ids[ODL_NCCL_MAX_DEVICES][64];
static int num_devices;

static struct odl_nccl_stats *stats_map;
static int stats_fd = -1;

/* ── Connection handle ───────────────────────────────────────────── */

struct odl_nccl_comm {
	odl_tb5_t handle;
	int refcount;

	/* Memory registration table (v4: lookup by data ptr) */
	struct {
		void *data;
		int  dmabuf_fd;
		size_t size;
		int  type;
	} mr[ODL_NCCL_MAX_MR];
	int mr_count;
	pthread_mutex_t mr_lock;
};

/* ── Memory region handle ────────────────────────────────────────── */

struct odl_nccl_mr {
	int     dmabuf_fd;
	void   *base_addr;
	size_t  size;
	int     type;
};

/* ── Async request tracking ──────────────────────────────────────── */

struct odl_nccl_request {
	bool done;
	int  size;
};

/* ── Listen handle ───────────────────────────────────────────────── */

struct odl_nccl_listen {
	int  dev_id;
	char hw_id[64];
};

/* ════════════════════════════════════════════════════════════════════
 *  CUDA driver dynamic loading
 * ════════════════════════════════════════════════════════════════════ */

static void cuda_init(void)
{
	cuda_lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
	if (!cuda_lib) {
		if (odl_logger)
			odl_logger(0, "ODL_NCCL: libcuda.so.1 not found, "
				   "CUDA memory registration disabled\n");
		return;
	}

	cuMemGetHandleForAddressRange_fn =
		dlsym(cuda_lib, "cuMemGetHandleForAddressRange");
	if (!cuMemGetHandleForAddressRange_fn) {
		if (odl_logger)
			odl_logger(0, "ODL_NCCL: cuMemGetHandleForAddressRange "
				   "not available (CUDA < 11.7?)\n");
		dlclose(cuda_lib);
		cuda_lib = NULL;
		return;
	}

	cuda_available = true;
	if (odl_logger)
		odl_logger(0, "ODL_NCCL: CUDA driver loaded, DMA-buf export "
			   "enabled\n");

	/* Optional nvCOMP path — stub if not built/linked with nvCOMP */
	odl_compress_init();
	if (odl_logger)
		odl_logger(0, "ODL_NCCL: compression %s (threshold=%zu)\n",
			   odl_compress_enabled() ? "ENABLED" : "disabled",
			   odl_compress_threshold());
}

static void cuda_cleanup(void)
{
	if (cuda_lib) {
		dlclose(cuda_lib);
		cuda_lib = NULL;
	}
	cuda_available = false;
	cuMemGetHandleForAddressRange_fn = NULL;
}

static int export_cuda_dmabuf(void *cuda_ptr, size_t size)
{
	int fd;

	if (!cuda_available || !cuMemGetHandleForAddressRange_fn)
		return -1;

	CUresult res = cuMemGetHandleForAddressRange_fn(
		&fd, (CUdeviceptr)cuda_ptr, size,
		CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

	if (res != CUDA_SUCCESS) {
		if (odl_logger)
			odl_logger(1, "ODL_NCCL: cuMemGetHandleForAddressRange "
				   "failed with code %d\n", res);
		return -1;
	}

	return fd;
}

/* ════════════════════════════════════════════════════════════════════
 *  Stats tracking (shared memory)
 * ════════════════════════════════════════════════════════════════════ */

static uint64_t clock_mono_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void stats_init(void)
{
	mkdir(ODL_NCCL_STATS_DIR, 0755);

	stats_fd = open(ODL_NCCL_STATS_PATH, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (stats_fd < 0)
		return;

	if (ftruncate(stats_fd, sizeof(struct odl_nccl_stats)) < 0) {
		close(stats_fd);
		stats_fd = -1;
		return;
	}

	stats_map = mmap(NULL, sizeof(struct odl_nccl_stats),
			 PROT_READ | PROT_WRITE, MAP_SHARED, stats_fd, 0);
	if (stats_map == MAP_FAILED) {
		stats_map = NULL;
		close(stats_fd);
		stats_fd = -1;
		return;
	}

	memset(stats_map, 0, sizeof(*stats_map));
	stats_map->magic = ODL_NCCL_STATS_MAGIC;
	stats_map->version = ODL_NCCL_STATS_VERSION;
	stats_map->start_time_ns = clock_mono_ns();
	__atomic_store_n(&stats_map->active, 1, __ATOMIC_RELEASE);
}

static void stats_cleanup(void)
{
	if (stats_map) {
		__atomic_store_n(&stats_map->active, 0, __ATOMIC_RELEASE);
		munmap(stats_map, sizeof(struct odl_nccl_stats));
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

/* ════════════════════════════════════════════════════════════════════
 *  Memory registration helpers
 * ════════════════════════════════════════════════════════════════════ */

static int mr_lookup_fd(struct odl_nccl_comm *comm, void *data)
{
	pthread_mutex_lock(&comm->mr_lock);
	for (int i = 0; i < comm->mr_count; i++) {
		if (comm->mr[i].data == data) {
			int fd = comm->mr[i].dmabuf_fd;
			pthread_mutex_unlock(&comm->mr_lock);
			return fd;
		}
	}
	pthread_mutex_unlock(&comm->mr_lock);
	return -1;
}

static int mr_add(struct odl_nccl_comm *comm, void *data,
		  int dmabuf_fd, size_t size, int type)
{
	pthread_mutex_lock(&comm->mr_lock);
	if (comm->mr_count >= ODL_NCCL_MAX_MR) {
		pthread_mutex_unlock(&comm->mr_lock);
		return -1;
	}
	int idx = comm->mr_count++;
	comm->mr[idx].data      = data;
	comm->mr[idx].dmabuf_fd = dmabuf_fd;
	comm->mr[idx].size      = size;
	comm->mr[idx].type      = type;
	pthread_mutex_unlock(&comm->mr_lock);
	return idx;
}

static void mr_remove_by_fd(struct odl_nccl_comm *comm, int dmabuf_fd)
{
	pthread_mutex_lock(&comm->mr_lock);
	for (int i = 0; i < comm->mr_count; i++) {
		if (comm->mr[i].dmabuf_fd == dmabuf_fd) {
			if (comm->mr[i].dmabuf_fd >= 0)
				close(comm->mr[i].dmabuf_fd);
			comm->mr[i] = comm->mr[--comm->mr_count];
			pthread_mutex_unlock(&comm->mr_lock);
			return;
		}
	}
	pthread_mutex_unlock(&comm->mr_lock);
}

/* ════════════════════════════════════════════════════════════════════
 *  Plugin entry points — v4 + v5 common
 * ════════════════════════════════════════════════════════════════════ */

static ncclResult_t nccl_odl_init(ncclDebugLogger_t logFunction)
{
	odl_logger = logFunction;
	cuda_init();
	stats_init();
	atexit(stats_cleanup);
	return ncclSuccess;
}

static ncclResult_t nccl_odl_devices(int *ndev)
{
	DIR *dir;
	struct dirent *entry;
	char path[256], uuid_buf[64];
	FILE *fp;

	num_devices = 0;
	dir = opendir("/sys/bus/thunderbolt/devices");
	if (!dir) {
		*ndev = 0;
		return ncclSuccess;
	}

	while ((entry = readdir(dir)) != NULL &&
	       num_devices < ODL_NCCL_MAX_DEVICES) {
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
	return ncclSuccess;
}

/* ── getProperties (v5) ───────────────────────────────────────────── */

static ncclResult_t nccl_odl_getProperties_v5(int dev,
		ncclNetProperties_v5_t *props)
{
	if (dev < 0 || dev >= num_devices)
		return ncclInvalidArgument;

	memset(props, 0, sizeof(*props));
	snprintf(props->name, sizeof(props->name),
		 "OdinLink TB5 DMA Ring");
	snprintf(props->pciPath, sizeof(props->pciPath),
		 "/sys/bus/thunderbolt");
	props->guid = (uint64_t)dev;
	props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
	props->speed = 80000;
	props->port = dev;
	props->maxComm = ODL_NCCL_MAX_COMMS;
	props->maxRecvs = ODL_NCCL_MAX_COMMS;

	return ncclSuccess;
}

/* ── listen ────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_listen(int dev, void *handle,
				    void **listenComm)
{
	struct odl_nccl_listen *lh;

	if (dev < 0 || dev >= num_devices)
		return ncclInvalidArgument;

	lh = calloc(1, sizeof(*lh));
	if (!lh)
		return ncclSystemError;

	lh->dev_id = dev;
	snprintf(lh->hw_id, sizeof(lh->hw_id), "%s", hw_ids[dev]);

	if (handle)
		memcpy(handle, lh->hw_id, sizeof(lh->hw_id));

	*listenComm = lh;
	return ncclSuccess;
}

/* ── connect ───────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_connect(int dev, void *handle,
				     void **sendComm, void **recvComm)
{
	struct odl_nccl_comm *comm;
	int ret;

	(void)handle;

	comm = calloc(1, sizeof(*comm));
	if (!comm)
		return ncclSystemError;

	pthread_mutex_init(&comm->mr_lock, NULL);

	ret = odl_tb5_open(&comm->handle, dev);
	if (ret < 0) {
		free(comm);
		return ncclSystemError;
	}

	ret = odl_tb5_wait_peer(comm->handle, 5000);
	if (ret < 0) {
		odl_tb5_close(comm->handle);
		free(comm);
		return ncclSystemError;
	}

	comm->refcount = 2;
	*sendComm = comm;
	*recvComm = comm;
	return ncclSuccess;
}

/* ── accept ────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_accept(void *listenComm,
				    void **recvComm, void **sendComm)
{
	struct odl_nccl_listen *lh = listenComm;
	struct odl_nccl_comm *comm;
	int ret;

	comm = calloc(1, sizeof(*comm));
	if (!comm)
		return ncclSystemError;

	pthread_mutex_init(&comm->mr_lock, NULL);

	ret = odl_tb5_open(&comm->handle, lh->dev_id);
	if (ret < 0) {
		free(comm);
		return ncclSystemError;
	}

	ret = odl_tb5_wait_peer(comm->handle, 5000);
	if (ret < 0) {
		odl_tb5_close(comm->handle);
		free(comm);
		return ncclSystemError;
	}

	comm->refcount = 2;
	*sendComm = comm;
	*recvComm = comm;
	return ncclSuccess;
}

/* ── regMr ─────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_regMr(void *comm_ptr, void *data,
				   size_t size, int type,
				   void **mhandle)
{
	struct odl_nccl_comm *comm = comm_ptr;
	struct odl_nccl_mr *mr;
	int fd = -1;

	if (type == NCCL_PTR_CUDA && cuda_available) {
		fd = export_cuda_dmabuf(data, size);
		if (fd < 0) {
			if (odl_logger)
				odl_logger(0, "ODL_NCCL: failed to export "
					   "CUDA memory as DMA-buf FD\n");
			return ncclSystemError;
		}
	}

	mr = calloc(1, sizeof(*mr));
	if (!mr) {
		if (fd >= 0) close(fd);
		return ncclSystemError;
	}

	mr->dmabuf_fd = fd;
	mr->base_addr = data;
	mr->size      = size;
	mr->type      = type;

	/* Store in comm lookup table (for v4 compatibility) */
	mr_add(comm, data, fd, size, type);

	*mhandle = mr;

	if (odl_logger)
		odl_logger(0, "ODL_NCCL: regMr ptr=%p size=%zu type=%s "
			   "fd=%d\n",
			   data, size,
			   type == NCCL_PTR_CUDA ? "CUDA" : "HOST",
			   fd);

	return ncclSuccess;
}

/* ── deregMr ───────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_deregMr(void *comm_ptr, void *mhandle)
{
	struct odl_nccl_mr *mr = mhandle;

	if (!mr)
		return ncclSuccess;

	if (mr->dmabuf_fd >= 0) {
		mr_remove_by_fd(comm_ptr, mr->dmabuf_fd);
		close(mr->dmabuf_fd);
	}

	free(mr);
	return ncclSuccess;
}

/* ── getMr (v5 only) ──────────────────────────────────────────────── */

static ncclResult_t nccl_odl_getMr(void *comm_ptr, void *data,
				   size_t size, void *mhandle,
				   void **mr_out)
{
	/* Reuse existing mhandle or create a temporary one */
	struct odl_nccl_mr *existing = mhandle;
	if (existing) {
		*mr_out = existing;
		return ncclSuccess;
	}

	/* Look up from comm table by data pointer */
	struct odl_nccl_comm *comm = comm_ptr;
	int fd = mr_lookup_fd(comm, data);
	if (fd >= 0) {
		struct odl_nccl_mr *mr = calloc(1, sizeof(*mr));
		if (!mr) return ncclSystemError;
		mr->dmabuf_fd = fd;
		mr->base_addr = data;
		mr->size = size;
		mr->type = NCCL_PTR_CUDA;
		*mr_out = mr;
		return ncclSuccess;
	}

	return ncclInvalidArgument;
}

/* ── optional compress (nvCOMP) then isend ────────────────────────── */

static int try_compress_send(struct odl_nccl_comm *comm, void *data, int size)
{
	/* GPU path only; falls back if nvCOMP missing / ODL_COMPRESS off / small msg. */
	int fd_probe = export_cuda_dmabuf(data, (size_t)size);
	int is_cuda = (fd_probe >= 0);
	if (fd_probe >= 0)
		close(fd_probe);
	if (!odl_compress_should((size_t)size, is_cuda))
		return -1;

	size_t wire_cap = odl_compress_max_wire_bytes((size_t)size);
	void *d_wire = odl_compress_device_alloc(wire_cap);
	if (!d_wire)
		return -1;

	size_t wire_bytes = 0;
	if (odl_compress_gpu(data, (size_t)size, d_wire, wire_cap,
			     &wire_bytes, NULL) != 0) {
		odl_compress_device_free(d_wire);
		return -1;
	}

	/* Transport fixed max size so recv can match without a size exchange.
	 * Actual compressed length is in the header; trailing bytes ignored. */
	size_t send_bytes = wire_cap;
	int fd = export_cuda_dmabuf(d_wire, send_bytes);
	int ret = -1;
	if (fd >= 0) {
		ret = odl_tb5_send_dmabuf(comm->handle, fd, 0, (int)send_bytes);
		close(fd);
	}
	odl_compress_device_free(d_wire);
	if (ret == 0 && odl_logger)
		odl_logger(0, "ODL_NCCL: compress send %d -> wire_cap=%zu actual=%zu\n",
			   size, send_bytes, wire_bytes);
	return ret;
}

/* ── isend ─────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_isend_v5(void *sendComm, void *data,
				      int size, int tag,
				      void *mhandle,
				      void **request)
{
	struct odl_nccl_comm *comm = sendComm;
	struct odl_nccl_request *req;
	int ret;

	(void)tag;

	req = calloc(1, sizeof(*req));
	if (!req)
		return ncclSystemError;

	/* Optional compress path (no-op when stub / env off) */
	if (try_compress_send(comm, data, size) == 0) {
		stats_record_tx(size);
		req->done = true;
		req->size = size;
		*request = req;
		return ncclSuccess;
	}

	if (mhandle) {
		struct odl_nccl_mr *mr = mhandle;
		if (mr->dmabuf_fd >= 0) {
			/* Zero-copy DMA-buf path (GPU memory) */
			ret = odl_tb5_send_dmabuf(comm->handle,
						  mr->dmabuf_fd,
						  0, size);
		} else {
			/* Host memory: copy through mmap'd buffer */
			size_t buf_size;
			void *tx_buf = odl_tb5_tx_buffer(comm->handle,
							  &buf_size);
			if (!tx_buf || (size_t)size > buf_size) {
				free(req);
				return ncclInvalidUsage;
			}
			memcpy(tx_buf, data, size);
			ret = odl_tb5_send(comm->handle, 0, size);
		}
	} else {
		/* No mhandle: try on-the-fly export */
		int fd = export_cuda_dmabuf(data, size);
		if (fd >= 0) {
			ret = odl_tb5_send_dmabuf(comm->handle, fd, 0, size);
			close(fd);
		} else {
			size_t buf_size;
			void *tx_buf = odl_tb5_tx_buffer(comm->handle,
							  &buf_size);
			if (!tx_buf || (size_t)size > buf_size) {
				free(req);
				return ncclInvalidUsage;
			}
			memcpy(tx_buf, data, size);
			ret = odl_tb5_send(comm->handle, 0, size);
		}
	}

	if (ret < 0) {
		free(req);
		return ncclSystemError;
	}

	stats_record_tx(size);

	req->done = true;
	req->size = size;
	*request = req;
	return ncclSuccess;
}

/* ── isend v4 (no mhandle, lookup by data ptr) ────────────────────── */

static ncclResult_t nccl_odl_isend_v4(void *sendComm, void *data,
				      int size, int tag,
				      void **request)
{
	struct odl_nccl_comm *comm = sendComm;
	int fd = mr_lookup_fd(comm, data);

	if (fd >= 0) {
		return nccl_odl_isend_v5(sendComm, data, size, tag,
					 &(struct odl_nccl_mr){
						.dmabuf_fd = fd,
						.base_addr = data,
						.size = size,
						.type = NCCL_PTR_CUDA,
					 }, request);
	}

	return nccl_odl_isend_v5(sendComm, data, size, tag, NULL, request);
}

/* ── compressed irecv: land max-wire staging, decompress into user buf ─ */

static int try_compress_recv(struct odl_nccl_comm *comm, void *data, int size)
{
	if (!odl_compress_should((size_t)size, 1))
		return -1;

	size_t wire_cap = odl_compress_max_wire_bytes((size_t)size);
	void *d_wire = odl_compress_device_alloc(wire_cap);
	if (!d_wire)
		return -1;

	int fd = export_cuda_dmabuf(d_wire, wire_cap);
	int ret = -1;
	if (fd >= 0) {
		ret = odl_tb5_recv_dmabuf(comm->handle, fd, 0, (int)wire_cap);
		close(fd);
	}
	if (ret == 0) {
		size_t orig = 0;
		if (odl_decompress_gpu(d_wire, wire_cap, data, (size_t)size,
				       &orig, NULL) != 0) {
			/* Peer sent raw despite our expectation — fall through. */
			ret = -1;
		}
	}
	odl_compress_device_free(d_wire);
	return ret;
}

/* ── irecv ─────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_irecv_v5(void *recvComm, void *data,
				      int size, int tag,
				      void *mhandle,
				      void **request)
{
	struct odl_nccl_comm *comm = recvComm;
	struct odl_nccl_request *req;
	int ret;

	(void)tag;

	req = calloc(1, sizeof(*req));
	if (!req)
		return ncclSystemError;

	/* Optional compress path — only if we would also compress on send */
	if (try_compress_recv(comm, data, size) == 0) {
		stats_record_rx(size);
		req->done = true;
		req->size = size;
		*request = req;
		return ncclSuccess;
	}

	if (mhandle) {
		struct odl_nccl_mr *mr = mhandle;
		if (mr->dmabuf_fd >= 0) {
			/* Zero-copy DMA-buf path (GPU memory) */
			ret = odl_tb5_recv_dmabuf(comm->handle,
						  mr->dmabuf_fd,
						  0, size);
		} else {
			/* Host memory: recv through mmap'd buffer, copy out */
			ret = odl_tb5_recv(comm->handle, 0, size);
			if (ret == 0) {
				size_t buf_size;
				void *rx_buf = odl_tb5_rx_buffer(comm->handle,
								 &buf_size);
				if (rx_buf && (size_t)size <= buf_size)
					memcpy(data, rx_buf, size);
			}
		}
	} else {
		int fd = export_cuda_dmabuf(data, size);
		if (fd >= 0) {
			ret = odl_tb5_recv_dmabuf(comm->handle, fd, 0, size);
			close(fd);
		} else {
			ret = odl_tb5_recv(comm->handle, 0, size);
			if (ret == 0) {
				size_t buf_size;
				void *rx_buf = odl_tb5_rx_buffer(comm->handle,
								 &buf_size);
				if (rx_buf && (size_t)size <= buf_size)
					memcpy(data, rx_buf, size);
			}
		}
	}

	if (ret < 0) {
		free(req);
		return ncclSystemError;
	}

	stats_record_rx(size);

	req->done = true;
	req->size = size;
	*request = req;
	return ncclSuccess;
}

/* ── irecv v4 (multi-buffer) ──────────────────────────────────────── */

static ncclResult_t nccl_odl_irecv_v4(void *recvComm, int n,
				      void **data, int *sizes,
				      int *tags, void ***request)
{
	/* For multi-recv, handle each buffer individually.
	 * In practice NCCL usually passes n=1. */
	void **reqs = calloc((size_t)n, sizeof(void *));
	if (!reqs)
		return ncclSystemError;

	for (int i = 0; i < n; i++) {
		ncclResult_t res = nccl_odl_irecv_v5(
			recvComm, data[i], sizes[i],
			tags ? tags[i] : 0,
			NULL, &reqs[i]);
		if (res != ncclSuccess) {
			for (int j = 0; j < i; j++)
				free(reqs[j]);
			free(reqs);
			return res;
		}
	}

	*request = reqs;
	return ncclSuccess;
}

/* ── iflush ────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_iflush_v5(void *recvComm, void *data,
				       int size, void *mhandle,
				       void **request)
{
	(void)recvComm;
	(void)data;
	(void)size;
	(void)mhandle;

	*request = NULL;
	return ncclSuccess;
}

static ncclResult_t nccl_odl_iflush_v4(void *recvComm, int n,
				       void **data, int *sizes,
				       void ***request)
{
	(void)recvComm;
	(void)n;
	(void)data;
	(void)sizes;

	*request = NULL;
	return ncclSuccess;
}

/* ── test ──────────────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_test(void *request, int *done, int *size)
{
	struct odl_nccl_request *req = request;

	if (!req) {
		*done = 1;
		return ncclSuccess;
	}

	*done = req->done ? 1 : 0;
	if (size)
		*size = req->size;

	free(req);
	return ncclSuccess;
}

/* ── closeSend / closeRecv ─────────────────────────────────────────── */

static ncclResult_t nccl_odl_closeSend(void *sendComm)
{
	struct odl_nccl_comm *comm = sendComm;

	if (!comm)
		return ncclSuccess;

	if (__atomic_sub_fetch(&comm->refcount, 1, __ATOMIC_SEQ_CST) == 0) {
		odl_tb5_close(comm->handle);
		pthread_mutex_destroy(&comm->mr_lock);
		free(comm);
	}

	return ncclSuccess;
}

static ncclResult_t nccl_odl_closeRecv(void *recvComm)
{
	return nccl_odl_closeSend(recvComm);
}

/* ── closeListen ───────────────────────────────────────────────────── */

static ncclResult_t nccl_odl_closeListen(void *listenComm)
{
	free(listenComm);
	return ncclSuccess;
}

/* ════════════════════════════════════════════════════════════════════
 *  v4 → v5 wrapper functions (ABI-safe)
 *
 *  v4 and v5 APIs differ in several function signatures (int vs
 *  size_t, different struct sizes).  These wrappers adapt the v4
 *  calling convention to the internal v5 implementations so the
 *  function-pointer casts below remain valid and don't cause ABI
 *  mismatches on x86-64 (where incorrect upper register bits would
 *  corrupt parameters).
 * ════════════════════════════════════════════════════════════════════ */

static ncclResult_t nccl_odl_getProperties_v4(int dev,
		ncclNetProperties_v4_t *props_v4)
{
	ncclNetProperties_v5_t props_v5;
	ncclResult_t ret;

	ret = nccl_odl_getProperties_v5(dev, &props_v5);
	if (ret != ncclSuccess)
		return ret;

	memcpy(props_v4, &props_v5, sizeof(*props_v4));
	return ncclSuccess;
}

static ncclResult_t nccl_odl_regMr_v4(void *comm, void *data,
				      int size, int type,
				      void **mhandle)
{
	return nccl_odl_regMr(comm, data, (size_t)size, type, mhandle);
}

/* ════════════════════════════════════════════════════════════════════
 *  Plugin struct exports
 * ════════════════════════════════════════════════════════════════════ */

ncclNet_v4_t ncclNetPlugin_v4 = {
	.name        = "ODL_TB5",
	.init        = nccl_odl_init,
	.devices     = nccl_odl_devices,
	.getProperties = nccl_odl_getProperties_v4,
	.listen      = nccl_odl_listen,
	.connect     = nccl_odl_connect,
	.accept      = nccl_odl_accept,
	.regMr       = nccl_odl_regMr_v4,
	.deregMr     = nccl_odl_deregMr,
	.isend       = nccl_odl_isend_v4,
	.irecv       = nccl_odl_irecv_v4,
	.iflush      = nccl_odl_iflush_v4,
	.test        = nccl_odl_test,
	.closeSend   = nccl_odl_closeSend,
	.closeRecv   = nccl_odl_closeRecv,
	.closeListen = nccl_odl_closeListen,
};

ncclNet_v5_t ncclNetPlugin_v5 = {
	.name        = "ODL_TB5",
	.init        = nccl_odl_init,
	.devices     = nccl_odl_devices,
	.getProperties = nccl_odl_getProperties_v5,
	.listen      = nccl_odl_listen,
	.connect     = nccl_odl_connect,
	.accept      = nccl_odl_accept,
	.regMr       = nccl_odl_regMr,
	.deregMr     = nccl_odl_deregMr,
	.isend       = nccl_odl_isend_v5,
	.irecv       = nccl_odl_irecv_v5,
	.iflush      = nccl_odl_iflush_v5,
	.test        = nccl_odl_test,
	.closeSend   = nccl_odl_closeSend,
	.closeRecv   = nccl_odl_closeRecv,
	.closeListen = nccl_odl_closeListen,
	.getMr       = nccl_odl_getMr,
};
