/*
 * OdinLink Thunderbolt 5 - RCCL Net v7 Plugin
 *
 * Implements the RCCL/NCCL network plugin interface (ncclNet_v7) using the
 * OdinLink TB5 driver's stream API.
 *
 * Design:
 *  - Advertises NCCL_PTR_HOST only, so RCCL stages GPU<->host itself and
 *    hands us host pointers (no HIP dependency here).
 *  - One shared, ref-counted device handle per process; each NCCL
 *    connection is a stream multiplexed over the single TB point-to-point
 *    link (rather than opening the device N times).
 *  - odl_tb5_stream_send/recv are blocking, so isend/irecv run the actual
 *    transfer on a detached background thread and return an immediately-
 *    pollable request; test() reports the done flag.  This preserves the
 *    non-blocking semantics RCCL's proxy progress loop requires.
 *
 * Exposes shared-memory stats at /run/odl_tb5/rccl_stats.
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
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "net_v7.h"
#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_rccl_stats.h>

#define ODL_TB5_MAX_RCCL_DEVICES 16

/* ── Opt-in debug tracing (ODL_DEBUG=1 request-level, =2 also per-chunk) ────
 * Writes one file per process: /tmp/odl_plugin.<pid>.log (line-buffered).
 * Purpose: see whether RCCL actually posts a collective to this net plugin,
 * and if so which isend/irecv on which stream blocks. */
static int   odl_dbg;
static FILE *odl_dbg_fp;
static void odl_dbg_init(void)
{
	const char *e = getenv("ODL_DEBUG");
	odl_dbg = e ? atoi(e) : 0;
	if (!odl_dbg)
		return;
	char path[128];
	snprintf(path, sizeof(path), "/tmp/odl_plugin.%d.log", (int)getpid());
	odl_dbg_fp = fopen(path, "w");
	if (!odl_dbg_fp)
		odl_dbg_fp = stderr;
	setvbuf(odl_dbg_fp, NULL, _IOLBF, 0);
}
#define DBG(lvl, fmt, ...) do { \
	if (odl_dbg >= (lvl) && odl_dbg_fp) { \
		struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); \
		fprintf(odl_dbg_fp, "[%ld.%03ld t%04lx] " fmt "\n", \
			(long)_ts.tv_sec, _ts.tv_nsec / 1000000, \
			(unsigned long)pthread_self() & 0xffff, ##__VA_ARGS__); \
	} } while (0)

/* Protocol-level faults must be visible regardless of debug level: silently
 * dropping them is what let the stream-desync bug corrupt whole runs. */
#define WARN(fmt, ...) do { \
	FILE *_f = odl_dbg_fp ? odl_dbg_fp : stderr; \
	fprintf(_f, "odl_tb5 WARN: " fmt "\n", ##__VA_ARGS__); \
} while (0)

static rcclDebugLogger_t odl_logger;
static char hw_ids[ODL_TB5_MAX_RCCL_DEVICES][64];
static int num_devices;

static struct odl_rccl_stats *stats_map;
static int stats_fd = -1;

/* ── Shared device handle (one TB link, many streams) ──────────────── */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static odl_tb5_t       g_handle;            /* opened lazily */
static int             g_handle_refs;
static uint8_t         g_next_cid = 1;      /* connection/stream id alloc */

struct odl_tb5_request;

/* Per-connection communicator.  A single worker thread drains a FIFO queue
 * of requests so transfers on this stream happen strictly in the order RCCL
 * posted them (RCCL matches sends/recvs in order per connection) and never
 * concurrently. */
struct odl_tb5_comm {
	odl_tb5_t handle;    /* shared g_handle */
	uint8_t   stream_id; /* local stream to send-from / recv-on */
	uint8_t   dst_id;    /* remote stream to deliver to (send side) */
	int       is_send;

	pthread_t       worker;
	pthread_mutex_t q_lock;
	pthread_cond_t  q_cond;
	struct odl_tb5_request *q_head, *q_tail;
	int             stop;
	int             worker_started;
};

/* Async request: queued to its comm's worker thread. */
struct odl_tb5_request {
	struct odl_tb5_comm *comm;
	void   *data;
	int     size;         /* requested size */
	int     done_size;    /* actual transferred size */
	int     is_send;
	volatile int done;    /* set by worker thread */
	volatile int failed;
	struct odl_tb5_request *next;   /* queue linkage */
};

/* Listen handle: owns a pre-opened recv stream whose (auto-assigned) id is
 * advertised to the connecting peer so it can target it as dst. */
struct odl_tb5_listen_handle {
	int       dev_id;
	odl_tb5_t handle;     /* shared handle ref held until accept/closeListen */
	uint8_t   stream_id;  /* auto-assigned recv stream */
	int       accepted;   /* ref transferred to recvComm */
	char      hw_id[64];
};

/* Memory registration handle (host staging: nothing to pin). */
struct odl_tb5_mr {
	void  *data;
	size_t size;
};

static int comm_start_worker(struct odl_tb5_comm *comm);
static void comm_stop_worker(struct odl_tb5_comm *comm);

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

/* Open (or reuse) the shared device handle and wait for the peer link. */
static rcclResult_t get_shared_handle(int dev, odl_tb5_t *out)
{
	rcclResult_t res = rcclSuccess;

	pthread_mutex_lock(&g_lock);
	if (!g_handle) {
		if (odl_tb5_open(&g_handle, dev) < 0) {
			g_handle = NULL;
			res = rcclSystemError;
			goto out;
		}
		if (odl_tb5_wait_peer(g_handle, 10000) < 0) {
			odl_tb5_close(g_handle);
			g_handle = NULL;
			res = rcclSystemError;
			goto out;
		}
	}
	g_handle_refs++;
	*out = g_handle;
out:
	pthread_mutex_unlock(&g_lock);
	return res;
}

static void put_shared_handle(void)
{
	pthread_mutex_lock(&g_lock);
	if (g_handle && --g_handle_refs == 0) {
		odl_tb5_close(g_handle);
		g_handle = NULL;
	}
	pthread_mutex_unlock(&g_lock);
}

static rcclResult_t odl_tb5_init(rcclDebugLogger_t logFunction)
{
	odl_logger = logFunction;
	odl_dbg_init();
	DBG(1, "init: plugin loaded pid=%d", (int)getpid());
	stats_init();
	atexit(stats_cleanup);
	return rcclSuccess;
}

static rcclResult_t odl_tb5_devices(int *ndev)
{
	/* Single point-to-point TB link => one usable net device. */
	num_devices = 1;
	snprintf(hw_ids[0], sizeof(hw_ids[0]), "odl_tb5_0");
	*ndev = 1;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_getProperties(int dev, rcclNetProperties_v7_t *props)
{
	if (dev < 0 || dev >= num_devices)
		return rcclInvalidArgument;

	memset(props, 0, sizeof(*props));
	props->name = (char *)"OdinLink-TB5";
	props->pciPath = (char *)"/sys/bus/thunderbolt";
	props->guid = (uint64_t)dev;
	props->ptrSupport = NCCL_PTR_HOST;   /* RCCL stages GPU<->host itself */
	props->speed = 80000;
	props->port = dev;
	props->latency = 0.0f;
	props->maxComm = 0x7fffffff;
	props->maxRecvs = 1;
	props->netDeviceType = 0;            /* NCCL_NET_DEVICE_HOST */
	props->netDeviceVersion = 0;
	DBG(1, "getProperties dev=%d name=%s ptrSupport=HOST", dev, props->name);
	return rcclSuccess;
}

static rcclResult_t odl_tb5_listen(int dev, void *handle, void **listenComm)
{
	struct odl_tb5_listen_handle *lh;

	if (dev < 0 || dev >= num_devices)
		return rcclInvalidArgument;

	lh = calloc(1, sizeof(*lh));
	if (!lh)
		return rcclSystemError;

	lh->dev_id = dev;
	snprintf(lh->hw_id, sizeof(lh->hw_id), "%s", hw_ids[dev]);

	/* Open the recv stream now (auto id, collision-free) and advertise its
	 * real id so the peer's connect() can target it as dst. */
	if (get_shared_handle(dev, &lh->handle) != rcclSuccess) {
		free(lh);
		return rcclSystemError;
	}
	if (odl_tb5_stream_open(lh->handle, 0, &lh->stream_id) < 0) {
		put_shared_handle();
		free(lh);
		return rcclSystemError;
	}

	if (handle) {
		memset(handle, 0, 64);
		((uint8_t *)handle)[0] = lh->stream_id;
		memcpy((uint8_t *)handle + 1, lh->hw_id, sizeof(lh->hw_id));
	}

	DBG(1, "listen  dev=%d -> recv stream_id=%u (lh=%p)", dev, lh->stream_id, (void *)lh);
	*listenComm = lh;
	return rcclSuccess;
}

/* Sender side. RCCL v7 passes sendDevComm (device-side handle) which this
 * CPU-staged plugin does not use — set it to NULL. */
static rcclResult_t odl_tb5_connect(int dev, void *handle, void **sendComm,
				    rcclNetDeviceHandle_v7_t **sendDevComm)
{
	struct odl_tb5_comm *comm;
	uint8_t peer_cid = ((uint8_t *)handle)[0];
	uint8_t sid = 0;

	if (sendDevComm)
		*sendDevComm = NULL;

	/*
	 * The v7 contract: when the connection cannot be made *yet*, return
	 * rcclSuccess with *sendComm left NULL and RCCL will call again. Only a
	 * genuinely fatal condition returns an error.
	 *
	 * Returning rcclSystemError for a transient case aborts ncclCommInitRank
	 * outright. That is not theoretical - it is exactly what happened here:
	 * the peer rpc-server started before the link reached READY, this
	 * returned an error, and the whole world init failed with "world
	 * communicator init failed; collectives will fall back to lazy init".
	 */
	*sendComm = NULL;

	comm = calloc(1, sizeof(*comm));
	if (!comm)
		return rcclSystemError;   /* out of memory: genuinely fatal */

	if (get_shared_handle(dev, &comm->handle) != rcclSuccess) {
		/* Device not open or link not READY yet - transient. */
		free(comm);
		DBG(1, "connect dev=%d: device not ready, asking RCCL to retry", dev);
		return rcclSuccess;
	}

	/* Local send stream (auto-assigned); target the peer's advertised id. */
	if (odl_tb5_stream_open(comm->handle, 0, &sid) < 0) {
		/* No stream available yet - transient. */
		put_shared_handle();
		free(comm);
		DBG(1, "connect dev=%d: no stream available, asking RCCL to retry", dev);
		return rcclSuccess;
	}
	comm->stream_id = sid;
	comm->dst_id = peer_cid;
	comm->is_send = 1;

	if (comm_start_worker(comm) < 0) {
		odl_tb5_stream_close(comm->handle, comm->stream_id);
		put_shared_handle();
		free(comm);
		return rcclSystemError;
	}

	DBG(1, "connect dev=%d send stream_id=%u -> dst=%u (comm=%p)", dev, sid, peer_cid, (void *)comm);
	*sendComm = comm;
	return rcclSuccess;
}

/* Receiver side. */
static rcclResult_t odl_tb5_accept(void *listenComm, void **recvComm,
				   rcclNetDeviceHandle_v7_t **recvDevComm)
{
	struct odl_tb5_listen_handle *lh = listenComm;
	struct odl_tb5_comm *comm;

	if (recvDevComm)
		*recvDevComm = NULL;

	/* Same retry contract as connect(): NULL comm + rcclSuccess means
	 * "not yet, call me again", not "failed". */
	*recvComm = NULL;

	comm = calloc(1, sizeof(*comm));
	if (!comm)
		return rcclSystemError;   /* out of memory: genuinely fatal */

	/* Reuse the recv stream + handle ref already opened in listen(). */
	comm->handle = lh->handle;
	comm->stream_id = lh->stream_id;
	comm->dst_id = 0;
	comm->is_send = 0;
	lh->accepted = 1;   /* ref now owned by recvComm */

	if (comm_start_worker(comm) < 0) {
		odl_tb5_stream_close(comm->handle, comm->stream_id);
		put_shared_handle();
		free(comm);
		return rcclSystemError;
	}

	DBG(1, "accept  recv stream_id=%u (lh=%p comm=%p)", comm->stream_id, listenComm, (void *)comm);
	*recvComm = comm;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_closeListen(void *listenComm)
{
	struct odl_tb5_listen_handle *lh = listenComm;

	if (lh && !lh->accepted && lh->handle) {
		/* accept() never took ownership — release the recv stream + ref. */
		odl_tb5_stream_close(lh->handle, lh->stream_id);
		put_shared_handle();
	}
	free(lh);
	return rcclSuccess;
}

/* ── Memory registration (host staging: record only) ───────────────── */
static rcclResult_t odl_tb5_regMr(void *comm, void *data, int size, int type,
				  void **mhandle)
{
	struct odl_tb5_mr *mr = calloc(1, sizeof(*mr));
	(void)comm; (void)type;
	if (!mr)
		return rcclSystemError;
	mr->data = data;
	mr->size = (size_t)size;
	*mhandle = mr;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_regMrDmaBuf(void *comm, void *data, size_t size,
					int type, uint64_t offset, int fd,
					void **mhandle)
{
	(void)offset; (void)fd;
	return odl_tb5_regMr(comm, data, (int)size, type, mhandle);
}

static rcclResult_t odl_tb5_deregMr(void *comm, void *mhandle)
{
	(void)comm;
	free(mhandle);
	return rcclSuccess;
}

/* Max payload that fits in a single TB frame (frame 4096 - 5B stream hdr).
 * The kernel's multi-frame reassembly is unreliable, but single-frame
 * messages are byte-exact, so we chunk every transfer to <= this and send
 * each chunk as its own atomic single-frame message.  The per-connection
 * FIFO worker + single stream keep chunks strictly ordered on both ends. */
#define ODL_CHUNK 4000

/* ── Per-connection FIFO worker ────────────────────────────────────── */
static void do_transfer(struct odl_tb5_comm *comm, struct odl_tb5_request *req)
{
	uint8_t src_id = 0;
	uint32_t actual = 0;
	int ret;
	int off = 0;

	DBG(1, "xfer  START %s comm=%p sid=%u dst=%u size=%d",
	    req->is_send ? "SEND" : "RECV", (void *)comm, comm->stream_id, comm->dst_id, req->size);
	if (req->is_send) {
		/* 4-byte length header (own single frame) tells the receiver
		 * the exact byte count, then the payload in <=1-frame chunks. */
		uint32_t hdr = (uint32_t)req->size;
		DBG(2, "  send hdr sid=%u dst=%u", comm->stream_id, comm->dst_id);
		ret = odl_tb5_stream_send(comm->handle, comm->stream_id,
					  comm->dst_id, &hdr, sizeof(hdr));
		while (ret >= 0 && off < req->size) {
			int n = req->size - off;
			if (n > ODL_CHUNK)
				n = ODL_CHUNK;
			DBG(2, "  send chunk sid=%u off=%d n=%d", comm->stream_id, off, n);
			ret = odl_tb5_stream_send(comm->handle, comm->stream_id,
						  comm->dst_id,
						  (char *)req->data + off,
						  (uint32_t)n);
			off += n;
		}
		if (ret < 0)
			req->failed = 1;
		else
			stats_record_tx(req->size);
		req->done_size = req->size;
	} else {
		uint32_t total = 0;
		DBG(2, "  recv hdr sid=%u", comm->stream_id);
		ret = odl_tb5_stream_recv(comm->handle, comm->stream_id,
					  &total, sizeof(total), &src_id, &actual);
		if (ret < 0 || actual != sizeof(total)) {
			req->failed = 1;
		} else {
			int overflow = total > (uint32_t)req->size;

			/*
			 * If the sender advertises more than the posted receive
			 * can hold, consume every complete chunk into a scratch
			 * buffer. Do not read a partial chunk into the caller's
			 * buffer: stream_recv discards the uncopied part of that
			 * individual message, so accounting it as still queued
			 * would make the drain wait forever.
			 *
			 * The request still fails, but consuming all complete
			 * chunks leaves the next length header aligned.
			 */
			char sink[ODL_CHUNK];

			while (off < (int)total) {
				int n = (int)total - off;

				if (n > ODL_CHUNK)
					n = ODL_CHUNK;
				DBG(2, "  recv chunk sid=%u off=%d n=%d", comm->stream_id, off, n);
				ret = odl_tb5_stream_recv(comm->handle,
							  comm->stream_id,
							  overflow
							  ? sink
							  : (char *)req->data + off,
							  (uint32_t)n, &src_id,
							  &actual);
				if (ret < 0) {
					req->failed = 1;
					break;
				}
				off += (int)actual;
				if (actual == 0)
					break;
			}
			if (overflow) {
				WARN("recv overflow on sid=%u: advertised=%u posted=%d, drained=%u - failing request%s",
				     comm->stream_id, total, req->size,
				     (unsigned int)off,
				     off == (int)total
				     ? " after restoring framing"
				     : " with framing not restored");
				req->failed = 1;
			}

			if (!req->failed)
				stats_record_rx(off);
		}
		req->done_size = off;
	}
	DBG(1, "xfer  %s   %s comm=%p sid=%u size=%d ret=%d off=%d",
	    req->failed ? "FAIL" : "DONE", req->is_send ? "SEND" : "RECV",
	    (void *)comm, comm->stream_id, req->size, ret, off);
	__atomic_store_n(&req->done, 1, __ATOMIC_RELEASE);
}

static void *comm_worker(void *arg)
{
	struct odl_tb5_comm *comm = arg;

	pthread_mutex_lock(&comm->q_lock);
	for (;;) {
		while (!comm->q_head && !comm->stop)
			pthread_cond_wait(&comm->q_cond, &comm->q_lock);
		if (!comm->q_head && comm->stop)
			break;
		struct odl_tb5_request *req = comm->q_head;
		comm->q_head = req->next;
		if (!comm->q_head)
			comm->q_tail = NULL;
		pthread_mutex_unlock(&comm->q_lock);

		do_transfer(comm, req);        /* blocking, in FIFO order */

		pthread_mutex_lock(&comm->q_lock);
	}
	pthread_mutex_unlock(&comm->q_lock);
	return NULL;
}

static int comm_start_worker(struct odl_tb5_comm *comm)
{
	pthread_mutex_init(&comm->q_lock, NULL);
	pthread_cond_init(&comm->q_cond, NULL);
	comm->q_head = comm->q_tail = NULL;
	comm->stop = 0;
	if (pthread_create(&comm->worker, NULL, comm_worker, comm) != 0)
		return -1;
	comm->worker_started = 1;
	return 0;
}

static void comm_stop_worker(struct odl_tb5_comm *comm)
{
	if (!comm->worker_started)
		return;
	pthread_mutex_lock(&comm->q_lock);
	comm->stop = 1;
	pthread_cond_signal(&comm->q_cond);
	pthread_mutex_unlock(&comm->q_lock);

	/*
	 * The worker may be parked inside a BLOCKING odl_tb5_stream_recv(),
	 * where it cannot observe ->stop. Setting the flag and joining would
	 * then hang forever whenever the peer disconnects or simply never
	 * sends the header/chunk we are waiting for - which is exactly the
	 * error-recovery path RCCL relies on. Close the stream first: the
	 * pending recv fails, the worker unwinds, and the join completes.
	 * Closing twice is harmless; the caller's close is now a no-op.
	 */
	if (comm->stream_id > 0) {
		odl_tb5_stream_close(comm->handle, comm->stream_id);
		comm->stream_id = 0;
	}

	pthread_join(comm->worker, NULL);
	pthread_mutex_destroy(&comm->q_lock);
	pthread_cond_destroy(&comm->q_cond);
}

static rcclResult_t start_request(struct odl_tb5_comm *comm, void *data,
				  int size, int is_send, void **request)
{
	struct odl_tb5_request *req = calloc(1, sizeof(*req));

	if (!req)
		return rcclSystemError;
	req->comm = comm;
	req->data = data;
	req->size = size;
	req->is_send = is_send;
	req->done = 0;
	req->next = NULL;

	pthread_mutex_lock(&comm->q_lock);
	if (comm->q_tail)
		comm->q_tail->next = req;
	else
		comm->q_head = req;
	comm->q_tail = req;
	pthread_cond_signal(&comm->q_cond);
	pthread_mutex_unlock(&comm->q_lock);

	*request = req;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_isend(void *sendComm, void *data, int size,
				  int tag, void *mhandle, void **request)
{
	struct odl_tb5_comm *c = sendComm;
	(void)mhandle;
	DBG(1, "isend   POST comm=%p sid=%u dst=%u size=%d tag=%d", sendComm, c->stream_id, c->dst_id, size, tag);
	return start_request(sendComm, data, size, 1, request);
}

static rcclResult_t odl_tb5_irecv(void *recvComm, int n, void **data,
				  int *sizes, int *tags, void **mhandles,
				  void **request)
{
	struct odl_tb5_comm *c = recvComm;
	(void)tags; (void)mhandles;
	/*
	 * getProperties advertises maxRecvs = 1, so RCCL should only ever pass
	 * n == 1. Reject anything else rather than silently servicing element
	 * zero and dropping the rest - that would look like a successful
	 * transfer while losing the caller's data, which is the hardest class
	 * of bug to find. Observed n has always been 1 in practice; this makes
	 * the assumption enforced instead of implicit.
	 */
	if (n != 1) {
		WARN("irecv called with n=%d but maxRecvs=1 - refusing", n);
		return rcclInvalidArgument;
	}
	DBG(1, "irecv   POST comm=%p sid=%u n=%d size=%d", recvComm, c->stream_id, n, sizes[0]);
	return start_request(recvComm, data[0], sizes[0], 0, request);
}

static rcclResult_t odl_tb5_iflush(void *recvComm, int n, void **data,
				   int *sizes, void **mhandles, void **request)
{
	(void)recvComm; (void)n; (void)data; (void)sizes; (void)mhandles;
	*request = NULL;   /* host memory: nothing to flush */
	return rcclSuccess;
}

static rcclResult_t odl_tb5_test(void *request, int *done, int *sizes)
{
	struct odl_tb5_request *req = request;

	if (!req) {
		*done = 1;
		return rcclSuccess;
	}

	if (__atomic_load_n(&req->done, __ATOMIC_ACQUIRE)) {
		*done = 1;
		if (req->failed) {
			free(req);
			return rcclSystemError;
		}
		if (sizes)
			sizes[0] = req->done_size;
		free(req);
	} else {
		*done = 0;
	}
	return rcclSuccess;
}

static rcclResult_t odl_tb5_closeSend(void *sendComm)
{
	struct odl_tb5_comm *comm = sendComm;
	if (!comm)
		return rcclSuccess;
	DBG(1, "close   comm=%p sid=%u is_send=%d", (void *)comm, comm->stream_id, comm->is_send);
	comm_stop_worker(comm);
	if (comm->stream_id > 0)   /* may already be closed by stop_worker */
		odl_tb5_stream_close(comm->handle, comm->stream_id);
	put_shared_handle();
	free(comm);
	return rcclSuccess;
}

static rcclResult_t odl_tb5_closeRecv(void *recvComm)
{
	return odl_tb5_closeSend(recvComm);
}

static rcclResult_t odl_tb5_getDeviceMr(void *comm, void *mhandle,
					void **dptr_mhandle)
{
	(void)comm; (void)mhandle;
	if (dptr_mhandle)
		*dptr_mhandle = NULL;
	return rcclSuccess;
}

static rcclResult_t odl_tb5_irecvConsumed(void *recvComm, int n, void *request)
{
	(void)recvComm; (void)n; (void)request;
	return rcclSuccess;
}

rcclNet_v7_t rcclNetPlugin_v7 = {
	.name          = "ODL_TB5",
	.init          = odl_tb5_init,
	.devices       = odl_tb5_devices,
	.getProperties = odl_tb5_getProperties,
	.listen        = odl_tb5_listen,
	.connect       = odl_tb5_connect,
	.accept        = odl_tb5_accept,
	.regMr         = odl_tb5_regMr,
	.regMrDmaBuf   = odl_tb5_regMrDmaBuf,
	.deregMr       = odl_tb5_deregMr,
	.isend         = odl_tb5_isend,
	.irecv         = odl_tb5_irecv,
	.iflush        = odl_tb5_iflush,
	.test          = odl_tb5_test,
	.closeSend     = odl_tb5_closeSend,
	.closeRecv     = odl_tb5_closeRecv,
	.closeListen   = odl_tb5_closeListen,
	.getDeviceMr   = odl_tb5_getDeviceMr,
	.irecvConsumed = odl_tb5_irecvConsumed,
};

/* RCCL's plugin loader dlsym()s the NCCL-prefixed symbol ncclNetPlugin_vN. */
#ifdef __APPLE__
/* Mach-O has no symbol aliases, so publish a real second symbol and fill it
 * in at load time — before any dlsym() caller can reach it. */
rcclNet_v7_t ncclNetPlugin_v7;

__attribute__((constructor))
static void odl_tb5_export_nccl_symbol(void)
{
	ncclNetPlugin_v7 = rcclNetPlugin_v7;
}
#else
extern rcclNet_v7_t ncclNetPlugin_v7 __attribute__((alias("rcclNetPlugin_v7")));
#endif
