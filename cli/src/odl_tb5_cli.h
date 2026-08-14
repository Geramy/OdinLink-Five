/*
 * OdinLink — CLI: Bandwidth, Latency, Jitter Tests Over Thunderbolt
 *
 * The command-line tool for measuring real-world performance:
 * point-to-point bandwidth, round-trip latency, jitter patterns,
 * and MIMO (multiple simultaneous streams).
 */
#ifndef ODL_TB5_CLI_H
#define ODL_TB5_CLI_H

#include <odl_tb5/odl_tb5.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define ODL_CLI_MAGIC  0x4F444C43

/* Well-known stream IDs for multiplexed I/O */
#define ODL_STREAM_TEST    1   /* daemon test / sysinfo server */
#define ODL_STREAM_SYNC    2   /* daemon file sync engine */
#define ODL_STREAM_CLI     10  /* CLI test client */

enum odl_cli_msg_type {
	ODL_CLI_MSG_HELLO       = 0x01,
	ODL_CLI_MSG_HELLO_ACK   = 0x02,
	ODL_CLI_MSG_TEST_REQ    = 0x10,
	ODL_CLI_MSG_TEST_ACK    = 0x11,
	ODL_CLI_MSG_TEST_START  = 0x12,
	ODL_CLI_MSG_TEST_STOP   = 0x13,
	ODL_CLI_MSG_TEST_DATA   = 0x20,
	ODL_CLI_MSG_PING        = 0x30,
	ODL_CLI_MSG_PONG        = 0x31,
	ODL_CLI_MSG_RESULT      = 0x40,
	ODL_CLI_MSG_DONE        = 0x50,
	ODL_CLI_MSG_SYSINFO_REQ = 0x60,
	ODL_CLI_MSG_SYSINFO_RESP= 0x61,
};

enum odl_cli_test_type {
	ODL_TEST_BANDWIDTH     = 1,
	ODL_TEST_LATENCY       = 2,
	ODL_TEST_LATENCY_LOAD  = 3,
	ODL_TEST_MIMO          = 4,
	ODL_TEST_JITTER        = 5,
	ODL_TEST_ALL           = 99,
};

struct odl_cli_header {
	uint32_t magic;
	uint32_t type;
	uint32_t sequence;
	uint32_t payload_len;
	uint64_t timestamp_ns;
};

struct odl_cli_hello {
	struct odl_cli_header hdr;
	char hostname[64];
	uint32_t version;
	uint32_t capabilities;
};

struct odl_cli_test_req {
	struct odl_cli_header hdr;
	uint32_t test_type;
	uint32_t block_size;
	uint32_t iterations;
	uint32_t duration_sec;
	uint32_t flags;
	uint32_t num_streams;
	uint32_t bg_block_size;
	uint32_t reserved;
};

#define ODL_TEST_FLAG_BIDIR     (1 << 0)
#define ODL_TEST_FLAG_WARMUP    (1 << 1)

struct odl_cli_result {
	struct odl_cli_header hdr;
	uint64_t bytes_transferred;
	uint64_t elapsed_ns;
	uint64_t min_latency_ns;
	uint64_t max_latency_ns;
	uint64_t avg_latency_ns;
	uint64_t p50_latency_ns;
	uint64_t p99_latency_ns;
	uint64_t p999_latency_ns;
	uint64_t jitter_ns;
};

#define ODL_SYSINFO_MAX_CPUS_WIRE  8
#define ODL_SYSINFO_MAX_GPUS_WIRE  8

struct odl_cli_cpu_wire {
	char     model[128];
	uint32_t cores;
	uint32_t threads;
	uint32_t freq_mhz;
	uint32_t reserved;
};

struct odl_cli_gpu_wire {
	char     name[128];
	uint32_t vram_total_mb;
	uint32_t vram_used_mb;
};

struct odl_cli_sysinfo_payload {
	uint32_t num_cpus;
	uint32_t ram_total_mb;
	uint32_t ram_available_mb;
	uint32_t num_gpus;
	struct odl_cli_cpu_wire cpus[ODL_SYSINFO_MAX_CPUS_WIRE];
	struct odl_cli_gpu_wire gpus[ODL_SYSINFO_MAX_GPUS_WIRE];
};

struct odl_cli_params {
	int device_index;
	enum odl_cli_test_type test_type;
	uint32_t block_sizes[16];
	int num_block_sizes;
	uint32_t iterations;
	uint32_t duration_sec;
	uint32_t num_streams;
	uint32_t bg_block_size;
	uint32_t warmup_iters;
	bool bidir;
	bool verbose;
	bool quiet;
	const char *output_file;
};

#define ODL_DEFAULT_ITERATIONS     1000
#define ODL_DEFAULT_DURATION       10
#define ODL_DEFAULT_STREAMS        4
#define ODL_DEFAULT_BG_BLOCK_SIZE  (1024 * 1024)
#define ODL_DEFAULT_WARMUP         10
#define ODL_DEFAULT_BLOCK_SIZE     (1024 * 1024)

#define ODL_STATS_MAX_SAMPLES   1000000
#define ODL_HIST_BUCKETS        13

struct odl_stats {
	uint64_t *samples;
	size_t    count;
	size_t    capacity;
	uint64_t  min_ns;
	uint64_t  max_ns;
	uint64_t  sum_ns;
	double    avg_ns;
	double    stddev_ns;
	uint64_t  median_ns;
	uint64_t  p50_ns;
	uint64_t  p95_ns;
	uint64_t  p99_ns;
	uint64_t  p999_ns;
	uint64_t  hist[ODL_HIST_BUCKETS];
};

static inline uint64_t odl_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline const char *odl_format_latency(uint64_t ns, char *buf, size_t len)
{
	if (ns < 1000)
		snprintf(buf, len, "%lu ns", (unsigned long)ns);
	else if (ns < 1000000)
		snprintf(buf, len, "%.2f us", ns / 1000.0);
	else if (ns < 1000000000ULL)
		snprintf(buf, len, "%.2f ms", ns / 1000000.0);
	else
		snprintf(buf, len, "%.3f s", ns / 1000000000.0);
	return buf;
}

static inline const char *odl_format_throughput(uint64_t bytes, uint64_t ns,
						char *buf, size_t len)
{
	if (ns == 0) {
		snprintf(buf, len, "N/A");
		return buf;
	}
	double gbps = (double)bytes * 8.0 / (double)ns;
	double gbytes_s = (double)bytes / (double)ns;
	snprintf(buf, len, "%.2f Gb/s (%.2f GB/s)", gbps, gbytes_s);
	return buf;
}

static inline const char *odl_format_size(uint64_t bytes, char *buf, size_t len)
{
	if (bytes < 1024)
		snprintf(buf, len, "%lu B", (unsigned long)bytes);
	else if (bytes < 1024 * 1024)
		snprintf(buf, len, "%lu KB", (unsigned long)(bytes / 1024));
	else if (bytes < 1024ULL * 1024 * 1024)
		snprintf(buf, len, "%lu MB", (unsigned long)(bytes / (1024 * 1024)));
	else
		snprintf(buf, len, "%.1f GB", bytes / (1024.0 * 1024 * 1024));
	return buf;
}

int odl_cli_send_msg(odl_tb5_t handle, uint8_t stream_id, uint8_t dst_id,
		     uint32_t type, uint32_t seq,
		     const void *payload, size_t payload_len);
int odl_cli_recv_msg(odl_tb5_t handle, uint8_t stream_id,
		     void *buf, size_t buf_size,
		     uint32_t *type, uint32_t *seq, uint8_t *src_id);
int odl_cli_send_hello(odl_tb5_t handle, uint8_t stream_id, uint8_t dst_id);
int odl_cli_recv_hello(odl_tb5_t handle, uint8_t stream_id);

int  odl_stats_init(struct odl_stats *stats, size_t capacity);
void odl_stats_free(struct odl_stats *stats);
void odl_stats_add(struct odl_stats *stats, uint64_t sample_ns);
void odl_stats_finalize(struct odl_stats *stats);
void odl_stats_print(const struct odl_stats *stats, const char *label);
void odl_stats_print_histogram(const struct odl_stats *stats);
void odl_stats_write_csv(const struct odl_stats *stats, const char *path);

int odl_cli_run_server(const struct odl_cli_params *params);

int odl_cli_run_diag(const struct odl_cli_params *params);

int odl_cli_run_client(const struct odl_cli_params *params);

int odl_cli_bandwidth_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			     const struct odl_cli_params *params);
int odl_cli_latency_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			    const struct odl_cli_params *params);
int odl_cli_latency_load_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
				const struct odl_cli_params *params);
int odl_cli_mimo_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			 const struct odl_cli_params *params);
int odl_cli_jitter_client(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			   const struct odl_cli_params *params);

int odl_cli_bandwidth_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			      const struct odl_cli_test_req *req);
int odl_cli_latency_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			    const struct odl_cli_test_req *req);
int odl_cli_latency_load_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
				 const struct odl_cli_test_req *req);
int odl_cli_mimo_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			 const struct odl_cli_test_req *req);
int odl_cli_jitter_server(odl_tb5_t handle, uint8_t sid, uint8_t dst,
			   const struct odl_cli_test_req *req);

#endif
