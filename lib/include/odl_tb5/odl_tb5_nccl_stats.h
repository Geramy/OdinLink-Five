#ifndef ODL_TB5_NCCL_STATS_H
#define ODL_TB5_NCCL_STATS_H

#include <stdint.h>

#define ODL_NCCL_STATS_MAGIC   0x4F444C4E53435453ULL
#define ODL_NCCL_STATS_VERSION 1
#define ODL_NCCL_STATS_DIR     "/run/odl_tb5"
#define ODL_NCCL_STATS_PATH    "/run/odl_tb5/nccl_stats"

struct odl_nccl_stats {
	uint64_t magic;
	uint64_t version;
	uint64_t tx_bytes;
	uint64_t rx_bytes;
	uint64_t tx_ops;
	uint64_t rx_ops;
	uint64_t start_time_ns;
	uint64_t last_update_ns;
	uint32_t active;
	uint32_t reserved[15];
};

#endif
