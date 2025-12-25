/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2025-2026 OdinLink Project */
#ifndef ODL_TB5_DAEMON_SYSINFO_H
#define ODL_TB5_DAEMON_SYSINFO_H

#include <stdint.h>

#define ODL_SYSINFO_MAX_CPUS  8
#define ODL_SYSINFO_MAX_GPUS  8

struct odl_cpu_info {
	char         model[128];
	unsigned int cores;
	unsigned int threads;
	unsigned int freq_mhz;
};

struct odl_gpu_info {
	char         name[128];
	unsigned int vram_total_mb;
	unsigned int vram_used_mb;
};

struct odl_sysinfo {
	struct odl_cpu_info cpus[ODL_SYSINFO_MAX_CPUS];
	int                 num_cpus;
	unsigned int        ram_total_mb;
	unsigned int        ram_available_mb;
	struct odl_gpu_info gpus[ODL_SYSINFO_MAX_GPUS];
	int                 num_gpus;
};

void odl_daemon_sysinfo_collect(struct odl_sysinfo *info);

#endif
