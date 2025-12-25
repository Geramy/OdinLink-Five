/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2025-2026 OdinLink Project */

#include "odl_tb5_daemon_sysinfo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <unistd.h>

/* Strip trailing whitespace from a string in place. */
static void trim(char *s)
{
	size_t len = strlen(s);
	if (len == 0)
		return;
	char *end = s + len - 1;
	while (end > s && isspace((unsigned char)*end))
		*end-- = '\0';
}

/* Read a sysfs file into a buffer and trim trailing whitespace. */
static int read_sysfs_str(const char *path, char *buf, size_t bufsz)
{
	FILE *f = fopen(path, "r");
	if (!f)
		return -1;
	if (!fgets(buf, (int)bufsz, f)) {
		fclose(f);
		return -1;
	}
	fclose(f);
	trim(buf);
	return 0;
}

/* Read a sysfs file and return its contents as an unsigned long long. */
static unsigned long long read_sysfs_ull(const char *path)
{
	char buf[64];
	if (read_sysfs_str(path, buf, sizeof(buf)) < 0)
		return 0;
	return strtoull(buf, NULL, 10);
}

/* Find existing CPU entry by physical ID or append a new one. */
static int find_or_add_cpu(struct odl_sysinfo *info, int phys_id)
{
	for (int i = 0; i < info->num_cpus; i++) {
		if (i == phys_id)
			return i;
	}

	if (info->num_cpus >= ODL_SYSINFO_MAX_CPUS)
		return -1;

	int idx = info->num_cpus++;
	memset(&info->cpus[idx], 0, sizeof(info->cpus[idx]));
	return idx;
}

/* Parse /proc/cpuinfo and populate per-socket CPU entries. */
static void collect_cpus(struct odl_sysinfo *info)
{
	FILE *f = fopen("/proc/cpuinfo", "r");
	if (!f)
		return;

	char line[256];
	int cur_phys_id = 0;
	int cur_idx = -1;
	char cur_model[128] = {0};
	int cur_cores = 0;
	double cur_mhz = 0;

	int phys_id_map[256];
	memset(phys_id_map, -1, sizeof(phys_id_map));

	while (fgets(line, sizeof(line), f)) {
		if (line[0] == '\n') {
			if (cur_model[0] && cur_phys_id < 256) {
				int idx = phys_id_map[cur_phys_id];
				if (idx < 0 && info->num_cpus < ODL_SYSINFO_MAX_CPUS) {
					idx = info->num_cpus++;
					phys_id_map[cur_phys_id] = idx;
					memset(&info->cpus[idx], 0, sizeof(info->cpus[idx]));
					snprintf(info->cpus[idx].model,
					         sizeof(info->cpus[idx].model),
					         "%s", cur_model);
					info->cpus[idx].cores = (unsigned int)cur_cores;
				}
				if (idx >= 0) {
					info->cpus[idx].threads++;
					if ((unsigned int)(cur_mhz + 0.5) > info->cpus[idx].freq_mhz)
						info->cpus[idx].freq_mhz = (unsigned int)(cur_mhz + 0.5);
				}
			}
			cur_model[0] = '\0';
			cur_phys_id = 0;
			cur_cores = 0;
			cur_mhz = 0;
			continue;
		}

		if (strncmp(line, "model name", 10) == 0) {
			char *val = strchr(line, ':');
			if (val) {
				val++;
				while (*val && isspace((unsigned char)*val))
					val++;
				snprintf(cur_model, sizeof(cur_model), "%s", val);
				trim(cur_model);
			}
		} else if (strncmp(line, "physical id", 11) == 0) {
			char *val = strchr(line, ':');
			if (val)
				cur_phys_id = atoi(val + 1);
		} else if (strncmp(line, "cpu cores", 9) == 0) {
			char *val = strchr(line, ':');
			if (val)
				cur_cores = atoi(val + 1);
		} else if (strncmp(line, "cpu MHz", 7) == 0) {
			char *val = strchr(line, ':');
			if (val)
				cur_mhz = atof(val + 1);
		}
	}

	if (cur_model[0] && cur_phys_id < 256) {
		int idx = phys_id_map[cur_phys_id];
		if (idx < 0 && info->num_cpus < ODL_SYSINFO_MAX_CPUS) {
			idx = info->num_cpus++;
			phys_id_map[cur_phys_id] = idx;
			memset(&info->cpus[idx], 0, sizeof(info->cpus[idx]));
			snprintf(info->cpus[idx].model,
			         sizeof(info->cpus[idx].model),
			         "%s", cur_model);
			info->cpus[idx].cores = (unsigned int)cur_cores;
		}
		if (idx >= 0) {
			info->cpus[idx].threads++;
			if ((unsigned int)(cur_mhz + 0.5) > info->cpus[idx].freq_mhz)
				info->cpus[idx].freq_mhz = (unsigned int)(cur_mhz + 0.5);
		}
	}

	fclose(f);

	for (int i = 0; i < info->num_cpus; i++) {
		char path[128];
		snprintf(path, sizeof(path),
		         "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
		unsigned long long khz = read_sysfs_ull(path);
		if (khz > 0 && (unsigned int)(khz / 1000) > info->cpus[i].freq_mhz)
			info->cpus[i].freq_mhz = (unsigned int)(khz / 1000);
	}
}

/* Parse /proc/meminfo and populate total and available RAM. */
static void collect_ram(struct odl_sysinfo *info)
{
	FILE *f = fopen("/proc/meminfo", "r");
	if (!f)
		return;

	char line[128];
	while (fgets(line, sizeof(line), f)) {
		unsigned long kb;
		if (sscanf(line, "MemTotal: %lu kB", &kb) == 1)
			info->ram_total_mb = (unsigned int)(kb / 1024);
		else if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1)
			info->ram_available_mb = (unsigned int)(kb / 1024);
	}

	fclose(f);
}

/* Add an AMD GPU entry from sysfs product_name and VRAM attributes. */
static void add_gpu_amd(struct odl_sysinfo *info, const char *card_name)
{
	if (info->num_gpus >= ODL_SYSINFO_MAX_GPUS)
		return;

	struct odl_gpu_info *gpu = &info->gpus[info->num_gpus];
	char path[512];

	snprintf(path, sizeof(path),
	         "/sys/class/drm/%s/device/product_name", card_name);
	if (read_sysfs_str(path, gpu->name, sizeof(gpu->name)) < 0) {
		char device[16] = {0};
		snprintf(path, sizeof(path),
		         "/sys/class/drm/%s/device/device", card_name);
		if (read_sysfs_str(path, device, sizeof(device)) < 0)
			return;
		snprintf(gpu->name, sizeof(gpu->name),
		         "AMD Radeon [%s]", device);
	}

	snprintf(path, sizeof(path),
	         "/sys/class/drm/%s/device/mem_info_vram_total", card_name);
	unsigned long long vram_bytes = read_sysfs_ull(path);
	gpu->vram_total_mb = (unsigned int)(vram_bytes / (1024 * 1024));

	snprintf(path, sizeof(path),
	         "/sys/class/drm/%s/device/mem_info_vram_used", card_name);
	unsigned long long vram_used = read_sysfs_ull(path);
	gpu->vram_used_mb = (unsigned int)(vram_used / (1024 * 1024));

	info->num_gpus++;
}

/* Add an NVIDIA GPU entry using nvidia-smi. */
static void add_gpu_nvidia(struct odl_sysinfo *info, int gpu_index)
{
	if (info->num_gpus >= ODL_SYSINFO_MAX_GPUS)
		return;

	char cmd[256];
	snprintf(cmd, sizeof(cmd),
	         "nvidia-smi -i %d "
	         "--query-gpu=name,memory.total,memory.used "
	         "--format=csv,noheader,nounits 2>/dev/null",
	         gpu_index);

	FILE *nv = popen(cmd, "r");
	if (!nv)
		return;

	struct odl_gpu_info *gpu = &info->gpus[info->num_gpus];
	char line[256];
	if (fgets(line, sizeof(line), nv)) {
		char *p1 = strchr(line, ',');
		if (p1) {
			*p1 = '\0';
			snprintf(gpu->name, sizeof(gpu->name), "%s", line);
			trim(gpu->name);

			unsigned int total = 0, used = 0;
			if (sscanf(p1 + 1, " %u, %u", &total, &used) >= 1) {
				gpu->vram_total_mb = total;
				gpu->vram_used_mb = used;
			}
			info->num_gpus++;
		}
	}

	pclose(nv);
}

/* Add a GPU entry using generic PCI vendor/device IDs. */
static void add_gpu_generic(struct odl_sysinfo *info, const char *card_name)
{
	if (info->num_gpus >= ODL_SYSINFO_MAX_GPUS)
		return;

	char path[512];
	char vendor[16] = {0};
	char device[16] = {0};

	snprintf(path, sizeof(path),
	         "/sys/class/drm/%s/device/vendor", card_name);
	read_sysfs_str(path, vendor, sizeof(vendor));

	snprintf(path, sizeof(path),
	         "/sys/class/drm/%s/device/device", card_name);
	read_sysfs_str(path, device, sizeof(device));

	if (!vendor[0] && !device[0])
		return;

	struct odl_gpu_info *gpu = &info->gpus[info->num_gpus];

	const char *vname = "Unknown";
	if (strcmp(vendor, "0x10de") == 0)
		vname = "NVIDIA";
	else if (strcmp(vendor, "0x1002") == 0)
		vname = "AMD";
	else if (strcmp(vendor, "0x8086") == 0)
		vname = "Intel";

	snprintf(gpu->name, sizeof(gpu->name),
	         "%s GPU [%s:%s]", vname, vendor, device);

	info->num_gpus++;
}

/* Enumerate DRM card devices and collect GPU name and VRAM info. */
static void collect_gpus(struct odl_sysinfo *info)
{
	DIR *drm = opendir("/sys/class/drm");
	if (!drm)
		return;

	int nvidia_index = 0;

	struct dirent *ent;
	while ((ent = readdir(drm)) != NULL) {
		if (strncmp(ent->d_name, "card", 4) != 0)
			continue;
		if (!isdigit((unsigned char)ent->d_name[4]))
			continue;
		if (strchr(ent->d_name + 4, '-'))
			continue;
		if (info->num_gpus >= ODL_SYSINFO_MAX_GPUS)
			break;

		char path[512];

		snprintf(path, sizeof(path),
		         "/sys/class/drm/%s/device/vendor", ent->d_name);
		char vendor[16] = {0};
		read_sysfs_str(path, vendor, sizeof(vendor));

		if (strcmp(vendor, "0x1002") == 0) {
			int before = info->num_gpus;
			add_gpu_amd(info, ent->d_name);
			if (info->num_gpus == before)
				add_gpu_generic(info, ent->d_name);
		} else if (strcmp(vendor, "0x10de") == 0) {
			int before = info->num_gpus;
			add_gpu_nvidia(info, nvidia_index++);
			if (info->num_gpus == before)
				add_gpu_generic(info, ent->d_name);
		} else {
			add_gpu_generic(info, ent->d_name);
		}
	}

	closedir(drm);
}

void odl_daemon_sysinfo_collect(struct odl_sysinfo *info)
{
	memset(info, 0, sizeof(*info));
	collect_cpus(info);
	collect_ram(info);
	collect_gpus(info);
}
