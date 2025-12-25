/*
 * OdinLink TB5 Daemon - Distributed File Operations Engine
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_SYNC_H
#define ODL_TB5_DAEMON_SYNC_H

#include <glib.h>
#include <stdbool.h>
#include <stdint.h>
#include <odl_tb5/odl_tb5.h>

struct odl_daemon_sync_status {
	bool     enabled;
	uint32_t files_pending;
	uint64_t bytes_transferred;
	char     last_sync_time[32];
	GMutex   lock;
};

extern struct odl_daemon_sync_status g_sync_status;

struct odl_catalog_entry;

int  odl_daemon_sync_init(const char *folder_path);
void odl_daemon_sync_shutdown(void);
int  odl_daemon_sync_set_folder(const char *path);
void odl_daemon_sync_set_enabled(bool enabled);

int  odl_daemon_sync_fetch_file(const char *rel_path);
int  odl_daemon_sync_transfer_file(const char *rel_path);
int  odl_daemon_sync_remove_from_peer(const char *rel_path);
int  odl_daemon_sync_remove_local(const char *rel_path);
char *odl_daemon_sync_list_files_json(const char *dir_path);

struct odl_sysinfo;

bool odl_daemon_sync_owns_device(int index);
int  odl_daemon_sync_request_sysinfo(int device_index,
				     struct odl_sysinfo *out);

#endif /* ODL_TB5_DAEMON_SYNC_H */
