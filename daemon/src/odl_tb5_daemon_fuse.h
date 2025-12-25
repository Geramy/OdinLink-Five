/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_FUSE_H
#define ODL_TB5_DAEMON_FUSE_H

#include <sys/stat.h>
#include <sys/types.h>

/* FUSE callbacks provided by the file operations engine. */
struct odl_fuse_callbacks {
	int (*getattr)(const char *rel_path, struct stat *st);
	int (*readdir)(const char *dir_path,
		       void (*add_entry)(const char *name,
					 const struct stat *st,
					 void *ctx),
		       void *ctx);
	char *(*get_local_path)(const char *rel_path);
	int (*fetch_remote)(const char *rel_path);
	int (*create_file)(const char *rel_path, mode_t mode);
	int (*delete_file)(const char *rel_path);
	int (*create_dir)(const char *rel_path, mode_t mode);
	int (*delete_dir)(const char *rel_path);
};

/* Register callbacks from the file operations engine. */
void odl_daemon_fuse_set_callbacks(const struct odl_fuse_callbacks *cb);

/* Initialize FUSE mount at the given path. */
int  odl_daemon_fuse_init(const char *mount_point);

/* Unmount and clean up. */
void odl_daemon_fuse_shutdown(void);

/* Returns true if FUSE is currently mounted. */
int  odl_daemon_fuse_is_mounted(void);

/* Invalidate FUSE cache for a path after a remote fetch. */
void odl_daemon_fuse_invalidate(const char *rel_path);

#endif /* ODL_TB5_DAEMON_FUSE_H */
