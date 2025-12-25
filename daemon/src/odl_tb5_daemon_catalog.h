/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_CATALOG_H
#define ODL_TB5_DAEMON_CATALOG_H

#include "odl_tb5_daemon_sync_proto.h"

#include <glib.h>
#include <stdbool.h>
#include <stdint.h>

struct odl_catalog_entry {
	char     rel_path[256];
	uint64_t file_size;
	uint64_t mtime_ns;
	uint32_t mode;
	bool     is_dir;
	uint8_t  sha256[32];
	enum odl_file_location location;
};

struct odl_catalog {
	GHashTable *entries;
	GRWLock     lock;
	char        local_store[512];
	char        cache_dir[512];
};

extern struct odl_catalog g_catalog;

int  odl_catalog_init(const char *local_store, const char *cache_dir);
void odl_catalog_shutdown(void);

/* Scan local_store and populate catalog with LOCAL entries. */
int  odl_catalog_scan_local(void);

/* Replace all REMOTE entries with the peer's listing. */
void odl_catalog_update_remote(GList *remote_entries);

/* Add or update a single entry. */
void odl_catalog_set(const char *rel_path,
		     const struct odl_catalog_entry *entry);

/* Remove an entry. */
void odl_catalog_remove(const char *rel_path);

/* Lookup an entry (caller must NOT free). */
const struct odl_catalog_entry *odl_catalog_lookup(const char *rel_path);

/* List direct children of a directory (caller frees list, not entries). */
GList *odl_catalog_list_dir(const char *dir_path);

/* Return the full local path (caller must g_free). */
char *odl_catalog_local_path(const char *rel_path);

/* Return the full cache path (caller must g_free). */
char *odl_catalog_cache_path(const char *rel_path);

/* Mark a remote file as cached after DMA fetch. */
void odl_catalog_mark_cached(const char *rel_path);

/* Mark a file as present on both sides. */
void odl_catalog_mark_both(const char *rel_path);

/* Remove the local copy (BOTH->REMOTE, CACHED->REMOTE, LOCAL->removed). */
int  odl_catalog_remove_local_copy(const char *rel_path);

/* Count entries with a local or remote presence. */
int  odl_catalog_count_local(void);
int  odl_catalog_count_remote(void);

#endif /* ODL_TB5_DAEMON_CATALOG_H */
