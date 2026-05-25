/*
 * OdinLink — Daemon: File Catalog (Index of Peer Files)
 *
 * Maintains a local index of which files exist on the peer and their
 * checksums. Used by the sync engine to decide what needs updating.
 * SPDX-License-Identifier: MIT
 */
#include "odl_tb5_daemon_catalog.h"
#include "odl_tb5_daemon_sync_proto.h"

#include <glib.h>
#include <glib/gstdio.h>

#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CATALOG_LOG_PREFIX  "odl_tb5_daemon: catalog: "

struct odl_catalog g_catalog;

static void catalog_scan_dir(const char *abs_dir, const char *rel_dir);

int odl_catalog_init(const char *local_store, const char *cache_dir)
{
	memset(&g_catalog, 0, sizeof(g_catalog));

	if (!local_store || !local_store[0]) {
		g_printerr(CATALOG_LOG_PREFIX
			   "local_store path must not be empty\n");
		return -EINVAL;
	}

	if (!cache_dir || !cache_dir[0]) {
		g_printerr(CATALOG_LOG_PREFIX
			   "cache_dir path must not be empty\n");
		return -EINVAL;
	}

	g_strlcpy(g_catalog.local_store, local_store,
		  sizeof(g_catalog.local_store));
	g_strlcpy(g_catalog.cache_dir, cache_dir,
		  sizeof(g_catalog.cache_dir));

	g_rw_lock_init(&g_catalog.lock);

	g_catalog.entries = g_hash_table_new_full(g_str_hash, g_str_equal,
						  g_free, g_free);

	if (g_mkdir_with_parents(g_catalog.local_store, 0755) < 0) {
		g_printerr(CATALOG_LOG_PREFIX
			   "failed to create local_store %s: %s\n",
			   g_catalog.local_store, strerror(errno));
		return -errno;
	}

	if (g_mkdir_with_parents(g_catalog.cache_dir, 0755) < 0) {
		g_printerr(CATALOG_LOG_PREFIX
			   "failed to create cache_dir %s: %s\n",
			   g_catalog.cache_dir, strerror(errno));
		return -errno;
	}

	g_printerr(CATALOG_LOG_PREFIX
		   "initialized (store: %s, cache: %s)\n",
		   g_catalog.local_store, g_catalog.cache_dir);

	return 0;
}

void odl_catalog_shutdown(void)
{
	if (g_catalog.entries) {
		g_hash_table_unref(g_catalog.entries);
		g_catalog.entries = NULL;
	}

	g_rw_lock_clear(&g_catalog.lock);

	g_printerr(CATALOG_LOG_PREFIX "shutdown complete\n");
}

static void catalog_scan_dir(const char *abs_dir, const char *rel_dir)
{
	GDir *dir = g_dir_open(abs_dir, 0, NULL);
	if (!dir) {
		g_printerr(CATALOG_LOG_PREFIX
			   "scan: cannot open directory %s: %s\n",
			   abs_dir, strerror(errno));
		return;
	}

	const gchar *name;
	while ((name = g_dir_read_name(dir)) != NULL) {
		if (name[0] == '.')
			continue;

		char *child_abs = g_build_filename(abs_dir, name, NULL);

		char *child_rel;
		if (rel_dir[0] == '\0')
			child_rel = g_strdup(name);
		else
			child_rel = g_strdup_printf("%s/%s", rel_dir, name);

		struct stat st;
		if (stat(child_abs, &st) < 0) {
			g_free(child_abs);
			g_free(child_rel);
			continue;
		}

		struct odl_catalog_entry *ent =
			g_malloc0(sizeof(struct odl_catalog_entry));

		g_strlcpy(ent->rel_path, child_rel,
			  sizeof(ent->rel_path));
		ent->mtime_ns = (uint64_t)st.st_mtim.tv_sec * 1000000000ULL +
				(uint64_t)st.st_mtim.tv_nsec;
		ent->mode = (uint32_t)st.st_mode & 07777;
		ent->location = ODL_FILE_LOCAL;

		if (S_ISDIR(st.st_mode)) {
			ent->is_dir = true;
			ent->file_size = 0;
			memset(ent->sha256, 0, sizeof(ent->sha256));

			g_hash_table_insert(g_catalog.entries,
					    g_strdup(child_rel), ent);

			catalog_scan_dir(child_abs, child_rel);
		} else if (S_ISREG(st.st_mode)) {
			ent->is_dir = false;
			ent->file_size = (uint64_t)st.st_size;
			memset(ent->sha256, 0, sizeof(ent->sha256));

			g_hash_table_insert(g_catalog.entries,
					    g_strdup(child_rel), ent);
		} else {
			g_free(ent);
		}

		g_free(child_abs);
		g_free(child_rel);
	}

	g_dir_close(dir);
}

int odl_catalog_scan_local(void)
{
	g_rw_lock_writer_lock(&g_catalog.lock);

	GHashTableIter iter;
	gpointer key, value;

	g_hash_table_iter_init(&iter, g_catalog.entries);
	while (g_hash_table_iter_next(&iter, &key, &value)) {
		struct odl_catalog_entry *ent = value;
		if (ent->location == ODL_FILE_LOCAL)
			g_hash_table_iter_remove(&iter);
	}

	catalog_scan_dir(g_catalog.local_store, "");

	g_hash_table_iter_init(&iter, g_catalog.entries);
	while (g_hash_table_iter_next(&iter, &key, &value)) {
		struct odl_catalog_entry *ent = value;
		if (ent->location == ODL_FILE_LOCAL && !ent->is_dir) {
			char *cache_path = g_strdup_printf(
				"%s/%s", g_catalog.cache_dir, ent->rel_path);
			if (g_file_test(cache_path, G_FILE_TEST_EXISTS))
				ent->location = ODL_FILE_BOTH;
			g_free(cache_path);
		}
	}

	int count = g_hash_table_size(g_catalog.entries);

	g_rw_lock_writer_unlock(&g_catalog.lock);

	g_printerr(CATALOG_LOG_PREFIX
		   "scan complete: %d entries\n", count);

	return count;
}

void odl_catalog_update_remote(GList *remote_entries)
{
	g_rw_lock_writer_lock(&g_catalog.lock);

	GHashTableIter iter;
	gpointer key, value;

	g_hash_table_iter_init(&iter, g_catalog.entries);
	while (g_hash_table_iter_next(&iter, &key, &value)) {
		struct odl_catalog_entry *ent = value;
		if (ent->location == ODL_FILE_REMOTE)
			g_hash_table_iter_remove(&iter);
	}

	for (GList *l = remote_entries; l != NULL; l = l->next) {
		const struct odl_catalog_entry *remote = l->data;
		struct odl_catalog_entry *existing;

		existing = g_hash_table_lookup(g_catalog.entries,
					       remote->rel_path);

		if (existing) {
			if (existing->location == ODL_FILE_LOCAL) {
				existing->location = ODL_FILE_BOTH;
			}
		} else {
			struct odl_catalog_entry *ent =
				g_malloc0(sizeof(struct odl_catalog_entry));

			memcpy(ent, remote, sizeof(*ent));
			ent->location = ODL_FILE_REMOTE;

			g_hash_table_insert(g_catalog.entries,
					    g_strdup(ent->rel_path), ent);
		}
	}

	g_rw_lock_writer_unlock(&g_catalog.lock);

	g_printerr(CATALOG_LOG_PREFIX "remote listing updated\n");
}

void odl_catalog_set(const char *rel_path,
		     const struct odl_catalog_entry *entry)
{
	if (!rel_path || !entry)
		return;

	struct odl_catalog_entry *ent =
		g_malloc0(sizeof(struct odl_catalog_entry));
	memcpy(ent, entry, sizeof(*ent));

	g_strlcpy(ent->rel_path, rel_path, sizeof(ent->rel_path));

	g_rw_lock_writer_lock(&g_catalog.lock);
	g_hash_table_insert(g_catalog.entries, g_strdup(rel_path), ent);
	g_rw_lock_writer_unlock(&g_catalog.lock);
}

void odl_catalog_remove(const char *rel_path)
{
	if (!rel_path)
		return;

	g_rw_lock_writer_lock(&g_catalog.lock);
	g_hash_table_remove(g_catalog.entries, rel_path);
	g_rw_lock_writer_unlock(&g_catalog.lock);
}

struct odl_catalog_entry *odl_catalog_lookup(const char *rel_path)
{
	if (!rel_path)
		return NULL;

	struct odl_catalog_entry *copy = NULL;

	g_rw_lock_reader_lock(&g_catalog.lock);
	const struct odl_catalog_entry *ent =
		g_hash_table_lookup(g_catalog.entries, rel_path);
	if (ent) {
		copy = g_malloc(sizeof(*copy));
		memcpy(copy, ent, sizeof(*copy));
	}
	g_rw_lock_reader_unlock(&g_catalog.lock);

	return copy;
}

GList *odl_catalog_list_dir(const char *dir_path)
{
	GList *result = NULL;
	size_t prefix_len;
	bool is_root;

	if (!dir_path || dir_path[0] == '\0') {
		is_root = true;
		prefix_len = 0;
	} else {
		is_root = false;
		prefix_len = strlen(dir_path);
	}

	g_rw_lock_reader_lock(&g_catalog.lock);

	GHashTableIter iter;
	gpointer key, value;

	g_hash_table_iter_init(&iter, g_catalog.entries);
	while (g_hash_table_iter_next(&iter, &key, &value)) {
		const char *path = key;
		struct odl_catalog_entry *ent = value;

		if (is_root) {
			if (strchr(path, '/') == NULL)
				result = g_list_prepend(result, ent);
		} else {
			if (strncmp(path, dir_path, prefix_len) != 0)
				continue;
			if (path[prefix_len] != '/')
				continue;

			const char *rest = path + prefix_len + 1;
			if (rest[0] == '\0')
				continue;
			if (strchr(rest, '/') != NULL)
				continue;

			result = g_list_prepend(result, ent);
		}
	}

	g_rw_lock_reader_unlock(&g_catalog.lock);

	return result;
}

char *odl_catalog_local_path(const char *rel_path)
{
	return g_strdup_printf("%s/%s", g_catalog.local_store, rel_path);
}

char *odl_catalog_cache_path(const char *rel_path)
{
	return g_strdup_printf("%s/%s", g_catalog.cache_dir, rel_path);
}

void odl_catalog_mark_cached(const char *rel_path)
{
	if (!rel_path)
		return;

	g_rw_lock_writer_lock(&g_catalog.lock);

	struct odl_catalog_entry *ent =
		g_hash_table_lookup(g_catalog.entries, rel_path);

	if (ent && ent->location == ODL_FILE_REMOTE) {
		ent->location = ODL_FILE_CACHED;
		g_printerr(CATALOG_LOG_PREFIX
			   "marked cached: %s\n", rel_path);
	}

	g_rw_lock_writer_unlock(&g_catalog.lock);
}

void odl_catalog_mark_both(const char *rel_path)
{
	if (!rel_path)
		return;

	g_rw_lock_writer_lock(&g_catalog.lock);

	struct odl_catalog_entry *ent =
		g_hash_table_lookup(g_catalog.entries, rel_path);

	if (ent) {
		ent->location = ODL_FILE_BOTH;
		g_printerr(CATALOG_LOG_PREFIX
			   "marked both: %s\n", rel_path);
	}

	g_rw_lock_writer_unlock(&g_catalog.lock);
}

int odl_catalog_remove_local_copy(const char *rel_path)
{
	if (!rel_path)
		return -EINVAL;

	int ret = 0;

	g_rw_lock_writer_lock(&g_catalog.lock);

	struct odl_catalog_entry *ent =
		g_hash_table_lookup(g_catalog.entries, rel_path);

	if (!ent) {
		g_rw_lock_writer_unlock(&g_catalog.lock);
		return -ENOENT;
	}

	switch (ent->location) {
	case ODL_FILE_BOTH: {
		char *local = g_strdup_printf("%s/%s",
					      g_catalog.local_store,
					      rel_path);
		if (g_unlink(local) < 0 && errno != ENOENT) {
			g_printerr(CATALOG_LOG_PREFIX
				   "remove_local: unlink %s: %s\n",
				   local, strerror(errno));
			ret = -errno;
		} else {
			ent->location = ODL_FILE_REMOTE;
			g_printerr(CATALOG_LOG_PREFIX
				   "remove_local: %s -> REMOTE\n",
				   rel_path);
		}
		g_free(local);
		break;
	}

	case ODL_FILE_CACHED: {
		char *cache = g_strdup_printf("%s/%s",
					      g_catalog.cache_dir,
					      rel_path);
		if (g_unlink(cache) < 0 && errno != ENOENT) {
			g_printerr(CATALOG_LOG_PREFIX
				   "remove_local: unlink cache %s: %s\n",
				   cache, strerror(errno));
			ret = -errno;
		} else {
			ent->location = ODL_FILE_REMOTE;
			g_printerr(CATALOG_LOG_PREFIX
				   "remove_local: %s (cached) -> REMOTE\n",
				   rel_path);
		}
		g_free(cache);
		break;
	}

	case ODL_FILE_LOCAL: {
		char *local = g_strdup_printf("%s/%s",
					      g_catalog.local_store,
					      rel_path);
		if (g_unlink(local) < 0 && errno != ENOENT) {
			g_printerr(CATALOG_LOG_PREFIX
				   "remove_local: unlink %s: %s\n",
				   local, strerror(errno));
			ret = -errno;
		} else {
			g_hash_table_remove(g_catalog.entries, rel_path);
			g_printerr(CATALOG_LOG_PREFIX
				   "remove_local: %s removed entirely\n",
				   rel_path);
		}
		g_free(local);
		break;
	}

	case ODL_FILE_REMOTE:
		ret = -ENOENT;
		break;
	}

	g_rw_lock_writer_unlock(&g_catalog.lock);

	return ret;
}

int odl_catalog_count_local(void)
{
	int count = 0;

	g_rw_lock_reader_lock(&g_catalog.lock);

	GHashTableIter iter;
	gpointer key, value;

	g_hash_table_iter_init(&iter, g_catalog.entries);
	while (g_hash_table_iter_next(&iter, &key, &value)) {
		const struct odl_catalog_entry *ent = value;
		if (ent->location == ODL_FILE_LOCAL ||
		    ent->location == ODL_FILE_BOTH ||
		    ent->location == ODL_FILE_CACHED)
			count++;
	}

	g_rw_lock_reader_unlock(&g_catalog.lock);

	return count;
}

int odl_catalog_count_remote(void)
{
	int count = 0;

	g_rw_lock_reader_lock(&g_catalog.lock);

	GHashTableIter iter;
	gpointer key, value;

	g_hash_table_iter_init(&iter, g_catalog.entries);
	while (g_hash_table_iter_next(&iter, &key, &value)) {
		const struct odl_catalog_entry *ent = value;
		if (ent->location == ODL_FILE_REMOTE ||
		    ent->location == ODL_FILE_BOTH)
			count++;
	}

	g_rw_lock_reader_unlock(&g_catalog.lock);

	return count;
}
