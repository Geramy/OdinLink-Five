/*
 * OdinLink TB5 Daemon - Hybrid inotify + On-Demand Streaming Engine
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#include "odl_tb5_daemon_sync.h"
#include "odl_tb5_daemon_sync_proto.h"
#include "odl_tb5_daemon_catalog.h"
#include "odl_tb5_daemon_fuse.h"
#include "odl_tb5_daemon_dbus.h"
#include "odl_tb5_daemon_config.h"
#include "odl_tb5_daemon_sysinfo.h"
#include "odl_tb5_cli.h"

#include <glib.h>
#include <glib/gstdio.h>
#include <gio/gio.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <dirent.h>
#include <pthread.h>
#include <signal.h>

#define SYNC_LOG_PREFIX       "odl_tb5_daemon: sync: "
#define SYNC_TMP_PREFIX       ".odl_fetch_tmp_"
#define SYNC_PEER_TIMEOUT     5000
#define SYNC_CATALOG_INTERVAL 60000

#define RECV_BUF_SIZE  (sizeof(struct odl_sync_file_data) + ODL_SYNC_CHUNK_SIZE)

#define INOTIFY_BUF_SIZE  (16 * (sizeof(struct inotify_event) + NAME_MAX + 1))

#define INOTIFY_WATCH_MASK  (IN_CLOSE_WRITE | IN_CREATE | IN_DELETE | \
			     IN_MOVED_FROM | IN_MOVED_TO | IN_ISDIR)

struct odl_sync_engine {
	char         shared_folder[512];
	char         fuse_mount[512];
	char         cache_dir[512];
	bool         enabled;

	odl_tb5_t    handle;
	uint8_t      sid;
	int          device_index;
	GMutex       tx_lock;

	GThread     *recv_thread;
	volatile bool recv_running;
	pthread_t    recv_tid;

	uint32_t     send_seq;

	GMutex       sysinfo_lock;
	GCond        sysinfo_cond;
	bool         sysinfo_pending;
	int          sysinfo_result;
	struct odl_sysinfo *sysinfo_out;

	GMutex       fetch_lock;
	GCond        fetch_cond;
	char         fetch_pending_path[256];
	int          fetch_result;
	bool         fetch_in_progress;

	int          recv_tmp_fd;
	char        *recv_tmp_path;
	char         recv_rel_path[256];
	uint64_t     recv_file_size;
	uint64_t     recv_mtime_ns;
	uint32_t     recv_mode;
	uint32_t     recv_num_chunks;
	uint32_t     recv_chunks_got;
	uint8_t      recv_sha256[32];

	GList       *peer_listing;
	bool         listing_in_progress;

	guint        catalog_timer_id;

	int          inotify_fd;
	guint        inotify_watch_id;
	GIOChannel  *inotify_channel;
	GHashTable  *wd_to_path;
	GHashTable  *path_to_wd;

	GHashTable  *suppress_set;
	GMutex       suppress_lock;

	struct odl_daemon_sync_status *status;
};

struct odl_daemon_sync_status g_sync_status;

static struct odl_sync_engine g_engine;
static bool g_engine_initialized;

static void     sync_exchange_catalog(struct odl_sync_engine *eng);
static gboolean sync_catalog_timer_cb(gpointer data);
static gpointer sync_receiver_loop(gpointer data);

static void sync_handle_sync_req(struct odl_sync_engine *eng);
static void sync_handle_listing_entry(struct odl_sync_engine *eng,
				      const struct odl_sync_listing_entry *ent);
static void sync_handle_listing_end(struct odl_sync_engine *eng);
static void sync_handle_fetch_req(struct odl_sync_engine *eng,
				  const struct odl_sync_fetch_req *req);
static void sync_handle_fetch_resp(struct odl_sync_engine *eng,
				   const struct odl_sync_fetch_resp *resp);
static void sync_handle_file_data(struct odl_sync_engine *eng,
				  const void *buf);
static void sync_handle_file_meta(struct odl_sync_engine *eng,
				  const struct odl_sync_file_meta *meta);
static void sync_handle_file_ack(struct odl_sync_engine *eng,
				 const struct odl_sync_file_ack *ack);
static void sync_handle_file_delete(struct odl_sync_engine *eng,
				    const struct odl_sync_file_delete *del);
static void sync_handle_dir_create(struct odl_sync_engine *eng,
				   const struct odl_sync_dir_op *op);
static void sync_handle_dir_delete(struct odl_sync_engine *eng,
				   const struct odl_sync_dir_op *op);
static void sync_handle_remove_req(struct odl_sync_engine *eng,
				   const struct odl_sync_remove_req *req);
static void sync_handle_remove_ack(struct odl_sync_engine *eng,
				   const struct odl_sync_remove_ack *ack);
static void sync_handle_file_changed(struct odl_sync_engine *eng,
				     const struct odl_sync_file_changed *msg);
static void sync_handle_file_removed(struct odl_sync_engine *eng,
				     const struct odl_sync_file_removed *msg);

static void sync_handle_cli_msg(struct odl_sync_engine *eng,
				const void *buf, size_t len);
static void sync_respond_sysinfo(struct odl_sync_engine *eng);
static void sync_deliver_sysinfo_resp(struct odl_sync_engine *eng,
				      const void *buf, size_t len);

static int  fuse_cb_getattr(const char *rel_path, struct stat *st);
static int  fuse_cb_readdir(const char *dir_path,
			    void (*add_entry)(const char *name,
					      const struct stat *st,
					      void *ctx),
			    void *ctx);
static char *fuse_cb_get_local_path(const char *rel_path);
static int   fuse_cb_fetch_remote(const char *rel_path);
static int   fuse_cb_create_file(const char *rel_path, mode_t mode);
static int   fuse_cb_delete_file(const char *rel_path);
static int   fuse_cb_create_dir(const char *rel_path, mode_t mode);
static int   fuse_cb_delete_dir(const char *rel_path);

static int      inotify_start(struct odl_sync_engine *eng);
static void     inotify_stop(struct odl_sync_engine *eng);
static gboolean inotify_io_cb(GIOChannel *source, GIOCondition cond,
			      gpointer data);
static void     inotify_add_watch_recursive(struct odl_sync_engine *eng,
					    const char *abs_dir,
					    const char *rel_dir);
static void     inotify_remove_watch_for_path(struct odl_sync_engine *eng,
					      const char *rel_dir);

static void sync_create_remote_symlink(struct odl_sync_engine *eng,
				       const char *rel_path);
static void sync_remove_remote_symlink(struct odl_sync_engine *eng,
				       const char *rel_path);

static void sync_suppress_add(struct odl_sync_engine *eng,
			      const char *rel_path);
static bool sync_suppress_check_and_remove(struct odl_sync_engine *eng,
					   const char *rel_path);

static void sync_update_last_sync_time(void);
static void sync_status_add_bytes(uint64_t bytes);
static void sync_status_add_pending(int delta);

static char *sync_shared_path(const struct odl_sync_engine *eng,
			      const char *rel_path)
{
	return g_build_filename(eng->shared_folder, rel_path, NULL);
}

static char *sync_fuse_path(const struct odl_sync_engine *eng,
			    const char *rel_path)
{
	return g_build_filename(eng->fuse_mount, rel_path, NULL);
}

static char *sync_cache_path(const struct odl_sync_engine *eng,
			     const char *rel_path)
{
	return g_build_filename(eng->cache_dir, rel_path, NULL);
}

static uint64_t sync_get_mtime_ns(const char *abs_path)
{
	struct stat st;

	if (stat(abs_path, &st) < 0)
		return 0;

	return (uint64_t)st.st_mtim.tv_sec * 1000000000ULL +
	       (uint64_t)st.st_mtim.tv_nsec;
}

static const char *sync_rel_from_abs(const struct odl_sync_engine *eng,
				     const char *abs_path)
{
	size_t base_len = strlen(eng->shared_folder);

	if (strncmp(abs_path, eng->shared_folder, base_len) != 0)
		return NULL;

	const char *rel = abs_path + base_len;

	if (*rel == '/')
		rel++;

	return rel;
}

static void sync_update_last_sync_time(void)
{
	time_t now = time(NULL);
	struct tm tm;

	localtime_r(&now, &tm);

	g_mutex_lock(&g_sync_status.lock);
	strftime(g_sync_status.last_sync_time,
		 sizeof(g_sync_status.last_sync_time),
		 "%Y-%m-%d %H:%M:%S", &tm);
	g_mutex_unlock(&g_sync_status.lock);
}

static void sync_status_add_pending(int delta)
{
	g_mutex_lock(&g_sync_status.lock);
	if (delta < 0 && (uint32_t)(-delta) > g_sync_status.files_pending)
		g_sync_status.files_pending = 0;
	else
		g_sync_status.files_pending += delta;
	g_mutex_unlock(&g_sync_status.lock);
}

static void sync_status_add_bytes(uint64_t bytes)
{
	g_mutex_lock(&g_sync_status.lock);
	g_sync_status.bytes_transferred += bytes;
	g_mutex_unlock(&g_sync_status.lock);
}

static void sync_suppress_add(struct odl_sync_engine *eng,
			      const char *rel_path)
{
	g_mutex_lock(&eng->suppress_lock);
	g_hash_table_insert(eng->suppress_set,
			    g_strdup(rel_path), GINT_TO_POINTER(1));
	g_mutex_unlock(&eng->suppress_lock);
}

static bool sync_suppress_check_and_remove(struct odl_sync_engine *eng,
					   const char *rel_path)
{
	bool found;

	g_mutex_lock(&eng->suppress_lock);
	found = g_hash_table_remove(eng->suppress_set, rel_path);
	g_mutex_unlock(&eng->suppress_lock);

	return found;
}

static void sync_create_remote_symlink(struct odl_sync_engine *eng,
				       const char *rel_path)
{
	char *symlink_path = sync_shared_path(eng, rel_path);
	char *target = sync_fuse_path(eng, rel_path);

	sync_suppress_add(eng, rel_path);

	struct stat lst;
	if (lstat(symlink_path, &lst) == 0)
		unlink(symlink_path);

	char *parent = g_path_get_dirname(symlink_path);
	g_mkdir_with_parents(parent, 0755);
	g_free(parent);

	if (symlink(target, symlink_path) < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "symlink: failed to create %s -> %s: %s\n",
			   symlink_path, target, strerror(errno));
	} else {
		g_printerr(SYNC_LOG_PREFIX
			   "symlink: created %s -> %s\n",
			   rel_path, target);
	}

	g_free(symlink_path);
	g_free(target);
}

static void sync_remove_remote_symlink(struct odl_sync_engine *eng,
				       const char *rel_path)
{
	char *symlink_path = sync_shared_path(eng, rel_path);

	sync_suppress_add(eng, rel_path);

	struct stat lst;
	if (lstat(symlink_path, &lst) == 0) {
		if (S_ISLNK(lst.st_mode)) {
			unlink(symlink_path);
			g_printerr(SYNC_LOG_PREFIX
				   "symlink: removed %s\n", rel_path);
		}
	}

	g_free(symlink_path);
}

static void inotify_add_watch_recursive(struct odl_sync_engine *eng,
					const char *abs_dir,
					const char *rel_dir)
{
	int wd = inotify_add_watch(eng->inotify_fd, abs_dir,
				   INOTIFY_WATCH_MASK);
	if (wd < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "inotify: failed to watch %s: %s\n",
			   abs_dir, strerror(errno));
		return;
	}

	char *rel_dup = g_strdup(rel_dir);
	g_hash_table_insert(eng->wd_to_path, GINT_TO_POINTER(wd), rel_dup);
	g_hash_table_insert(eng->path_to_wd,
			    g_strdup(rel_dir), GINT_TO_POINTER(wd));

	DIR *d = opendir(abs_dir);
	if (!d)
		return;

	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		if (g_strcmp0(de->d_name, ".") == 0 ||
		    g_strcmp0(de->d_name, "..") == 0)
			continue;

		char *child_abs = g_build_filename(abs_dir, de->d_name, NULL);

		struct stat lst;
		if (lstat(child_abs, &lst) == 0 && S_ISDIR(lst.st_mode)) {
			char *child_rel;
			if (rel_dir[0] == '\0')
				child_rel = g_strdup(de->d_name);
			else
				child_rel = g_strdup_printf("%s/%s",
							    rel_dir,
							    de->d_name);

			inotify_add_watch_recursive(eng, child_abs,
						    child_rel);
			g_free(child_rel);
		}

		g_free(child_abs);
	}

	closedir(d);
}

static void inotify_remove_watch_for_path(struct odl_sync_engine *eng,
					  const char *rel_dir)
{
	gpointer wd_ptr;

	if (!g_hash_table_lookup_extended(eng->path_to_wd, rel_dir,
					  NULL, &wd_ptr))
		return;

	int wd = GPOINTER_TO_INT(wd_ptr);

	inotify_rm_watch(eng->inotify_fd, wd);
	g_hash_table_remove(eng->wd_to_path, GINT_TO_POINTER(wd));
	g_hash_table_remove(eng->path_to_wd, rel_dir);
}

static void inotify_send_file_changed(struct odl_sync_engine *eng,
				      const char *rel_path,
				      bool is_dir)
{
	char *abs_path = sync_shared_path(eng, rel_path);
	struct stat st;

	uint64_t file_size = 0;
	uint64_t mtime_ns = 0;
	uint32_t mode = 0;
	uint8_t sha256[32];

	memset(sha256, 0, sizeof(sha256));

	if (stat(abs_path, &st) == 0) {
		file_size = (uint64_t)st.st_size;
		mtime_ns = (uint64_t)st.st_mtim.tv_sec * 1000000000ULL +
			   (uint64_t)st.st_mtim.tv_nsec;
		mode = (uint32_t)st.st_mode & 07777;

		if (!is_dir && file_size > 0)
			odl_sync_sha256_file(abs_path, sha256);
	}

	struct odl_catalog_entry ce = {0};
	g_strlcpy(ce.rel_path, rel_path, sizeof(ce.rel_path));
	ce.file_size = file_size;
	ce.mtime_ns = mtime_ns;
	ce.mode = mode;
	ce.is_dir = is_dir;
	memcpy(ce.sha256, sha256, 32);
	ce.location = ODL_FILE_LOCAL;
	odl_catalog_set(rel_path, &ce);

	g_mutex_lock(&eng->tx_lock);
	int ret = odl_sync_send_file_changed(eng->handle, eng->sid,
					     ODL_STREAM_SYNC, &eng->send_seq,
					     rel_path, file_size, mtime_ns,
					     mode, is_dir, sha256);
	g_mutex_unlock(&eng->tx_lock);

	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "inotify: failed to send FILE_CHANGED for %s: %s\n",
			   rel_path, strerror(-ret));
	} else {
		g_printerr(SYNC_LOG_PREFIX
			   "inotify: sent FILE_CHANGED for %s\n", rel_path);
	}

	g_free(abs_path);
}

static void inotify_send_file_removed(struct odl_sync_engine *eng,
				      const char *rel_path,
				      bool is_dir)
{
	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);
	if (ce) {
		if (ce->location == ODL_FILE_BOTH) {
			struct odl_catalog_entry updated = *ce;
			updated.location = ODL_FILE_REMOTE;
			odl_catalog_set(rel_path, &updated);
		} else {
			odl_catalog_remove(rel_path);
		}
	}

	g_mutex_lock(&eng->tx_lock);
	int ret = odl_sync_send_file_removed(eng->handle, eng->sid,
					     ODL_STREAM_SYNC, &eng->send_seq,
					     rel_path, is_dir);
	g_mutex_unlock(&eng->tx_lock);

	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "inotify: failed to send FILE_REMOVED for %s: %s\n",
			   rel_path, strerror(-ret));
	} else {
		g_printerr(SYNC_LOG_PREFIX
			   "inotify: sent FILE_REMOVED for %s\n", rel_path);
	}
}

static gboolean inotify_io_cb(GIOChannel *source, GIOCondition cond,
			      gpointer data)
{
	struct odl_sync_engine *eng = data;
	char buf[INOTIFY_BUF_SIZE]
		__attribute__((aligned(__alignof__(struct inotify_event))));

	(void)source;
	(void)cond;

	if (!eng->enabled)
		return G_SOURCE_REMOVE;

	ssize_t len = read(eng->inotify_fd, buf, sizeof(buf));
	if (len <= 0)
		return G_SOURCE_CONTINUE;

	const struct inotify_event *event;

	for (char *ptr = buf; ptr < buf + len;
	     ptr += sizeof(struct inotify_event) + event->len) {
		event = (const struct inotify_event *)ptr;

		if (event->len == 0)
			continue;

		const char *dir_rel = g_hash_table_lookup(
			eng->wd_to_path, GINT_TO_POINTER(event->wd));
		if (!dir_rel)
			continue;

		char *rel_path;
		if (dir_rel[0] == '\0')
			rel_path = g_strdup(event->name);
		else
			rel_path = g_strdup_printf("%s/%s",
						   dir_rel, event->name);

		if (sync_suppress_check_and_remove(eng, rel_path)) {
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: suppressed event for %s\n",
				   rel_path);
			g_free(rel_path);
			continue;
		}

		char *abs_path = sync_shared_path(eng, rel_path);
		struct stat lst;
		bool exists = (lstat(abs_path, &lst) == 0);

		if (exists && S_ISLNK(lst.st_mode)) {
			g_free(abs_path);
			g_free(rel_path);
			continue;
		}

		if (event->mask & IN_CLOSE_WRITE) {
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: CLOSE_WRITE %s\n", rel_path);
			inotify_send_file_changed(eng, rel_path, false);

		} else if ((event->mask & IN_CREATE) &&
			   (event->mask & IN_ISDIR)) {
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: CREATE dir %s\n", rel_path);
			inotify_send_file_changed(eng, rel_path, true);

			inotify_add_watch_recursive(eng, abs_path, rel_path);

		} else if ((event->mask & IN_CREATE) &&
			   !(event->mask & IN_ISDIR)) {

		} else if ((event->mask & IN_DELETE) &&
			   !(event->mask & IN_ISDIR)) {
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: DELETE file %s\n", rel_path);
			inotify_send_file_removed(eng, rel_path, false);

		} else if ((event->mask & IN_DELETE) &&
			   (event->mask & IN_ISDIR)) {
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: DELETE dir %s\n", rel_path);
			inotify_send_file_removed(eng, rel_path, true);

			inotify_remove_watch_for_path(eng, rel_path);

		} else if (event->mask & IN_MOVED_FROM) {
			bool is_dir = (event->mask & IN_ISDIR) != 0;
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: MOVED_FROM %s%s\n",
				   rel_path, is_dir ? " (dir)" : "");
			inotify_send_file_removed(eng, rel_path, is_dir);

			if (is_dir)
				inotify_remove_watch_for_path(eng, rel_path);

		} else if (event->mask & IN_MOVED_TO) {
			bool is_dir = (event->mask & IN_ISDIR) != 0;
			g_printerr(SYNC_LOG_PREFIX
				   "inotify: MOVED_TO %s%s\n",
				   rel_path, is_dir ? " (dir)" : "");
			inotify_send_file_changed(eng, rel_path, is_dir);

			if (is_dir)
				inotify_add_watch_recursive(eng, abs_path,
							    rel_path);
		}

		g_free(abs_path);
		g_free(rel_path);
	}

	return G_SOURCE_CONTINUE;
}

static int inotify_start(struct odl_sync_engine *eng)
{
	eng->inotify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
	if (eng->inotify_fd < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "inotify: init failed: %s\n", strerror(errno));
		return -errno;
	}

	eng->wd_to_path = g_hash_table_new_full(g_direct_hash,
						g_direct_equal,
						NULL, g_free);
	eng->path_to_wd = g_hash_table_new_full(g_str_hash, g_str_equal,
						g_free, NULL);

	inotify_add_watch_recursive(eng, eng->shared_folder, "");

	eng->inotify_channel = g_io_channel_unix_new(eng->inotify_fd);
	g_io_channel_set_encoding(eng->inotify_channel, NULL, NULL);
	g_io_channel_set_buffered(eng->inotify_channel, FALSE);

	eng->inotify_watch_id = g_io_add_watch(eng->inotify_channel,
						G_IO_IN | G_IO_PRI,
						inotify_io_cb, eng);

	g_printerr(SYNC_LOG_PREFIX "inotify: watching %s (recursive)\n",
		   eng->shared_folder);

	return 0;
}

static void inotify_stop(struct odl_sync_engine *eng)
{
	if (eng->inotify_watch_id > 0) {
		g_source_remove(eng->inotify_watch_id);
		eng->inotify_watch_id = 0;
	}

	if (eng->inotify_channel) {
		g_io_channel_unref(eng->inotify_channel);
		eng->inotify_channel = NULL;
	}

	if (eng->inotify_fd >= 0) {
		close(eng->inotify_fd);
		eng->inotify_fd = -1;
	}

	if (eng->wd_to_path) {
		g_hash_table_destroy(eng->wd_to_path);
		eng->wd_to_path = NULL;
	}

	if (eng->path_to_wd) {
		g_hash_table_destroy(eng->path_to_wd);
		eng->path_to_wd = NULL;
	}

	g_printerr(SYNC_LOG_PREFIX "inotify: stopped\n");
}

static void sync_exchange_catalog(struct odl_sync_engine *eng)
{
	GList *entries;
	GList *l;

	g_printerr(SYNC_LOG_PREFIX "exchanging catalog with peer\n");

	g_mutex_lock(&eng->tx_lock);

	int ret = odl_sync_send_sync_req(eng->handle, eng->sid,
					 ODL_STREAM_SYNC, &eng->send_seq);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "catalog: failed to send SYNC_REQ: %s\n",
			   strerror(-ret));
		g_mutex_unlock(&eng->tx_lock);
		return;
	}

	entries = odl_catalog_list_dir("");

	for (l = entries; l != NULL; l = l->next) {
		const struct odl_catalog_entry *ent = l->data;

		if (ent->location != ODL_FILE_LOCAL &&
		    ent->location != ODL_FILE_BOTH)
			continue;

		odl_sync_send_listing_entry(eng->handle, eng->sid,
					    ODL_STREAM_SYNC, &eng->send_seq,
					    ent->rel_path, ent->file_size,
					    ent->mtime_ns, ent->mode,
					    ent->is_dir, ent->sha256);
	}

	odl_sync_send_listing_end(eng->handle, eng->sid,
				  ODL_STREAM_SYNC, &eng->send_seq);

	g_mutex_unlock(&eng->tx_lock);

	g_list_free(entries);

	sync_update_last_sync_time();

	g_printerr(SYNC_LOG_PREFIX "catalog exchange sent\n");
}

static gboolean sync_catalog_timer_cb(gpointer data)
{
	struct odl_sync_engine *eng = data;

	if (!eng->enabled)
		return G_SOURCE_REMOVE;

	sync_exchange_catalog(eng);
	return G_SOURCE_CONTINUE;
}

static void sync_respond_sysinfo(struct odl_sync_engine *eng)
{
	struct odl_sysinfo si;
	struct odl_cli_sysinfo_payload payload;

	odl_daemon_sysinfo_collect(&si);

	memset(&payload, 0, sizeof(payload));
	payload.num_cpus = (uint32_t)si.num_cpus;
	payload.ram_total_mb = si.ram_total_mb;
	payload.ram_available_mb = si.ram_available_mb;
	payload.num_gpus = (uint32_t)si.num_gpus;

	for (int i = 0; i < si.num_cpus && i < ODL_SYSINFO_MAX_CPUS_WIRE; i++) {
		memcpy(payload.cpus[i].model, si.cpus[i].model,
		       sizeof(payload.cpus[i].model));
		payload.cpus[i].cores = si.cpus[i].cores;
		payload.cpus[i].threads = si.cpus[i].threads;
		payload.cpus[i].freq_mhz = si.cpus[i].freq_mhz;
	}

	for (int i = 0; i < si.num_gpus && i < ODL_SYSINFO_MAX_GPUS_WIRE; i++) {
		memcpy(payload.gpus[i].name, si.gpus[i].name,
		       sizeof(payload.gpus[i].name));
		payload.gpus[i].vram_total_mb = si.gpus[i].vram_total_mb;
		payload.gpus[i].vram_used_mb = si.gpus[i].vram_used_mb;
	}

	g_mutex_lock(&eng->tx_lock);
	odl_cli_send_msg(eng->handle, eng->sid, ODL_STREAM_SYNC,
			 ODL_CLI_MSG_SYSINFO_RESP, 0,
			 &payload, sizeof(payload));
	g_mutex_unlock(&eng->tx_lock);

	g_printerr(SYNC_LOG_PREFIX "responded to SYSINFO_REQ from peer\n");
}

static void sync_deliver_sysinfo_resp(struct odl_sync_engine *eng,
				      const void *buf, size_t len)
{
	g_mutex_lock(&eng->sysinfo_lock);
	if (!eng->sysinfo_pending || !eng->sysinfo_out) {
		g_mutex_unlock(&eng->sysinfo_lock);
		g_printerr(SYNC_LOG_PREFIX
			   "got SYSINFO_RESP but no pending request\n");
		return;
	}

	if (len < sizeof(struct odl_cli_header) +
		   sizeof(struct odl_cli_sysinfo_payload)) {
		eng->sysinfo_result = -EPROTO;
		eng->sysinfo_pending = false;
		eng->sysinfo_out = NULL;
		g_cond_signal(&eng->sysinfo_cond);
		g_mutex_unlock(&eng->sysinfo_lock);
		g_printerr(SYNC_LOG_PREFIX "SYSINFO_RESP too short\n");
		return;
	}

	const struct odl_cli_sysinfo_payload *p =
		(const struct odl_cli_sysinfo_payload *)
		((const uint8_t *)buf + sizeof(struct odl_cli_header));
	struct odl_sysinfo *out = eng->sysinfo_out;

	memset(out, 0, sizeof(*out));
	out->num_cpus = (int)p->num_cpus;
	if (out->num_cpus > ODL_SYSINFO_MAX_CPUS)
		out->num_cpus = ODL_SYSINFO_MAX_CPUS;
	out->ram_total_mb = p->ram_total_mb;
	out->ram_available_mb = p->ram_available_mb;
	out->num_gpus = (int)p->num_gpus;
	if (out->num_gpus > ODL_SYSINFO_MAX_GPUS)
		out->num_gpus = ODL_SYSINFO_MAX_GPUS;

	for (int i = 0; i < out->num_cpus; i++) {
		memcpy(out->cpus[i].model, p->cpus[i].model,
		       sizeof(out->cpus[i].model));
		out->cpus[i].model[sizeof(out->cpus[i].model) - 1] = '\0';
		out->cpus[i].cores = p->cpus[i].cores;
		out->cpus[i].threads = p->cpus[i].threads;
		out->cpus[i].freq_mhz = p->cpus[i].freq_mhz;
	}

	for (int i = 0; i < out->num_gpus; i++) {
		memcpy(out->gpus[i].name, p->gpus[i].name,
		       sizeof(out->gpus[i].name));
		out->gpus[i].name[sizeof(out->gpus[i].name) - 1] = '\0';
		out->gpus[i].vram_total_mb = p->gpus[i].vram_total_mb;
		out->gpus[i].vram_used_mb = p->gpus[i].vram_used_mb;
	}

	eng->sysinfo_result = 0;
	eng->sysinfo_pending = false;
	eng->sysinfo_out = NULL;
	g_cond_signal(&eng->sysinfo_cond);
	g_mutex_unlock(&eng->sysinfo_lock);

	g_printerr(SYNC_LOG_PREFIX "delivered SYSINFO_RESP to requester\n");
}

static void sync_handle_cli_msg(struct odl_sync_engine *eng,
				const void *buf, size_t len)
{
	if (len < sizeof(struct odl_cli_header))
		return;

	const struct odl_cli_header *hdr = buf;

	switch (hdr->type) {
	case ODL_CLI_MSG_SYSINFO_REQ:
		sync_respond_sysinfo(eng);
		break;
	case ODL_CLI_MSG_SYSINFO_RESP:
		sync_deliver_sysinfo_resp(eng, buf, len);
		break;
	default:
		g_printerr(SYNC_LOG_PREFIX
			   "receiver: ignoring CLI msg type 0x%x\n",
			   hdr->type);
		break;
	}
}

static void sync_dispatch_msg(struct odl_sync_engine *eng,
			      uint32_t msg_type, void *buf)
{
	switch (msg_type) {
	case ODL_SYNC_MSG_SYNC_REQ:
		sync_handle_sync_req(eng);
		break;
	case ODL_SYNC_MSG_LISTING_ENTRY:
		sync_handle_listing_entry(eng,
			(const struct odl_sync_listing_entry *)buf);
		break;
	case ODL_SYNC_MSG_LISTING_END:
		sync_handle_listing_end(eng);
		break;
	case ODL_SYNC_MSG_FETCH_REQ:
		sync_handle_fetch_req(eng,
			(const struct odl_sync_fetch_req *)buf);
		break;
	case ODL_SYNC_MSG_FETCH_RESP:
		sync_handle_fetch_resp(eng,
			(const struct odl_sync_fetch_resp *)buf);
		break;
	case ODL_SYNC_MSG_FILE_DATA:
		sync_handle_file_data(eng, buf);
		break;
	case ODL_SYNC_MSG_FILE_META:
		sync_handle_file_meta(eng,
			(const struct odl_sync_file_meta *)buf);
		break;
	case ODL_SYNC_MSG_FILE_ACK:
		sync_handle_file_ack(eng,
			(const struct odl_sync_file_ack *)buf);
		break;
	case ODL_SYNC_MSG_FILE_DELETE:
		sync_handle_file_delete(eng,
			(const struct odl_sync_file_delete *)buf);
		break;
	case ODL_SYNC_MSG_DIR_CREATE:
		sync_handle_dir_create(eng,
			(const struct odl_sync_dir_op *)buf);
		break;
	case ODL_SYNC_MSG_DIR_DELETE:
		sync_handle_dir_delete(eng,
			(const struct odl_sync_dir_op *)buf);
		break;
	case ODL_SYNC_MSG_REMOVE_REQ:
		sync_handle_remove_req(eng,
			(const struct odl_sync_remove_req *)buf);
		break;
	case ODL_SYNC_MSG_REMOVE_ACK:
		sync_handle_remove_ack(eng,
			(const struct odl_sync_remove_ack *)buf);
		break;
	case ODL_SYNC_MSG_FILE_CHANGED:
		sync_handle_file_changed(eng,
			(const struct odl_sync_file_changed *)buf);
		break;
	case ODL_SYNC_MSG_FILE_REMOVED:
		sync_handle_file_removed(eng,
			(const struct odl_sync_file_removed *)buf);
		break;
	default:
		g_printerr(SYNC_LOG_PREFIX
			   "receiver: unknown sync msg type 0x%02x\n",
			   msg_type);
		break;
	}
}

static gpointer sync_receiver_loop(gpointer data)
{
	struct odl_sync_engine *eng = data;

	eng->recv_tid = pthread_self();

	void *buf = g_malloc(RECV_BUF_SIZE);

	g_printerr(SYNC_LOG_PREFIX "receiver thread started\n");

	while (eng->recv_running) {
		uint8_t src_id;
		uint32_t actual_len;
		int ret;

		ret = odl_tb5_stream_wait_rx(eng->handle, eng->sid, 1000);
		if (ret < 0) {
			if (!eng->recv_running)
				break;
			if (ret == -EAGAIN || ret == -EINTR || ret == -ETIMEDOUT)
				continue;
			g_printerr(SYNC_LOG_PREFIX
				   "receiver: wait_rx failed: %s\n",
				   strerror(-ret));
			break;
		}

		ret = odl_tb5_stream_recv(eng->handle, eng->sid,
					  buf, RECV_BUF_SIZE,
					  &src_id, &actual_len);
		if (ret < 0) {
			if (!eng->recv_running)
				break;
			if (ret == -EAGAIN || ret == -EINTR || ret == -ETIMEDOUT)
				continue;
			g_printerr(SYNC_LOG_PREFIX
				   "receiver: recv failed: %s\n",
				   strerror(-ret));
			break;
		}

		if (actual_len < 4)
			continue;

		uint32_t magic = *(const uint32_t *)buf;

		if (magic == ODL_SYNC_MAGIC) {
			struct odl_sync_header *hdr =
				(struct odl_sync_header *)buf;
			if (actual_len < sizeof(*hdr))
				continue;
			sync_dispatch_msg(eng, hdr->type, buf);

		} else if (magic == ODL_CLI_MAGIC) {
			struct odl_cli_header *hdr =
				(struct odl_cli_header *)buf;
			if (actual_len < sizeof(*hdr))
				continue;
			sync_handle_cli_msg(eng, buf, actual_len);

		} else {
			g_printerr(SYNC_LOG_PREFIX
				   "receiver: unknown magic 0x%08x, "
				   "discarding\n", magic);
		}
	}

	g_free(buf);
	g_printerr(SYNC_LOG_PREFIX "receiver thread stopped\n");
	return NULL;
}

static void sync_handle_sync_req(struct odl_sync_engine *eng)
{
	g_printerr(SYNC_LOG_PREFIX "received SYNC_REQ from peer\n");

	sync_exchange_catalog(eng);
}

static void sync_handle_listing_entry(struct odl_sync_engine *eng,
				      const struct odl_sync_listing_entry *ent)
{
	struct odl_catalog_entry *ce = g_malloc0(sizeof(*ce));

	g_strlcpy(ce->rel_path, ent->rel_path, sizeof(ce->rel_path));
	ce->file_size = ent->file_size;
	ce->mtime_ns = ent->mtime_ns;
	ce->mode = ent->mode;
	ce->is_dir = (ent->is_dir != 0);
	memcpy(ce->sha256, ent->sha256, 32);
	ce->location = ODL_FILE_REMOTE;

	eng->peer_listing = g_list_prepend(eng->peer_listing, ce);
	eng->listing_in_progress = true;
}

static void sync_handle_listing_end(struct odl_sync_engine *eng)
{
	if (!eng->peer_listing && !eng->listing_in_progress) {
		g_printerr(SYNC_LOG_PREFIX
			   "LISTING_END with no entries collected\n");
		return;
	}

	guint count = g_list_length(eng->peer_listing);
	g_printerr(SYNC_LOG_PREFIX "received peer listing (%u entries)\n",
		   count);

	odl_catalog_update_remote(eng->peer_listing);

	for (GList *l = eng->peer_listing; l != NULL; l = l->next) {
		const struct odl_catalog_entry *ce = l->data;
		const struct odl_catalog_entry *existing =
			odl_catalog_lookup(ce->rel_path);

		if (existing &&
		    (existing->location == ODL_FILE_REMOTE ||
		     existing->location == ODL_FILE_CACHED)) {
			if (!ce->is_dir)
				sync_create_remote_symlink(eng, ce->rel_path);
		}
	}

	g_list_free_full(eng->peer_listing, g_free);
	eng->peer_listing = NULL;
	eng->listing_in_progress = false;

	sync_update_last_sync_time();
}

static void sync_handle_fetch_req(struct odl_sync_engine *eng,
				  const struct odl_sync_fetch_req *req)
{
	char rel_path[ODL_SYNC_PATH_MAX];
	g_strlcpy(rel_path, req->rel_path, sizeof(rel_path));

	g_printerr(SYNC_LOG_PREFIX "fetch_req: peer wants %s\n", rel_path);

	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);

	if (!ce || (ce->location != ODL_FILE_LOCAL &&
		    ce->location != ODL_FILE_BOTH)) {
		g_mutex_lock(&eng->tx_lock);
		odl_sync_send_fetch_resp(eng->handle, eng->sid,
					 ODL_STREAM_SYNC, &eng->send_seq,
					 rel_path, 0, 0, 0, 0,
					 ODL_FETCH_NOT_FOUND, NULL);
		g_mutex_unlock(&eng->tx_lock);
		g_printerr(SYNC_LOG_PREFIX
			   "fetch_req: %s not found locally\n", rel_path);
		return;
	}

	char *abs_path = sync_shared_path(eng, rel_path);
	struct stat st;

	if (stat(abs_path, &st) < 0 || !S_ISREG(st.st_mode)) {
		g_mutex_lock(&eng->tx_lock);
		odl_sync_send_fetch_resp(eng->handle, eng->sid,
					 ODL_STREAM_SYNC, &eng->send_seq,
					 rel_path, 0, 0, 0, 0,
					 ODL_FETCH_ERROR, NULL);
		g_mutex_unlock(&eng->tx_lock);
		g_printerr(SYNC_LOG_PREFIX
			   "fetch_req: cannot stat %s: %s\n",
			   rel_path, strerror(errno));
		g_free(abs_path);
		return;
	}

	uint64_t file_size = (uint64_t)st.st_size;
	uint64_t mtime_ns = sync_get_mtime_ns(abs_path);
	uint32_t mode = (uint32_t)st.st_mode & 07777;
	uint32_t num_chunks = (file_size > 0)
		? (uint32_t)((file_size + ODL_SYNC_CHUNK_SIZE - 1) /
			     ODL_SYNC_CHUNK_SIZE)
		: 1;

	uint8_t sha256[32];
	memset(sha256, 0, sizeof(sha256));
	if (file_size > 0)
		odl_sync_sha256_file(abs_path, sha256);

	int fd = open(abs_path, O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		g_mutex_lock(&eng->tx_lock);
		odl_sync_send_fetch_resp(eng->handle, eng->sid,
					 ODL_STREAM_SYNC, &eng->send_seq,
					 rel_path, 0, 0, 0, 0,
					 ODL_FETCH_ERROR, NULL);
		g_mutex_unlock(&eng->tx_lock);
		g_printerr(SYNC_LOG_PREFIX
			   "fetch_req: cannot open %s: %s\n",
			   rel_path, strerror(errno));
		g_free(abs_path);
		return;
	}

	g_mutex_lock(&eng->tx_lock);

	int ret = odl_sync_send_fetch_resp(eng->handle, eng->sid,
					   ODL_STREAM_SYNC, &eng->send_seq,
					   rel_path, file_size, mtime_ns,
					   mode, num_chunks,
					   ODL_FETCH_OK, sha256);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "fetch_req: send fetch_resp failed for %s\n",
			   rel_path);
		g_mutex_unlock(&eng->tx_lock);
		close(fd);
		g_free(abs_path);
		return;
	}

	uint8_t *chunk_buf = g_malloc(ODL_SYNC_CHUNK_SIZE);
	uint64_t bytes_sent = 0;

	for (uint32_t i = 0; i < num_chunks; i++) {
		uint32_t to_read = ODL_SYNC_CHUNK_SIZE;
		if (bytes_sent + to_read > file_size)
			to_read = (uint32_t)(file_size - bytes_sent);

		ssize_t nread = 0;
		if (to_read > 0) {
			nread = read(fd, chunk_buf, to_read);
			if (nread < 0) {
				g_printerr(SYNC_LOG_PREFIX
					   "fetch_req: read error %s: %s\n",
					   rel_path, strerror(errno));
				break;
			}
		}

		ret = odl_sync_send_file_data(eng->handle, eng->sid,
					      ODL_STREAM_SYNC, &eng->send_seq,
					      i, chunk_buf, (uint32_t)nread);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "fetch_req: send chunk %u failed for %s\n",
				   i, rel_path);
			break;
		}

		bytes_sent += (uint64_t)nread;
	}

	g_mutex_unlock(&eng->tx_lock);

	g_free(chunk_buf);
	close(fd);

	sync_status_add_bytes(bytes_sent);

	g_printerr(SYNC_LOG_PREFIX "fetch_req: served %s (%" G_GUINT64_FORMAT
		   " bytes)\n", rel_path, bytes_sent);

	g_free(abs_path);
}

static void sync_handle_fetch_resp(struct odl_sync_engine *eng,
				   const struct odl_sync_fetch_resp *resp)
{
	g_printerr(SYNC_LOG_PREFIX "fetch_resp: %s status=%u\n",
		   resp->rel_path, resp->status);

	g_mutex_lock(&eng->fetch_lock);

	if (!eng->fetch_in_progress ||
	    g_strcmp0(eng->fetch_pending_path, resp->rel_path) != 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "fetch_resp: unexpected response for %s\n",
			   resp->rel_path);
		g_mutex_unlock(&eng->fetch_lock);
		return;
	}

	if (resp->status != ODL_FETCH_OK) {
		eng->fetch_result = (resp->status == ODL_FETCH_NOT_FOUND)
			? -ENOENT : -EIO;
		eng->fetch_in_progress = false;
		g_cond_signal(&eng->fetch_cond);
		g_mutex_unlock(&eng->fetch_lock);
		return;
	}

	g_strlcpy(eng->recv_rel_path, resp->rel_path,
		  sizeof(eng->recv_rel_path));
	eng->recv_file_size = resp->file_size;
	eng->recv_mtime_ns = resp->mtime_ns;
	eng->recv_mode = resp->mode;
	eng->recv_num_chunks = resp->num_chunks;
	eng->recv_chunks_got = 0;
	memcpy(eng->recv_sha256, resp->sha256, 32);

	char *cache_path = sync_cache_path(eng, resp->rel_path);
	char *cache_parent = g_path_get_dirname(cache_path);
	g_mkdir_with_parents(cache_parent, 0755);
	g_free(cache_parent);
	g_free(cache_path);

	char *tmp_template = g_strdup_printf("%s/%sXXXXXX",
					     eng->cache_dir,
					     SYNC_TMP_PREFIX);
	eng->recv_tmp_fd = g_mkstemp(tmp_template);
	if (eng->recv_tmp_fd < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "fetch_resp: failed to create temp: %s\n",
			   strerror(errno));
		g_free(tmp_template);
		eng->recv_tmp_path = NULL;
		eng->fetch_result = -EIO;
		eng->fetch_in_progress = false;
		g_cond_signal(&eng->fetch_cond);
		g_mutex_unlock(&eng->fetch_lock);
		return;
	}
	eng->recv_tmp_path = tmp_template;

	g_printerr(SYNC_LOG_PREFIX
		   "fetch_resp: expecting %s (%" G_GUINT64_FORMAT
		   " bytes, %u chunks)\n",
		   resp->rel_path, resp->file_size, resp->num_chunks);

	if (resp->num_chunks == 0 || resp->file_size == 0) {
		close(eng->recv_tmp_fd);
		eng->recv_tmp_fd = -1;

		char *final_path = sync_cache_path(eng, eng->recv_rel_path);
		char *final_parent = g_path_get_dirname(final_path);
		g_mkdir_with_parents(final_parent, 0755);
		g_free(final_parent);

		g_rename(eng->recv_tmp_path, final_path);
		if (eng->recv_mode)
			g_chmod(final_path, eng->recv_mode);
		g_free(final_path);

		g_free(eng->recv_tmp_path);
		eng->recv_tmp_path = NULL;

		odl_catalog_mark_cached(eng->recv_rel_path);

		eng->fetch_result = 0;
		eng->fetch_in_progress = false;
		g_cond_signal(&eng->fetch_cond);
	}

	g_mutex_unlock(&eng->fetch_lock);
}

static void sync_handle_file_data(struct odl_sync_engine *eng,
				  const void *buf)
{
	const struct odl_sync_file_data *data = buf;

	g_mutex_lock(&eng->fetch_lock);

	if (eng->recv_tmp_fd < 0 || !eng->recv_tmp_path) {
		g_printerr(SYNC_LOG_PREFIX
			   "file_data: no active fetch transfer\n");
		g_mutex_unlock(&eng->fetch_lock);
		return;
	}

	const uint8_t *chunk_data = (const uint8_t *)buf +
				    sizeof(struct odl_sync_file_data);
	uint32_t chunk_len = data->chunk_len;

	if (chunk_len > 0) {
		ssize_t nw = write(eng->recv_tmp_fd, chunk_data, chunk_len);
		if (nw < 0 || (uint32_t)nw != chunk_len) {
			g_printerr(SYNC_LOG_PREFIX
				   "file_data: write error chunk %u: %s\n",
				   data->chunk_index, strerror(errno));
			close(eng->recv_tmp_fd);
			eng->recv_tmp_fd = -1;
			if (eng->recv_tmp_path) {
				g_unlink(eng->recv_tmp_path);
				g_free(eng->recv_tmp_path);
				eng->recv_tmp_path = NULL;
			}
			eng->fetch_result = -EIO;
			eng->fetch_in_progress = false;
			g_cond_signal(&eng->fetch_cond);
			g_mutex_unlock(&eng->fetch_lock);
			return;
		}
	}

	eng->recv_chunks_got++;
	sync_status_add_bytes((uint64_t)chunk_len);

	g_printerr(SYNC_LOG_PREFIX "file_data: chunk %u/%u for %s (%u bytes)\n",
		   eng->recv_chunks_got, eng->recv_num_chunks,
		   eng->recv_rel_path, chunk_len);

	if (eng->recv_chunks_got >= eng->recv_num_chunks) {
		close(eng->recv_tmp_fd);
		eng->recv_tmp_fd = -1;

		char *final_path = sync_cache_path(eng, eng->recv_rel_path);
		char *final_parent = g_path_get_dirname(final_path);
		g_mkdir_with_parents(final_parent, 0755);
		g_free(final_parent);

		if (g_rename(eng->recv_tmp_path, final_path) < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "file_data: rename failed: %s\n",
				   strerror(errno));
			g_unlink(eng->recv_tmp_path);
			g_free(final_path);
			g_free(eng->recv_tmp_path);
			eng->recv_tmp_path = NULL;
			eng->fetch_result = -EIO;
			eng->fetch_in_progress = false;
			g_cond_signal(&eng->fetch_cond);
			g_mutex_unlock(&eng->fetch_lock);
			return;
		}

		if (eng->recv_mode)
			g_chmod(final_path, eng->recv_mode);

		struct timespec times[2];
		times[0].tv_sec = 0;
		times[0].tv_nsec = UTIME_OMIT;
		times[1].tv_sec = (time_t)(eng->recv_mtime_ns / 1000000000ULL);
		times[1].tv_nsec = (long)(eng->recv_mtime_ns % 1000000000ULL);
		utimensat(AT_FDCWD, final_path, times, 0);

		g_free(final_path);
		g_free(eng->recv_tmp_path);
		eng->recv_tmp_path = NULL;

		odl_catalog_mark_cached(eng->recv_rel_path);

		odl_daemon_fuse_invalidate(eng->recv_rel_path);

		sync_status_add_bytes(eng->recv_file_size);

		g_printerr(SYNC_LOG_PREFIX
			   "file_data: fetch complete for %s\n",
			   eng->recv_rel_path);

		if (eng->fetch_in_progress) {
			eng->fetch_result = 0;
			eng->fetch_in_progress = false;
			g_cond_signal(&eng->fetch_cond);
		}

		odl_daemon_dbus_emit_sync_file_transferred(
			eng->recv_rel_path, "fetched", eng->recv_file_size);
	}

	g_mutex_unlock(&eng->fetch_lock);
}

static void sync_handle_file_meta(struct odl_sync_engine *eng,
				  const struct odl_sync_file_meta *meta)
{
	g_printerr(SYNC_LOG_PREFIX "file_meta: incoming push %s "
		   "(%" G_GUINT64_FORMAT " bytes)\n",
		   meta->rel_path, meta->file_size);

	g_mutex_lock(&eng->fetch_lock);

	g_strlcpy(eng->recv_rel_path, meta->rel_path,
		  sizeof(eng->recv_rel_path));
	eng->recv_file_size = meta->file_size;
	eng->recv_mtime_ns = meta->mtime_ns;
	eng->recv_mode = meta->mode;
	eng->recv_num_chunks = meta->num_chunks;
	eng->recv_chunks_got = 0;
	memcpy(eng->recv_sha256, meta->sha256, 32);

	sync_suppress_add(eng, meta->rel_path);

	char *tmp_template = g_strdup_printf("%s/%sXXXXXX",
					     eng->shared_folder,
					     SYNC_TMP_PREFIX);
	eng->recv_tmp_fd = g_mkstemp(tmp_template);
	if (eng->recv_tmp_fd < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "file_meta: failed to create temp: %s\n",
			   strerror(errno));
		g_free(tmp_template);
		eng->recv_tmp_path = NULL;
		g_mutex_unlock(&eng->fetch_lock);
		return;
	}
	eng->recv_tmp_path = tmp_template;

	g_mutex_unlock(&eng->fetch_lock);
}

static void sync_handle_file_ack(struct odl_sync_engine *eng,
				 const struct odl_sync_file_ack *ack)
{
	(void)eng;

	const char *status_str;
	switch (ack->status) {
	case ODL_SYNC_ACK_OK:       status_str = "OK";       break;
	case ODL_SYNC_ACK_CONFLICT: status_str = "CONFLICT"; break;
	case ODL_SYNC_ACK_ERROR:    status_str = "ERROR";    break;
	case ODL_SYNC_ACK_REJECTED: status_str = "REJECTED"; break;
	default:                    status_str = "UNKNOWN";   break;
	}

	g_printerr(SYNC_LOG_PREFIX "file_ack: %s -> %s\n",
		   ack->rel_path, status_str);

	if (ack->status == ODL_SYNC_ACK_OK) {
		odl_catalog_mark_both(ack->rel_path);
		odl_daemon_dbus_emit_sync_file_transferred(
			ack->rel_path, "transferred", 0);
	}
}

static void sync_handle_file_delete(struct odl_sync_engine *eng,
				    const struct odl_sync_file_delete *del)
{
	g_printerr(SYNC_LOG_PREFIX "file_delete: peer deleted %s\n",
		   del->rel_path);

	const struct odl_catalog_entry *ce = odl_catalog_lookup(del->rel_path);
	if (!ce)
		return;

	if (ce->location == ODL_FILE_REMOTE) {
		sync_remove_remote_symlink(eng, del->rel_path);
		odl_catalog_remove(del->rel_path);
	} else if (ce->location == ODL_FILE_BOTH) {
		struct odl_catalog_entry updated = *ce;
		updated.location = ODL_FILE_LOCAL;
		odl_catalog_set(del->rel_path, &updated);
	} else if (ce->location == ODL_FILE_CACHED) {
		char *cache_path = sync_cache_path(eng, del->rel_path);
		g_unlink(cache_path);
		g_free(cache_path);
		sync_remove_remote_symlink(eng, del->rel_path);
		odl_catalog_remove(del->rel_path);
	}
}

static void sync_handle_dir_create(struct odl_sync_engine *eng,
				   const struct odl_sync_dir_op *op)
{
	char *abs_path = sync_shared_path(eng, op->rel_path);

	sync_suppress_add(eng, op->rel_path);

	int ret = g_mkdir_with_parents(abs_path, op->mode ? op->mode : 0755);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "dir_create: failed for %s: %s\n",
			   op->rel_path, strerror(errno));
	} else {
		g_printerr(SYNC_LOG_PREFIX "dir_create: %s\n", op->rel_path);

		struct odl_catalog_entry ce = {0};
		g_strlcpy(ce.rel_path, op->rel_path, sizeof(ce.rel_path));
		ce.is_dir = true;
		ce.mode = op->mode ? op->mode : 0755;
		ce.mtime_ns = op->mtime_ns;
		ce.location = ODL_FILE_BOTH;
		odl_catalog_set(op->rel_path, &ce);

		if (eng->inotify_fd >= 0)
			inotify_add_watch_recursive(eng, abs_path,
						    op->rel_path);
	}

	g_free(abs_path);
}

static void sync_handle_dir_delete(struct odl_sync_engine *eng,
				   const struct odl_sync_dir_op *op)
{
	char *abs_path = sync_shared_path(eng, op->rel_path);

	sync_suppress_add(eng, op->rel_path);

	if (g_rmdir(abs_path) < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "dir_delete: failed for %s: %s\n",
			   op->rel_path, strerror(errno));
	} else {
		g_printerr(SYNC_LOG_PREFIX "dir_delete: %s\n", op->rel_path);
		odl_catalog_remove(op->rel_path);

		if (eng->inotify_fd >= 0)
			inotify_remove_watch_for_path(eng, op->rel_path);
	}

	g_free(abs_path);
}

static void sync_handle_remove_req(struct odl_sync_engine *eng,
				   const struct odl_sync_remove_req *req)
{
	g_printerr(SYNC_LOG_PREFIX "remove_req: peer asks us to remove %s\n",
		   req->rel_path);

	const struct odl_catalog_entry *ce = odl_catalog_lookup(req->rel_path);

	if (!ce) {
		g_mutex_lock(&eng->tx_lock);
		odl_sync_send_remove_ack(eng->handle, eng->sid,
					 ODL_STREAM_SYNC, &eng->send_seq,
					 req->rel_path, 1);
		g_mutex_unlock(&eng->tx_lock);
		return;
	}

	sync_suppress_add(eng, req->rel_path);

	int ret = odl_catalog_remove_local_copy(req->rel_path);

	uint32_t ack_status = (ret == 0) ? 0 : 2;

	g_mutex_lock(&eng->tx_lock);
	odl_sync_send_remove_ack(eng->handle, eng->sid,
				 ODL_STREAM_SYNC, &eng->send_seq,
				 req->rel_path, ack_status);
	g_mutex_unlock(&eng->tx_lock);

	if (ret == 0) {
		g_printerr(SYNC_LOG_PREFIX "remove_req: removed %s\n",
			   req->rel_path);
	} else {
		g_printerr(SYNC_LOG_PREFIX
			   "remove_req: failed to remove %s\n",
			   req->rel_path);
	}
}

static void sync_handle_remove_ack(struct odl_sync_engine *eng,
				   const struct odl_sync_remove_ack *ack)
{
	(void)eng;

	g_printerr(SYNC_LOG_PREFIX "remove_ack: %s status=%u\n",
		   ack->rel_path, ack->status);

	if (ack->status == 0) {
		const struct odl_catalog_entry *ce =
			odl_catalog_lookup(ack->rel_path);
		if (ce) {
			if (ce->location == ODL_FILE_BOTH) {
				struct odl_catalog_entry updated = *ce;
				updated.location = ODL_FILE_LOCAL;
				odl_catalog_set(ack->rel_path, &updated);
			} else if (ce->location == ODL_FILE_REMOTE) {
				sync_remove_remote_symlink(eng,
							   ack->rel_path);
				odl_catalog_remove(ack->rel_path);
			}
		}
	}
}

static void sync_handle_file_changed(struct odl_sync_engine *eng,
				     const struct odl_sync_file_changed *msg)
{
	g_printerr(SYNC_LOG_PREFIX "file_changed: peer reports %s %s "
		   "(%" G_GUINT64_FORMAT " bytes)\n",
		   msg->is_dir ? "dir" : "file",
		   msg->rel_path, msg->file_size);

	const struct odl_catalog_entry *existing =
		odl_catalog_lookup(msg->rel_path);

	struct odl_catalog_entry ce = {0};
	g_strlcpy(ce.rel_path, msg->rel_path, sizeof(ce.rel_path));
	ce.file_size = msg->file_size;
	ce.mtime_ns = msg->mtime_ns;
	ce.mode = msg->mode;
	ce.is_dir = (msg->is_dir != 0);
	memcpy(ce.sha256, msg->sha256, 32);

	if (existing &&
	    (existing->location == ODL_FILE_LOCAL ||
	     existing->location == ODL_FILE_BOTH)) {
		ce.location = ODL_FILE_BOTH;
	} else {
		ce.location = ODL_FILE_REMOTE;
	}

	odl_catalog_set(msg->rel_path, &ce);

	if (!msg->is_dir && ce.location == ODL_FILE_REMOTE)
		sync_create_remote_symlink(eng, msg->rel_path);

	if (msg->is_dir && ce.location == ODL_FILE_REMOTE) {
		char *abs_path = sync_shared_path(eng, msg->rel_path);

		sync_suppress_add(eng, msg->rel_path);
		g_mkdir_with_parents(abs_path, msg->mode ? msg->mode : 0755);

		if (eng->inotify_fd >= 0)
			inotify_add_watch_recursive(eng, abs_path,
						    msg->rel_path);

		g_free(abs_path);
	}

	odl_daemon_fuse_invalidate(msg->rel_path);
}

static void sync_handle_file_removed(struct odl_sync_engine *eng,
				     const struct odl_sync_file_removed *msg)
{
	g_printerr(SYNC_LOG_PREFIX "file_removed: peer removed %s%s\n",
		   msg->rel_path, msg->is_dir ? " (dir)" : "");

	const struct odl_catalog_entry *ce =
		odl_catalog_lookup(msg->rel_path);
	if (!ce)
		return;

	if (ce->location == ODL_FILE_REMOTE) {
		if (!msg->is_dir)
			sync_remove_remote_symlink(eng, msg->rel_path);
		else {
			char *abs_path = sync_shared_path(eng, msg->rel_path);
			sync_suppress_add(eng, msg->rel_path);
			g_rmdir(abs_path);
			if (eng->inotify_fd >= 0)
				inotify_remove_watch_for_path(eng,
							      msg->rel_path);
			g_free(abs_path);
		}
		odl_catalog_remove(msg->rel_path);
	} else if (ce->location == ODL_FILE_BOTH) {
		struct odl_catalog_entry updated = *ce;
		updated.location = ODL_FILE_LOCAL;
		odl_catalog_set(msg->rel_path, &updated);
	} else if (ce->location == ODL_FILE_CACHED) {
		char *cache_path = sync_cache_path(eng, msg->rel_path);
		g_unlink(cache_path);
		g_free(cache_path);
		sync_remove_remote_symlink(eng, msg->rel_path);
		odl_catalog_remove(msg->rel_path);
	}

	odl_daemon_fuse_invalidate(msg->rel_path);
}

static int fuse_cb_getattr(const char *rel_path, struct stat *st)
{
	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);
	if (!ce)
		return -ENOENT;

	memset(st, 0, sizeof(*st));

	if (ce->is_dir) {
		st->st_mode = S_IFDIR | (ce->mode ? ce->mode : 0755);
		st->st_nlink = 2;
	} else {
		st->st_mode = S_IFREG | (ce->mode ? ce->mode : 0644);
		st->st_nlink = 1;
		st->st_size = (off_t)ce->file_size;
	}

	st->st_mtim.tv_sec = (time_t)(ce->mtime_ns / 1000000000ULL);
	st->st_mtim.tv_nsec = (long)(ce->mtime_ns % 1000000000ULL);
	st->st_atim = st->st_mtim;
	st->st_ctim = st->st_mtim;

	return 0;
}

static int fuse_cb_readdir(const char *dir_path,
			   void (*add_entry)(const char *name,
					     const struct stat *st,
					     void *ctx),
			   void *ctx)
{
	GList *entries = odl_catalog_list_dir(dir_path);
	GList *l;

	for (l = entries; l != NULL; l = l->next) {
		const struct odl_catalog_entry *ce = l->data;

		const char *slash = strrchr(ce->rel_path, '/');
		const char *name = slash ? slash + 1 : ce->rel_path;

		struct stat st;
		memset(&st, 0, sizeof(st));

		if (ce->is_dir) {
			st.st_mode = S_IFDIR | (ce->mode ? ce->mode : 0755);
			st.st_nlink = 2;
		} else {
			st.st_mode = S_IFREG | (ce->mode ? ce->mode : 0644);
			st.st_nlink = 1;
			st.st_size = (off_t)ce->file_size;
		}

		st.st_mtim.tv_sec =
			(time_t)(ce->mtime_ns / 1000000000ULL);
		st.st_mtim.tv_nsec =
			(long)(ce->mtime_ns % 1000000000ULL);

		add_entry(name, &st, ctx);
	}

	g_list_free(entries);
	return 0;
}

static char *fuse_cb_get_local_path(const char *rel_path)
{
	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);
	if (!ce)
		return NULL;

	switch (ce->location) {
	case ODL_FILE_LOCAL:
	case ODL_FILE_BOTH:
		return g_build_filename(g_engine.shared_folder,
					rel_path, NULL);

	case ODL_FILE_CACHED:
		return g_build_filename(g_engine.cache_dir,
					rel_path, NULL);

	case ODL_FILE_REMOTE:
	default:
		return NULL;
	}
}

static int fuse_cb_fetch_remote(const char *rel_path)
{
	return odl_daemon_sync_fetch_file(rel_path);
}

static int fuse_cb_create_file(const char *rel_path, mode_t mode)
{
	char *abs_path = sync_shared_path(&g_engine, rel_path);
	char *parent = g_path_get_dirname(abs_path);

	g_mkdir_with_parents(parent, 0755);
	g_free(parent);

	int fd = open(abs_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
		      mode);
	if (fd < 0) {
		int err = errno;
		g_printerr(SYNC_LOG_PREFIX
			   "create_file: failed %s: %s\n",
			   rel_path, strerror(err));
		g_free(abs_path);
		return -err;
	}

	struct odl_catalog_entry ce = {0};
	g_strlcpy(ce.rel_path, rel_path, sizeof(ce.rel_path));
	ce.mode = (uint32_t)mode;
	ce.is_dir = false;
	ce.location = ODL_FILE_LOCAL;

	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	ce.mtime_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
		      (uint64_t)ts.tv_nsec;

	odl_catalog_set(rel_path, &ce);

	g_free(abs_path);
	return fd;
}

static int fuse_cb_delete_file(const char *rel_path)
{
	char *abs_path = sync_shared_path(&g_engine, rel_path);
	int ret = 0;

	if (g_unlink(abs_path) < 0 && errno != ENOENT)
		ret = -errno;

	char *cache_path = sync_cache_path(&g_engine, rel_path);
	g_unlink(cache_path);
	g_free(cache_path);

	odl_catalog_remove(rel_path);

	g_free(abs_path);
	return ret;
}

static int fuse_cb_create_dir(const char *rel_path, mode_t mode)
{
	char *abs_path = sync_shared_path(&g_engine, rel_path);

	int ret = g_mkdir_with_parents(abs_path, mode);
	if (ret < 0) {
		int err = errno;
		g_printerr(SYNC_LOG_PREFIX
			   "create_dir: failed %s: %s\n",
			   rel_path, strerror(err));
		g_free(abs_path);
		return -err;
	}

	struct odl_catalog_entry ce = {0};
	g_strlcpy(ce.rel_path, rel_path, sizeof(ce.rel_path));
	ce.is_dir = true;
	ce.mode = (uint32_t)mode;
	ce.location = ODL_FILE_LOCAL;

	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	ce.mtime_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
		      (uint64_t)ts.tv_nsec;

	odl_catalog_set(rel_path, &ce);

	g_free(abs_path);
	return 0;
}

static int fuse_cb_delete_dir(const char *rel_path)
{
	char *abs_path = sync_shared_path(&g_engine, rel_path);

	int ret = 0;
	if (g_rmdir(abs_path) < 0 && errno != ENOENT)
		ret = -errno;

	odl_catalog_remove(rel_path);

	g_free(abs_path);
	return ret;
}

static void sync_scan_shared_folder_recursive(struct odl_sync_engine *eng,
					      const char *abs_dir,
					      const char *rel_dir)
{
	DIR *d = opendir(abs_dir);
	if (!d)
		return;

	struct dirent *de;
	while ((de = readdir(d)) != NULL) {
		if (g_strcmp0(de->d_name, ".") == 0 ||
		    g_strcmp0(de->d_name, "..") == 0)
			continue;

		if (g_str_has_prefix(de->d_name, SYNC_TMP_PREFIX))
			continue;

		char *child_abs = g_build_filename(abs_dir, de->d_name, NULL);

		struct stat lst;
		if (lstat(child_abs, &lst) < 0) {
			g_free(child_abs);
			continue;
		}

		if (S_ISLNK(lst.st_mode)) {
			g_free(child_abs);
			continue;
		}

		char *child_rel;
		if (rel_dir[0] == '\0')
			child_rel = g_strdup(de->d_name);
		else
			child_rel = g_strdup_printf("%s/%s",
						    rel_dir, de->d_name);

		if (S_ISDIR(lst.st_mode)) {
			struct odl_catalog_entry ce = {0};
			g_strlcpy(ce.rel_path, child_rel,
				  sizeof(ce.rel_path));
			ce.is_dir = true;
			ce.mode = (uint32_t)lst.st_mode & 07777;
			ce.mtime_ns =
				(uint64_t)lst.st_mtim.tv_sec * 1000000000ULL +
				(uint64_t)lst.st_mtim.tv_nsec;
			ce.location = ODL_FILE_LOCAL;
			odl_catalog_set(child_rel, &ce);

			sync_scan_shared_folder_recursive(eng, child_abs,
							  child_rel);

		} else if (S_ISREG(lst.st_mode)) {
			struct odl_catalog_entry ce = {0};
			g_strlcpy(ce.rel_path, child_rel,
				  sizeof(ce.rel_path));
			ce.file_size = (uint64_t)lst.st_size;
			ce.mtime_ns =
				(uint64_t)lst.st_mtim.tv_sec * 1000000000ULL +
				(uint64_t)lst.st_mtim.tv_nsec;
			ce.mode = (uint32_t)lst.st_mode & 07777;
			ce.is_dir = false;
			ce.location = ODL_FILE_LOCAL;

			if (ce.file_size > 0)
				odl_sync_sha256_file(child_abs, ce.sha256);

			odl_catalog_set(child_rel, &ce);
		}

		g_free(child_rel);
		g_free(child_abs);
	}

	closedir(d);
}

int odl_daemon_sync_init(const char *folder_path)
{
	(void)folder_path;

	memset(&g_engine, 0, sizeof(g_engine));

	g_mutex_init(&g_sync_status.lock);
	g_sync_status.enabled = false;
	g_sync_status.files_pending = 0;
	g_sync_status.bytes_transferred = 0;
	snprintf(g_sync_status.last_sync_time,
		 sizeof(g_sync_status.last_sync_time), "never");

	const char *home = g_get_home_dir();

	char *shared_folder = g_build_filename(home, "OdinLink", NULL);
	g_strlcpy(g_engine.shared_folder, shared_folder,
		  sizeof(g_engine.shared_folder));
	g_free(shared_folder);

	char *fuse_mount = g_build_filename(home, ".odl_tb5", "remote", NULL);
	g_strlcpy(g_engine.fuse_mount, fuse_mount,
		  sizeof(g_engine.fuse_mount));
	g_free(fuse_mount);

	char *cache_dir = g_build_filename(home, ".cache", "odl_tb5",
					   "files", NULL);
	g_strlcpy(g_engine.cache_dir, cache_dir,
		  sizeof(g_engine.cache_dir));
	g_free(cache_dir);

	g_mkdir_with_parents(g_engine.shared_folder, 0755);

	char *odl_dir = g_build_filename(home, ".odl_tb5", NULL);
	struct stat st;
	if (stat(odl_dir, &st) == 0 && !S_ISDIR(st.st_mode)) {
		g_printerr(SYNC_LOG_PREFIX
			   "%s exists but is not a directory, removing\n",
			   odl_dir);
		g_unlink(odl_dir);
	}
	g_free(odl_dir);

	g_mkdir_with_parents(g_engine.fuse_mount, 0755);
	g_mkdir_with_parents(g_engine.cache_dir, 0755);

	g_mutex_init(&g_engine.tx_lock);
	g_mutex_init(&g_engine.fetch_lock);
	g_cond_init(&g_engine.fetch_cond);
	g_mutex_init(&g_engine.suppress_lock);

	g_engine.handle = NULL;
	g_engine.device_index = 0;
	g_engine.recv_thread = NULL;
	g_engine.recv_running = false;
	g_engine.recv_tid = 0;
	g_engine.send_seq = 0;

	g_mutex_init(&g_engine.sysinfo_lock);
	g_cond_init(&g_engine.sysinfo_cond);
	g_engine.sysinfo_pending = false;
	g_engine.sysinfo_result = 0;
	g_engine.sysinfo_out = NULL;
	g_engine.enabled = false;
	g_engine.recv_tmp_fd = -1;
	g_engine.recv_tmp_path = NULL;
	g_engine.peer_listing = NULL;
	g_engine.listing_in_progress = false;
	g_engine.catalog_timer_id = 0;
	g_engine.fetch_in_progress = false;
	g_engine.fetch_result = 0;
	g_engine.inotify_fd = -1;
	g_engine.inotify_watch_id = 0;
	g_engine.inotify_channel = NULL;
	g_engine.wd_to_path = NULL;
	g_engine.path_to_wd = NULL;

	g_engine.suppress_set = g_hash_table_new_full(g_str_hash, g_str_equal,
						      g_free, NULL);

	g_engine.status = &g_sync_status;

	int ret = odl_catalog_init(g_engine.shared_folder, g_engine.cache_dir);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "failed to init catalog: %s\n", strerror(-ret));
		return ret;
	}

	static const struct odl_fuse_callbacks fuse_cbs = {
		.getattr        = fuse_cb_getattr,
		.readdir        = fuse_cb_readdir,
		.get_local_path = fuse_cb_get_local_path,
		.fetch_remote   = fuse_cb_fetch_remote,
		.create_file    = fuse_cb_create_file,
		.delete_file    = fuse_cb_delete_file,
		.create_dir     = fuse_cb_create_dir,
		.delete_dir     = fuse_cb_delete_dir,
	};
	odl_daemon_fuse_set_callbacks(&fuse_cbs);

	ret = odl_daemon_fuse_init(g_engine.fuse_mount);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "failed to init FUSE mount at %s: %s\n",
			   g_engine.fuse_mount, strerror(-ret));
	}

	g_engine_initialized = true;

	g_printerr(SYNC_LOG_PREFIX "initialized\n");
	g_printerr(SYNC_LOG_PREFIX "  shared_folder: %s\n",
		   g_engine.shared_folder);
	g_printerr(SYNC_LOG_PREFIX "  fuse_mount:    %s\n",
		   g_engine.fuse_mount);
	g_printerr(SYNC_LOG_PREFIX "  cache_dir:     %s\n",
		   g_engine.cache_dir);

	return 0;
}

void odl_daemon_sync_shutdown(void)
{
	if (!g_engine_initialized)
		return;

	odl_daemon_sync_set_enabled(false);

	odl_daemon_fuse_shutdown();

	odl_catalog_shutdown();

	if (g_engine.recv_tmp_fd >= 0) {
		close(g_engine.recv_tmp_fd);
		g_engine.recv_tmp_fd = -1;
	}
	if (g_engine.recv_tmp_path) {
		g_unlink(g_engine.recv_tmp_path);
		g_free(g_engine.recv_tmp_path);
		g_engine.recv_tmp_path = NULL;
	}

	if (g_engine.peer_listing) {
		g_list_free_full(g_engine.peer_listing, g_free);
		g_engine.peer_listing = NULL;
	}

	if (g_engine.suppress_set) {
		g_hash_table_destroy(g_engine.suppress_set);
		g_engine.suppress_set = NULL;
	}

	g_mutex_clear(&g_engine.tx_lock);
	g_mutex_clear(&g_engine.fetch_lock);
	g_cond_clear(&g_engine.fetch_cond);
	g_mutex_clear(&g_engine.sysinfo_lock);
	g_cond_clear(&g_engine.sysinfo_cond);
	g_mutex_clear(&g_engine.suppress_lock);
	g_mutex_clear(&g_sync_status.lock);

	g_engine_initialized = false;

	g_printerr(SYNC_LOG_PREFIX "shutdown complete\n");
}

int odl_daemon_sync_set_folder(const char *path)
{
	if (!g_engine_initialized)
		return -EINVAL;

	if (!path || !path[0])
		return -EINVAL;

	if (g_strcmp0(g_engine.shared_folder, path) == 0)
		return 0;

	bool was_enabled = g_engine.enabled;

	if (was_enabled)
		odl_daemon_sync_set_enabled(false);

	g_strlcpy(g_engine.shared_folder, path,
		  sizeof(g_engine.shared_folder));
	g_mkdir_with_parents(g_engine.shared_folder, 0755);

	g_printerr(SYNC_LOG_PREFIX "shared_folder changed to %s\n",
		   g_engine.shared_folder);

	if (was_enabled)
		odl_daemon_sync_set_enabled(true);

	return 0;
}

void odl_daemon_sync_set_enabled(bool enabled)
{
	if (!g_engine_initialized) {
		g_mutex_lock(&g_sync_status.lock);
		g_sync_status.enabled = enabled;
		g_mutex_unlock(&g_sync_status.lock);
		return;
	}

	if (enabled && !g_engine.enabled) {
		g_printerr(SYNC_LOG_PREFIX "enabling\n");

		int ret = odl_tb5_open(&g_engine.handle,
				       g_engine.device_index);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "failed to open device %d: %s\n",
				   g_engine.device_index, strerror(-ret));
			return;
		}

		ret = odl_tb5_wait_peer(g_engine.handle, SYNC_PEER_TIMEOUT);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "peer not connected on device %d "
				   "(continuing)\n",
				   g_engine.device_index);
		}

		ret = odl_tb5_stream_open(g_engine.handle,
					  ODL_STREAM_SYNC, &g_engine.sid);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "failed to open stream: %s\n",
				   strerror(-ret));
			odl_tb5_close(g_engine.handle);
			g_engine.handle = NULL;
			return;
		}

		g_engine.enabled = true;

		g_engine.recv_running = true;
		g_engine.recv_thread = g_thread_new("odl-sync-recv",
						    sync_receiver_loop,
						    &g_engine);

		sync_scan_shared_folder_recursive(&g_engine,
						  g_engine.shared_folder, "");

		ret = inotify_start(&g_engine);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "failed to start inotify: %s\n",
				   strerror(-ret));
		}

		sync_exchange_catalog(&g_engine);

		g_engine.catalog_timer_id = g_timeout_add(
			SYNC_CATALOG_INTERVAL,
			sync_catalog_timer_cb, &g_engine);

		g_mutex_lock(&g_sync_status.lock);
		g_sync_status.enabled = true;
		g_mutex_unlock(&g_sync_status.lock);

		g_printerr(SYNC_LOG_PREFIX "enabled on device %d\n",
			   g_engine.device_index);

	} else if (!enabled && g_engine.enabled) {
		g_printerr(SYNC_LOG_PREFIX "disabling\n");

		g_engine.enabled = false;

		inotify_stop(&g_engine);

		if (g_engine.catalog_timer_id > 0) {
			g_source_remove(g_engine.catalog_timer_id);
			g_engine.catalog_timer_id = 0;
		}

		g_engine.recv_running = false;

		if (g_engine.recv_tid) {
			pthread_kill(g_engine.recv_tid, SIGUSR1);
		}

		if (g_engine.recv_thread) {
			g_thread_join(g_engine.recv_thread);
			g_engine.recv_thread = NULL;
		}
		g_engine.recv_tid = 0;

		if (g_engine.handle) {
			odl_tb5_stream_close(g_engine.handle, g_engine.sid);
			odl_tb5_close(g_engine.handle);
			g_engine.handle = NULL;
		}

		g_mutex_lock(&g_engine.fetch_lock);
		if (g_engine.fetch_in_progress) {
			g_engine.fetch_result = -ESHUTDOWN;
			g_engine.fetch_in_progress = false;
			g_cond_signal(&g_engine.fetch_cond);
		}
		g_mutex_unlock(&g_engine.fetch_lock);

		g_mutex_lock(&g_engine.sysinfo_lock);
		if (g_engine.sysinfo_pending) {
			g_engine.sysinfo_result = -ESHUTDOWN;
			g_engine.sysinfo_pending = false;
			g_engine.sysinfo_out = NULL;
			g_cond_signal(&g_engine.sysinfo_cond);
		}
		g_mutex_unlock(&g_engine.sysinfo_lock);

		g_mutex_lock(&g_sync_status.lock);
		g_sync_status.enabled = false;
		g_mutex_unlock(&g_sync_status.lock);

		g_printerr(SYNC_LOG_PREFIX "disabled\n");
	}
}

int odl_daemon_sync_fetch_file(const char *rel_path)
{
	if (!g_engine_initialized || !g_engine.enabled)
		return -ENOTCONN;

	if (!rel_path || !rel_path[0])
		return -EINVAL;

	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);
	if (ce && (ce->location == ODL_FILE_LOCAL ||
		   ce->location == ODL_FILE_BOTH)) {
		return 0;
	}

	if (ce && ce->location == ODL_FILE_CACHED)
		return 0;

	g_printerr(SYNC_LOG_PREFIX "fetch: requesting %s from peer\n",
		   rel_path);

	g_mutex_lock(&g_engine.fetch_lock);

	g_strlcpy(g_engine.fetch_pending_path, rel_path,
		  sizeof(g_engine.fetch_pending_path));
	g_engine.fetch_in_progress = true;
	g_engine.fetch_result = -EINPROGRESS;

	g_mutex_unlock(&g_engine.fetch_lock);

	g_mutex_lock(&g_engine.tx_lock);
	int ret = odl_sync_send_fetch_req(g_engine.handle, g_engine.sid,
					  ODL_STREAM_SYNC,
					  &g_engine.send_seq, rel_path);
	g_mutex_unlock(&g_engine.tx_lock);

	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "fetch: send FETCH_REQ failed: %s\n",
			   strerror(-ret));
		g_mutex_lock(&g_engine.fetch_lock);
		g_engine.fetch_in_progress = false;
		g_mutex_unlock(&g_engine.fetch_lock);
		return ret;
	}

	g_mutex_lock(&g_engine.fetch_lock);
	while (g_engine.fetch_in_progress)
		g_cond_wait(&g_engine.fetch_cond, &g_engine.fetch_lock);

	int result = g_engine.fetch_result;
	g_mutex_unlock(&g_engine.fetch_lock);

	if (result == 0) {
		char *cache_path = sync_cache_path(&g_engine, rel_path);
		char *shared_path = sync_shared_path(&g_engine, rel_path);

		sync_suppress_add(&g_engine, rel_path);

		struct stat lst;
		if (lstat(shared_path, &lst) == 0 && S_ISLNK(lst.st_mode))
			unlink(shared_path);

		char *parent = g_path_get_dirname(shared_path);
		g_mkdir_with_parents(parent, 0755);
		g_free(parent);

		if (g_rename(cache_path, shared_path) < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "fetch: failed to move %s to %s: %s\n",
				   cache_path, shared_path, strerror(errno));
		} else {
			odl_catalog_mark_both(rel_path);
		}

		odl_daemon_fuse_invalidate(rel_path);

		g_free(cache_path);
		g_free(shared_path);

		g_printerr(SYNC_LOG_PREFIX "fetch: %s complete "
			   "(downloaded to shared_folder)\n", rel_path);
	} else {
		g_printerr(SYNC_LOG_PREFIX "fetch: %s failed: %s\n",
			   rel_path, strerror(-result));
	}

	return result;
}

int odl_daemon_sync_transfer_file(const char *rel_path)
{
	if (!g_engine_initialized || !g_engine.enabled)
		return -ENOTCONN;

	if (!rel_path || !rel_path[0])
		return -EINVAL;

	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);
	if (!ce || (ce->location != ODL_FILE_LOCAL &&
		    ce->location != ODL_FILE_BOTH)) {
		g_printerr(SYNC_LOG_PREFIX
			   "transfer: %s not available locally\n", rel_path);
		return -ENOENT;
	}

	char *abs_path = sync_shared_path(&g_engine, rel_path);
	struct stat st;

	if (stat(abs_path, &st) < 0 || !S_ISREG(st.st_mode)) {
		g_printerr(SYNC_LOG_PREFIX
			   "transfer: cannot stat %s: %s\n",
			   rel_path, strerror(errno));
		g_free(abs_path);
		return -errno;
	}

	uint64_t file_size = (uint64_t)st.st_size;
	uint64_t mtime_ns = sync_get_mtime_ns(abs_path);
	uint32_t mode = (uint32_t)st.st_mode & 07777;
	uint32_t num_chunks = (file_size > 0)
		? (uint32_t)((file_size + ODL_SYNC_CHUNK_SIZE - 1) /
			     ODL_SYNC_CHUNK_SIZE)
		: 1;

	uint8_t sha256[32];
	memset(sha256, 0, sizeof(sha256));
	if (file_size > 0) {
		int ret = odl_sync_sha256_file(abs_path, sha256);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "transfer: sha256 failed for %s\n",
				   rel_path);
			g_free(abs_path);
			return ret;
		}
	}

	int fd = open(abs_path, O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		int err = errno;
		g_printerr(SYNC_LOG_PREFIX
			   "transfer: cannot open %s: %s\n",
			   rel_path, strerror(err));
		g_free(abs_path);
		return -err;
	}

	g_printerr(SYNC_LOG_PREFIX "transfer: sending %s (%" G_GUINT64_FORMAT
		   " bytes, %u chunks)\n",
		   rel_path, file_size, num_chunks);

	sync_status_add_pending(1);

	g_mutex_lock(&g_engine.tx_lock);

	int ret = odl_sync_send_file_meta(g_engine.handle, g_engine.sid,
					  ODL_STREAM_SYNC, &g_engine.send_seq,
					  rel_path, file_size, mtime_ns,
					  mode, num_chunks, sha256);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "transfer: send file_meta failed for %s: %s\n",
			   rel_path, strerror(-ret));
		g_mutex_unlock(&g_engine.tx_lock);
		close(fd);
		g_free(abs_path);
		sync_status_add_pending(-1);
		return ret;
	}

	uint8_t *chunk_buf = g_malloc(ODL_SYNC_CHUNK_SIZE);
	uint64_t bytes_sent = 0;

	for (uint32_t i = 0; i < num_chunks; i++) {
		uint32_t to_read = ODL_SYNC_CHUNK_SIZE;
		if (bytes_sent + to_read > file_size)
			to_read = (uint32_t)(file_size - bytes_sent);

		ssize_t nread = 0;
		if (to_read > 0) {
			nread = read(fd, chunk_buf, to_read);
			if (nread < 0) {
				g_printerr(SYNC_LOG_PREFIX
					   "transfer: read error %s: %s\n",
					   rel_path, strerror(errno));
				break;
			}
		}

		ret = odl_sync_send_file_data(g_engine.handle, g_engine.sid,
					      ODL_STREAM_SYNC,
					      &g_engine.send_seq,
					      i, chunk_buf, (uint32_t)nread);
		if (ret < 0) {
			g_printerr(SYNC_LOG_PREFIX
				   "transfer: send chunk %u failed for %s\n",
				   i, rel_path);
			break;
		}

		bytes_sent += (uint64_t)nread;
	}

	g_mutex_unlock(&g_engine.tx_lock);

	g_free(chunk_buf);
	close(fd);

	sync_status_add_bytes(bytes_sent);
	sync_status_add_pending(-1);
	sync_update_last_sync_time();

	odl_catalog_mark_both(rel_path);

	odl_daemon_dbus_emit_sync_file_transferred(rel_path, "transferred",
						   bytes_sent);

	g_printerr(SYNC_LOG_PREFIX "transfer: sent %s (%" G_GUINT64_FORMAT
		   " bytes)\n", rel_path, bytes_sent);

	g_free(abs_path);
	return 0;
}

int odl_daemon_sync_remove_from_peer(const char *rel_path)
{
	if (!g_engine_initialized || !g_engine.enabled)
		return -ENOTCONN;

	if (!rel_path || !rel_path[0])
		return -EINVAL;

	g_printerr(SYNC_LOG_PREFIX "remove_from_peer: %s\n", rel_path);

	g_mutex_lock(&g_engine.tx_lock);
	int ret = odl_sync_send_remove_req(g_engine.handle, g_engine.sid,
					   ODL_STREAM_SYNC,
					   &g_engine.send_seq, rel_path);
	g_mutex_unlock(&g_engine.tx_lock);

	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "remove_from_peer: send failed: %s\n",
			   strerror(-ret));
	}

	return ret;
}

int odl_daemon_sync_remove_local(const char *rel_path)
{
	if (!g_engine_initialized)
		return -EINVAL;

	if (!rel_path || !rel_path[0])
		return -EINVAL;

	g_printerr(SYNC_LOG_PREFIX "remove_local: %s\n", rel_path);

	const struct odl_catalog_entry *ce = odl_catalog_lookup(rel_path);
	if (!ce) {
		g_printerr(SYNC_LOG_PREFIX
			   "remove_local: %s not in catalog\n", rel_path);
		return -ENOENT;
	}

	sync_suppress_add(&g_engine, rel_path);

	if (ce->location == ODL_FILE_BOTH) {
		char *shared_path = sync_shared_path(&g_engine, rel_path);

		g_unlink(shared_path);

		sync_create_remote_symlink(&g_engine, rel_path);

		struct odl_catalog_entry updated = *ce;
		updated.location = ODL_FILE_REMOTE;
		odl_catalog_set(rel_path, &updated);

		g_free(shared_path);
		return 0;
	}

	int ret = odl_catalog_remove_local_copy(rel_path);
	if (ret < 0) {
		g_printerr(SYNC_LOG_PREFIX
			   "remove_local: failed for %s: %s\n",
			   rel_path, strerror(-ret));
	}

	return ret;
}

char *odl_daemon_sync_list_files_json(const char *dir_path)
{
	if (!g_engine_initialized)
		return NULL;

	GList *entries = odl_catalog_list_dir(dir_path ? dir_path : "");
	GList *l;

	GString *json = g_string_new("[");
	bool first = true;

	for (l = entries; l != NULL; l = l->next) {
		const struct odl_catalog_entry *ce = l->data;

		const char *loc_str;
		switch (ce->location) {
		case ODL_FILE_LOCAL:  loc_str = "local";  break;
		case ODL_FILE_REMOTE: loc_str = "remote"; break;
		case ODL_FILE_CACHED: loc_str = "cached"; break;
		case ODL_FILE_BOTH:   loc_str = "both";   break;
		default:              loc_str = "unknown"; break;
		}

		if (!first)
			g_string_append_c(json, ',');
		first = false;

		g_string_append_printf(json,
			"{\"path\":\"%s\","
			"\"location\":\"%s\","
			"\"is_dir\":%s,"
			"\"size\":%" G_GUINT64_FORMAT ","
			"\"mtime\":%" G_GUINT64_FORMAT "}",
			ce->rel_path,
			loc_str,
			ce->is_dir ? "true" : "false",
			ce->file_size,
			ce->mtime_ns);
	}

	g_string_append_c(json, ']');

	g_list_free(entries);

	return g_string_free(json, FALSE);
}

bool odl_daemon_sync_owns_device(int index)
{
	return g_engine_initialized && g_engine.enabled &&
	       g_engine.device_index == index;
}

int odl_daemon_sync_request_sysinfo(int device_index,
				    struct odl_sysinfo *out)
{
	if (!g_engine_initialized || !g_engine.enabled ||
	    g_engine.device_index != device_index || !out)
		return -ENODEV;

	g_mutex_lock(&g_engine.sysinfo_lock);
	if (g_engine.sysinfo_pending) {
		g_mutex_unlock(&g_engine.sysinfo_lock);
		return -EBUSY;
	}
	g_engine.sysinfo_pending = true;
	g_engine.sysinfo_out = out;
	g_engine.sysinfo_result = -1;
	g_mutex_unlock(&g_engine.sysinfo_lock);

	g_mutex_lock(&g_engine.tx_lock);
	int ret = odl_cli_send_msg(g_engine.handle, g_engine.sid,
				   ODL_STREAM_SYNC,
				   ODL_CLI_MSG_SYSINFO_REQ, 0, NULL, 0);
	g_mutex_unlock(&g_engine.tx_lock);

	if (ret < 0) {
		g_mutex_lock(&g_engine.sysinfo_lock);
		g_engine.sysinfo_pending = false;
		g_engine.sysinfo_out = NULL;
		g_mutex_unlock(&g_engine.sysinfo_lock);
		g_printerr(SYNC_LOG_PREFIX
			   "failed to send SYSINFO_REQ: %s\n",
			   strerror(-ret));
		return ret;
	}

	g_printerr(SYNC_LOG_PREFIX "sent SYSINFO_REQ, waiting for response\n");

	g_mutex_lock(&g_engine.sysinfo_lock);
	gint64 end_time = g_get_monotonic_time() + 10 * G_USEC_PER_SEC;
	while (g_engine.sysinfo_pending) {
		if (!g_cond_wait_until(&g_engine.sysinfo_cond,
				       &g_engine.sysinfo_lock, end_time)) {
			g_engine.sysinfo_pending = false;
			g_engine.sysinfo_out = NULL;
			g_mutex_unlock(&g_engine.sysinfo_lock);
			g_printerr(SYNC_LOG_PREFIX
				   "SYSINFO_REQ timed out\n");
			return -ETIMEDOUT;
		}
	}
	ret = g_engine.sysinfo_result;
	g_mutex_unlock(&g_engine.sysinfo_lock);

	return ret;
}
