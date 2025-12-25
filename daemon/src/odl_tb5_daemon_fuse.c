/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025-2026 OdinLink Project
 */
#define _GNU_SOURCE
#define FUSE_USE_VERSION 31

#include "odl_tb5_daemon_fuse.h"

#include <fuse3/fuse.h>
#include <fuse3/fuse_lowlevel.h>
#include <glib.h>
#include <glib/gstdio.h>

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <linux/fs.h>

#define FUSE_LOG_PREFIX  "odl_tb5_daemon: fuse: "

static struct fuse          *g_fuse;
static struct fuse_session  *g_fuse_se;
static GThread              *g_fuse_thread;
static volatile bool         g_fuse_mounted;
static char                  g_mount_point[512];

static struct odl_fuse_callbacks g_cb;
static bool                      g_cb_set;

/* Strip leading '/' to get a catalog-relative path. */
static const char *fuse_rel_path(const char *path)
{
	if (path[0] == '/')
		path++;
	return path;
}

static int odl_fuse_getattr(const char *path, struct stat *stbuf,
			     struct fuse_file_info *fi)
{
	(void)fi;
	const char *rel = fuse_rel_path(path);

	memset(stbuf, 0, sizeof(*stbuf));

	if (rel[0] == '\0') {
		stbuf->st_mode = S_IFDIR | 0755;
		stbuf->st_nlink = 2;
		return 0;
	}

	if (!g_cb_set || !g_cb.getattr) {
		g_printerr(FUSE_LOG_PREFIX
			   "getattr: no callbacks registered\n");
		return -ENOSYS;
	}

	return g_cb.getattr(rel, stbuf);
}

struct readdir_ctx {
	void            *buf;
	fuse_fill_dir_t  filler;
};

static void readdir_add_entry(const char *name, const struct stat *st,
			      void *ctx)
{
	struct readdir_ctx *rc = ctx;

	rc->filler(rc->buf, name, st, 0, 0);
}

static int odl_fuse_readdir(const char *path, void *buf,
			     fuse_fill_dir_t filler, off_t offset,
			     struct fuse_file_info *fi,
			     enum fuse_readdir_flags flags)
{
	(void)offset;
	(void)fi;
	(void)flags;

	const char *dir = fuse_rel_path(path);

	filler(buf, ".", NULL, 0, 0);
	filler(buf, "..", NULL, 0, 0);

	if (!g_cb_set || !g_cb.readdir) {
		g_printerr(FUSE_LOG_PREFIX
			   "readdir: no callbacks registered\n");
		return -ENOSYS;
	}

	struct readdir_ctx rc = {
		.buf    = buf,
		.filler = filler,
	};

	return g_cb.readdir(dir, readdir_add_entry, &rc);
}

static int odl_fuse_open(const char *path, struct fuse_file_info *fi)
{
	const char *rel = fuse_rel_path(path);
	char *local_path;
	int fd;

	if (!g_cb_set || !g_cb.get_local_path) {
		g_printerr(FUSE_LOG_PREFIX
			   "open: no callbacks registered\n");
		return -ENOSYS;
	}

	local_path = g_cb.get_local_path(rel);

	if (!local_path) {
		if (!g_cb.fetch_remote) {
			g_printerr(FUSE_LOG_PREFIX
				   "open: fetch_remote not available\n");
			return -ENOSYS;
		}

		int ret = g_cb.fetch_remote(rel);
		if (ret < 0) {
			g_printerr(FUSE_LOG_PREFIX
				   "open: fetch_remote failed for %s: %s\n",
				   rel, strerror(-ret));
			return ret;
		}

		local_path = g_cb.get_local_path(rel);
		if (!local_path) {
			g_printerr(FUSE_LOG_PREFIX
				   "open: file still not local after fetch: "
				   "%s\n", rel);
			return -EIO;
		}
	}

	fd = open(local_path, fi->flags & ~O_NOFOLLOW);
	if (fd < 0) {
		int err = errno;
		g_printerr(FUSE_LOG_PREFIX
			   "open: cannot open %s: %s\n",
			   local_path, strerror(err));
		g_free(local_path);
		return -err;
	}

	fi->fh = (uint64_t)fd;
	g_free(local_path);
	return 0;
}

static int odl_fuse_read(const char *path, char *buf, size_t size,
			  off_t offset, struct fuse_file_info *fi)
{
	(void)path;
	ssize_t nread;

	nread = pread((int)fi->fh, buf, size, offset);
	if (nread < 0)
		return -errno;

	return (int)nread;
}

static int odl_fuse_write(const char *path, const char *buf, size_t size,
			   off_t offset, struct fuse_file_info *fi)
{
	(void)path;
	ssize_t nw;

	nw = pwrite((int)fi->fh, buf, size, offset);
	if (nw < 0)
		return -errno;

	return (int)nw;
}

static int odl_fuse_release(const char *path, struct fuse_file_info *fi)
{
	(void)path;

	if (fi->fh != (uint64_t)-1)
		close((int)fi->fh);

	return 0;
}

static int odl_fuse_create(const char *path, mode_t mode,
			    struct fuse_file_info *fi)
{
	const char *rel = fuse_rel_path(path);

	if (!g_cb_set || !g_cb.create_file) {
		g_printerr(FUSE_LOG_PREFIX
			   "create: no callbacks registered\n");
		return -ENOSYS;
	}

	int fd = g_cb.create_file(rel, mode);
	if (fd < 0)
		return fd;

	fi->fh = (uint64_t)fd;
	return 0;
}

static int odl_fuse_unlink(const char *path)
{
	const char *rel = fuse_rel_path(path);

	if (!g_cb_set || !g_cb.delete_file) {
		g_printerr(FUSE_LOG_PREFIX
			   "unlink: no callbacks registered\n");
		return -ENOSYS;
	}

	return g_cb.delete_file(rel);
}

static int odl_fuse_mkdir(const char *path, mode_t mode)
{
	const char *rel = fuse_rel_path(path);

	if (!g_cb_set || !g_cb.create_dir) {
		g_printerr(FUSE_LOG_PREFIX
			   "mkdir: no callbacks registered\n");
		return -ENOSYS;
	}

	return g_cb.create_dir(rel, mode);
}

static int odl_fuse_rmdir(const char *path)
{
	const char *rel = fuse_rel_path(path);

	if (!g_cb_set || !g_cb.delete_dir) {
		g_printerr(FUSE_LOG_PREFIX
			   "rmdir: no callbacks registered\n");
		return -ENOSYS;
	}

	return g_cb.delete_dir(rel);
}

static int odl_fuse_truncate(const char *path, off_t size,
			      struct fuse_file_info *fi)
{
	const char *rel = fuse_rel_path(path);

	if (fi && fi->fh != (uint64_t)-1)
		return (ftruncate((int)fi->fh, size) < 0) ? -errno : 0;

	if (!g_cb_set || !g_cb.get_local_path)
		return -ENOSYS;

	char *local_path = g_cb.get_local_path(rel);
	if (!local_path)
		return -ENOENT;

	int ret = truncate(local_path, size);
	int err = errno;
	g_free(local_path);

	return (ret < 0) ? -err : 0;
}

static int odl_fuse_utimens(const char *path, const struct timespec tv[2],
			     struct fuse_file_info *fi)
{
	(void)fi;
	const char *rel = fuse_rel_path(path);

	if (!g_cb_set || !g_cb.get_local_path)
		return -ENOSYS;

	char *local_path = g_cb.get_local_path(rel);
	if (!local_path)
		return -ENOENT;

	int ret = utimensat(AT_FDCWD, local_path, tv, 0);
	int err = errno;
	g_free(local_path);

	return (ret < 0) ? -err : 0;
}

static int odl_fuse_statfs(const char *path, struct statvfs *stbuf)
{
	(void)path;

	if (statvfs(g_mount_point, stbuf) < 0)
		return -errno;

	return 0;
}

static int odl_fuse_rename(const char *from, const char *to,
			    unsigned int flags)
{
	const char *rel_from = fuse_rel_path(from);
	const char *rel_to   = fuse_rel_path(to);

	if (!g_cb_set || !g_cb.get_local_path)
		return -ENOSYS;

	char *local_from = g_cb.get_local_path(rel_from);
	if (!local_from)
		return -ENOENT;

	char *local_to = g_cb.get_local_path(rel_to);
	if (!local_to) {
		char *dir = g_path_get_dirname(local_from);
		char *base_dir = g_path_get_dirname(dir);
		g_free(dir);

		size_t from_len = strlen(rel_from);
		size_t local_from_len = strlen(local_from);

		if (local_from_len > from_len) {
			size_t root_len = local_from_len - from_len;
			char *root = g_strndup(local_from, root_len);

			local_to = g_strconcat(root, rel_to, NULL);
			g_free(root);
		} else {
			g_free(local_from);
			g_free(base_dir);
			return -EINVAL;
		}

		g_free(base_dir);
	}

	char *to_parent = g_path_get_dirname(local_to);
	g_mkdir_with_parents(to_parent, 0755);
	g_free(to_parent);

	int ret;

	if (flags & RENAME_NOREPLACE) {
		ret = renameat2(AT_FDCWD, local_from,
				AT_FDCWD, local_to, flags);
	} else {
		ret = rename(local_from, local_to);
	}

	int err = errno;
	g_free(local_from);
	g_free(local_to);

	return (ret < 0) ? -err : 0;
}

static const struct fuse_operations odl_fuse_ops = {
	.getattr  = odl_fuse_getattr,
	.readdir  = odl_fuse_readdir,
	.open     = odl_fuse_open,
	.read     = odl_fuse_read,
	.write    = odl_fuse_write,
	.release  = odl_fuse_release,
	.create   = odl_fuse_create,
	.unlink   = odl_fuse_unlink,
	.mkdir    = odl_fuse_mkdir,
	.rmdir    = odl_fuse_rmdir,
	.truncate = odl_fuse_truncate,
	.utimens  = odl_fuse_utimens,
	.statfs   = odl_fuse_statfs,
	.rename   = odl_fuse_rename,
};

static gpointer fuse_thread_func(gpointer data)
{
	(void)data;

	g_printerr(FUSE_LOG_PREFIX "fuse_loop thread started\n");

	int ret = fuse_loop(g_fuse);
	if (ret < 0)
		g_printerr(FUSE_LOG_PREFIX
			   "fuse_loop exited with error: %s\n",
			   strerror(-ret));
	else
		g_printerr(FUSE_LOG_PREFIX "fuse_loop exited cleanly\n");

	g_fuse_mounted = false;
	return NULL;
}

void odl_daemon_fuse_set_callbacks(const struct odl_fuse_callbacks *cb)
{
	if (cb) {
		memcpy(&g_cb, cb, sizeof(g_cb));
		g_cb_set = true;
	} else {
		memset(&g_cb, 0, sizeof(g_cb));
		g_cb_set = false;
	}
}

int odl_daemon_fuse_init(const char *mount_point)
{
	if (g_fuse_mounted) {
		g_printerr(FUSE_LOG_PREFIX "already mounted\n");
		return -EBUSY;
	}

	if (!mount_point || !mount_point[0]) {
		g_printerr(FUSE_LOG_PREFIX "no mount point specified\n");
		return -EINVAL;
	}

	g_strlcpy(g_mount_point, mount_point, sizeof(g_mount_point));

	if (g_mkdir_with_parents(g_mount_point, 0755) < 0) {
		int err = errno;
		g_printerr(FUSE_LOG_PREFIX
			   "failed to create mount point %s: %s\n",
			   g_mount_point, strerror(err));
		return -err;
	}

	const char *argv[] = {
		"odl_tb5_fuse",
		"-o", "default_permissions",
	};
	int argc = G_N_ELEMENTS(argv);

	struct fuse_args args = FUSE_ARGS_INIT(argc, (char **)argv);

	g_fuse = fuse_new(&args, &odl_fuse_ops,
			   sizeof(odl_fuse_ops), NULL);
	fuse_opt_free_args(&args);

	if (!g_fuse) {
		g_printerr(FUSE_LOG_PREFIX "fuse_new failed\n");
		return -EIO;
	}

	if (fuse_mount(g_fuse, g_mount_point) < 0) {
		g_printerr(FUSE_LOG_PREFIX
			   "fuse_mount failed for %s\n", g_mount_point);
		fuse_destroy(g_fuse);
		g_fuse = NULL;
		return -EIO;
	}

	g_fuse_se = fuse_get_session(g_fuse);
	g_fuse_mounted = true;

	g_fuse_thread = g_thread_new("odl-fuse-loop",
				      fuse_thread_func, NULL);
	if (!g_fuse_thread) {
		g_printerr(FUSE_LOG_PREFIX
			   "failed to create FUSE thread\n");
		fuse_unmount(g_fuse);
		fuse_destroy(g_fuse);
		g_fuse = NULL;
		g_fuse_mounted = false;
		return -ENOMEM;
	}

	g_printerr(FUSE_LOG_PREFIX "mounted at %s\n", g_mount_point);
	return 0;
}

void odl_daemon_fuse_shutdown(void)
{
	if (!g_fuse)
		return;

	g_printerr(FUSE_LOG_PREFIX "shutting down\n");

	if (g_fuse_se)
		fuse_session_exit(g_fuse_se);

	fuse_unmount(g_fuse);

	if (g_fuse_thread) {
		g_thread_join(g_fuse_thread);
		g_fuse_thread = NULL;
	}

	fuse_destroy(g_fuse);
	g_fuse = NULL;
	g_fuse_se = NULL;
	g_fuse_mounted = false;

	g_printerr(FUSE_LOG_PREFIX "shutdown complete\n");
}

int odl_daemon_fuse_is_mounted(void)
{
	return g_fuse_mounted;
}

void odl_daemon_fuse_invalidate(const char *rel_path)
{
	(void)rel_path;

	if (rel_path && rel_path[0])
		g_printerr(FUSE_LOG_PREFIX
			   "invalidate: %s (cache refresh on next access)\n",
			   rel_path);
}
