/*
 * OdinLink Daemon — FUSE API stubs (used when FUSE3 is not available)
 *
 * Provides no-op implementations of the FUSE integration functions so
 * the daemon links cleanly even without libfuse3-dev installed.
 * When FUSE3 IS available, the real implementations in
 * odl_tb5_daemon_fuse.c take precedence.
 */

#include "odl_tb5_daemon_fuse.h"
#include <stddef.h>

int odl_daemon_fuse_init(const char *mount_point)
{
    (void)mount_point;
    return -1; /* FUSE not available */
}

void odl_daemon_fuse_shutdown(void)
{
}

void odl_daemon_fuse_invalidate(const char *path)
{
    (void)path;
}

void odl_daemon_fuse_set_callbacks(const struct odl_fuse_callbacks *cbs)
{
    (void)cbs;
}
