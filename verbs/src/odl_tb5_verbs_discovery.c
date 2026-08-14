/*
 * OdinLink — Verbs: device discovery interposer
 *
 * The shim already implements open_device/query_port/create_qp/post_send/
 * poll_cq, and odl_init_context_ops() fills the context ops table that the
 * static-inline data-path verbs (ibv_post_send, ibv_poll_cq) dispatch
 * through. What was missing is *discovery*: every rdma-core consumer
 * (llama.cpp's ggml-rpc RDMA transport, RCCL's IB transport, perftest)
 * starts with
 *
 *     ibv_get_device_list() -> ibv_get_device_name() -> ibv_open_device()
 *     -> ibv_query_port() -> ibv_query_gid_ex()   (match GID to local IP)
 *
 * and none of ibv_get_device_list / ibv_free_device_list /
 * ibv_get_device_name / _ibv_query_gid_ex were exported, so OdinLink was
 * invisible: ibv_devices printed nothing and applications fell back to TCP.
 *
 * Those four are real (non-inline) exported symbols in libibverbs, so an
 * LD_PRELOAD interposer can supply them. Non-OdinLink devices are forwarded
 * to the real libibverbs so a Mellanox/rxe card in the same box keeps working.
 *
 * Usage:
 *     LD_PRELOAD=libodl_tb5_verbs.so \
 *     ODL_RDMA_GID_IFACE=bond0 \
 *     ./ggml-rpc-server ...
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>

#include <errno.h>

#include "odl_tb5_verbs.h"
#include "odl_tb5_verbs_wrapper.h"

/* ── real libibverbs entry points (for non-OdinLink devices) ─────────── */

static struct ibv_device **(*real_get_device_list)(int *);
static void               (*real_free_device_list)(struct ibv_device **);
static const char        *(*real_get_device_name)(struct ibv_device *);
static int                (*real_query_gid)(struct ibv_context *, uint8_t, int,
                                             union ibv_gid *);
static int                (*real_query_gid_ex)(struct ibv_context *, uint32_t,
                                               uint32_t, struct ibv_gid_entry *,
                                               uint32_t, size_t);

static void odl_resolve_real(void)
{
    static bool done = false;
    if (done) return;
    done = true;
    real_get_device_list  = dlsym(RTLD_NEXT, "ibv_get_device_list");
    real_free_device_list = dlsym(RTLD_NEXT, "ibv_free_device_list");
    real_get_device_name  = dlsym(RTLD_NEXT, "ibv_get_device_name");
    /* ibv_query_gid_ex() is static inline in verbs.h and forwards here. */
    real_query_gid        = dlsym(RTLD_NEXT, "ibv_query_gid");
    real_query_gid_ex     = dlsym(RTLD_NEXT, "_ibv_query_gid_ex");
}

/* ── local IPv4, used to synthesise a RoCE v2 GID ────────────────────── */

/*
 * Consumers match a GID against the address they bound their TCP socket to.
 * A RoCE v2 GID for IPv4 is the IPv4-mapped IPv6 form: ::ffff:a.b.c.d.
 * Pick the address of the interface carrying the Thunderbolt traffic
 * (ODL_RDMA_GID_IFACE, default bond0), or take it verbatim from
 * ODL_RDMA_GID_IP.
 */
static bool odl_local_ipv4(struct in_addr *out)
{
    const char *ip_env = getenv("ODL_RDMA_GID_IP");
    if (ip_env && inet_pton(AF_INET, ip_env, out) == 1)
        return true;

    const char *want = getenv("ODL_RDMA_GID_IFACE");
    if (!want) want = "bond0";

    struct ifaddrs *ifa = NULL;
    if (getifaddrs(&ifa) != 0) {
        static bool warned;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "odl_tb5 verbs: getifaddrs failed: %s\n",
                    strerror(errno));
        }
        return false;
    }

    bool found = false;
    for (struct ifaddrs *p = ifa; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        if (strcmp(p->ifa_name, want) != 0) continue;
        *out = ((struct sockaddr_in *)p->ifa_addr)->sin_addr;
        found = true;
        break;
    }
    freeifaddrs(ifa);
    if (!found) {
        static bool warned;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "odl_tb5 verbs: no IPv4 on interface \"%s\" "
                    "(default bond0). Set ODL_RDMA_GID_IFACE=en05 "
                    "or ODL_RDMA_GID_IP=a.b.c.d\n", want);
        }
    }
    return found;
}

static void odl_make_roce_v2_gid(union ibv_gid *gid)
{
    struct in_addr a;
    memset(gid, 0, sizeof(*gid));
    if (!odl_local_ipv4(&a)) return;
    /* ::ffff:a.b.c.d  */
    gid->raw[10] = 0xff;
    gid->raw[11] = 0xff;
    memcpy(&gid->raw[12], &a.s_addr, 4);
}

/* ── ibv_get_device_list / free / name ───────────────────────────────── */

struct ibv_device **ibv_get_device_list(int *num_devices)
{
    odl_resolve_real();

    int n_odl = odl_num_tb5_devices();

    int n_real = 0;
    struct ibv_device **real_list = NULL;
    if (real_get_device_list)
        real_list = real_get_device_list(&n_real);
    if (n_real < 0) n_real = 0;

    struct ibv_device **out = calloc((size_t)n_odl + n_real + 1,
                                     sizeof(*out));
    if (!out) {
        if (real_list && real_free_device_list) real_free_device_list(real_list);
        if (num_devices) *num_devices = 0;
        return NULL;
    }

    int k = 0;
    for (int i = 0; i < n_odl; i++) {
        struct ibv_device *d = odl_find_tb5_device(i);
        if (d) out[k++] = d;
    }
    for (int i = 0; i < n_real; i++)
        out[k++] = real_list[i];
    out[k] = NULL;

    /*
     * The real list's backing array is no longer needed — the ibv_device
     * pointers inside it stay valid (libibverbs owns them for process
     * lifetime), so it is safe to release the array itself here.
     */
    if (real_list && real_free_device_list)
        real_free_device_list(real_list);

    if (num_devices) *num_devices = k;
    return out;
}

void ibv_free_device_list(struct ibv_device **list)
{
    /* Our array is plain calloc'd; the devices themselves are not owned. */
    free(list);
}

const char *ibv_get_device_name(struct ibv_device *device)
{
    odl_resolve_real();
    if (!device) return NULL;
    if (odl_is_tb5_device(device)) return device->name;
    if (real_get_device_name) return real_get_device_name(device);
    return device->name;
}

/* ── _ibv_query_gid_ex (target of the inline ibv_query_gid_ex) ───────── */

int _ibv_query_gid_ex(struct ibv_context *context, uint32_t port_num,
                      uint32_t gid_index, struct ibv_gid_entry *entry,
                      uint32_t flags, size_t entry_size)
{
    odl_resolve_real();

    if (context && context->device && odl_is_tb5_device(context->device)) {
        if (port_num != 1 || gid_index != 0 || !entry) return ENODATA;
        if (entry_size < sizeof(*entry)) return EINVAL;
        memset(entry, 0, sizeof(*entry));
        odl_make_roce_v2_gid(&entry->gid);
        entry->gid_index    = gid_index;
        entry->port_num     = port_num;
        entry->gid_type     = IBV_GID_TYPE_ROCE_V2;
        entry->ndev_ifindex = 0;
        return 0;
    }

    if (real_query_gid_ex)
        return real_query_gid_ex(context, port_num, gid_index, entry, flags,
                                 entry_size);
    return EOPNOTSUPP;
}

/* ── ibv_query_gid (classic form; rdma_probe() calls this too) ───────── */

/*
 * Must be interposed as well as _ibv_query_gid_ex: the real libibverbs
 * implementation walks provider-private context state that an OdinLink
 * context does not have, and segfaults on it.
 */
int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                  union ibv_gid *gid)
{
    odl_resolve_real();

    if (context && context->device && odl_is_tb5_device(context->device)) {
        if (port_num != 1 || index != 0 || !gid) return ENODATA;
        odl_make_roce_v2_gid(gid);
        return 0;
    }

    if (real_query_gid)
        return real_query_gid(context, port_num, index, gid);
    return EOPNOTSUPP;
}
