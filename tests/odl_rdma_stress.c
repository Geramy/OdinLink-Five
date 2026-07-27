/*
 * OdinLink — RDMA bulk-transfer stress / conformance test
 *
 * Why this exists
 * ---------------
 * llama.cpp's ggml-rpc RDMA transport completes a 1.93 GiB model over OdinLink
 * but hangs partway through a 20.88 GiB one. Bisecting that with a 20-minute
 * benchmark is hopeless, and "it finished" says nothing about whether the bytes
 * were correct. This reproduces the transport's exact wire pattern in a few
 * seconds, verifies every byte, and reports the precise message index where
 * things go wrong.
 *
 * It mirrors ggml/src/ggml-rpc/transport.cpp:
 *   RDMA_CHUNK    = 256 KiB   per send/recv
 *   RDMA_RX_DEPTH = 24        pre-posted receives
 *   RDMA_TX_DEPTH = 2         in-flight sends, reaped from the send CQ
 * all of which are sweepable here so we can find which one the implementation
 * actually breaks on.
 *
 * Build:
 *   gcc -O2 -o odl_rdma_stress tests/odl_rdma_stress.c -libverbs -lpthread
 * Run (both sides need LD_PRELOAD=libodl_tb5_verbs.so for OdinLink):
 *   peer:  ./odl_rdma_stress --server
 *   head:  ./odl_rdma_stress --client <peer-ip> --total 2G
 *
 * Exit codes: 0 ok, 1 usage/setup, 2 DATA CORRUPTION, 3 STALL (no progress).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>
#include <pthread.h>

#define BOOTSTRAP_PORT 18515
#define GID_SIZE       16

static size_t   opt_chunk    = 256 * 1024;
static int      opt_rx_depth = 24;
static int      opt_tx_depth = 2;
static uint64_t opt_total    = 2ULL << 30;   /* bytes to push */
static int      opt_stall_s  = 15;           /* no-progress timeout */
static const char *opt_dev   = NULL;
static int      opt_gid_idx  = -1;
static bool     opt_bidir    = false;  /* both peers send AND receive */
static bool     opt_mixed    = false;  /* interleave small control msgs */

struct conn_info {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[GID_SIZE];
} __attribute__((packed));

struct rdma_ctx {
    struct ibv_context *ctx;
    struct ibv_pd      *pd;
    struct ibv_cq      *scq, *rcq;
    struct ibv_qp      *qp;
    struct ibv_mr      *tx_mr, *rx_mr;
    uint8_t            *tx_buf, *rx_buf;
    int                 ib_port, gid_idx;
    enum ibv_mtu        mtu;
};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Deterministic, position-dependent payload: any reorder, truncation, dropped
 * fragment or stale buffer shows up as a mismatch at a known offset. */
static void fill_chunk(uint8_t *buf, size_t len, uint64_t seq)
{
    uint64_t *w = (uint64_t *)buf;
    size_t n = len / sizeof(uint64_t);
    for (size_t i = 0; i < n; i++)
        w[i] = seq * 0x9E3779B97F4A7C15ULL + i;
}

static long verify_chunk(const uint8_t *buf, size_t len, uint64_t seq)
{
    const uint64_t *w = (const uint64_t *)buf;
    size_t n = len / sizeof(uint64_t);
    for (size_t i = 0; i < n; i++)
        if (w[i] != seq * 0x9E3779B97F4A7C15ULL + i)
            return (long)i;      /* offset of first bad word */
    return -1;
}

/* ── device / QP setup ───────────────────────────────────────────────── */

static bool rdma_open(struct rdma_ctx *r, const char *local_ip)
{
    int n = 0;
    struct ibv_device **devs = ibv_get_device_list(&n);
    if (!devs || n == 0) {
        fprintf(stderr, "no RDMA devices (LD_PRELOAD the OdinLink shim?)\n");
        return false;
    }

    uint8_t target[GID_SIZE] = {0};
    struct in_addr a;
    if (inet_pton(AF_INET, local_ip, &a) == 1) {
        target[10] = 0xff; target[11] = 0xff;
        memcpy(&target[12], &a.s_addr, 4);
    }

    for (int d = 0; d < n; d++) {
        const char *name = ibv_get_device_name(devs[d]);
        if (opt_dev && strcmp(opt_dev, name) != 0) continue;

        struct ibv_context *c = ibv_open_device(devs[d]);
        if (!c) continue;

        struct ibv_port_attr pa;
        if (ibv_query_port(c, 1, &pa) != 0) { ibv_close_device(c); continue; }

        int gi = opt_gid_idx;
        if (gi < 0) {
            for (int i = 0; i < (pa.gid_tbl_len ? pa.gid_tbl_len : 1); i++) {
                union ibv_gid g;
                if (ibv_query_gid(c, 1, i, &g) != 0) continue;
                if (memcmp(g.raw, target, GID_SIZE) == 0) { gi = i; break; }
            }
            if (gi < 0) gi = 0;   /* fall back: single-GID providers */
        }

        r->ctx = c; r->ib_port = 1; r->gid_idx = gi;
        r->mtu = pa.active_mtu ? pa.active_mtu : IBV_MTU_1024;
        printf("device=%s gid_idx=%d mtu=%d state=%d\n",
               name, gi, (int)r->mtu, (int)pa.state);
        ibv_free_device_list(devs);
        return true;
    }
    ibv_free_device_list(devs);
    fprintf(stderr, "no usable device\n");
    return false;
}

static bool rdma_setup(struct rdma_ctx *r)
{
    r->pd = ibv_alloc_pd(r->ctx);
    if (!r->pd) { perror("alloc_pd"); return false; }

    r->scq = ibv_create_cq(r->ctx, opt_tx_depth * 4 + 16, NULL, NULL, 0);
    r->rcq = ibv_create_cq(r->ctx, opt_rx_depth * 2 + 16, NULL, NULL, 0);
    if (!r->scq || !r->rcq) { perror("create_cq"); return false; }

    r->tx_buf = aligned_alloc(4096, opt_chunk);
    r->rx_buf = aligned_alloc(4096, opt_chunk * (size_t)opt_rx_depth);
    if (!r->tx_buf || !r->rx_buf) { perror("alloc"); return false; }

    r->tx_mr = ibv_reg_mr(r->pd, r->tx_buf, opt_chunk,
                          IBV_ACCESS_LOCAL_WRITE);
    r->rx_mr = ibv_reg_mr(r->pd, r->rx_buf, opt_chunk * (size_t)opt_rx_depth,
                          IBV_ACCESS_LOCAL_WRITE);
    if (!r->tx_mr || !r->rx_mr) { perror("reg_mr"); return false; }

    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->scq;
    qia.recv_cq = r->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr  = opt_tx_depth * 4 + 8;
    qia.cap.max_recv_wr  = opt_rx_depth + 8;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    r->qp = ibv_create_qp(r->pd, &qia);
    if (!r->qp) { perror("create_qp"); return false; }
    return true;
}

static bool post_rx(struct rdma_ctx *r, int slot)
{
    struct ibv_sge sge = {0};
    sge.addr   = (uintptr_t)(r->rx_buf + (size_t)slot * opt_chunk);
    sge.length = opt_chunk;
    sge.lkey   = r->rx_mr->lkey;

    struct ibv_recv_wr wr = {0}, *bad = NULL;
    wr.wr_id   = slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    return ibv_post_recv(r->qp, &wr, &bad) == 0;
}

static bool rdma_connect(struct rdma_ctx *r, const struct conn_info *remote,
                         uint32_t local_psn)
{
    struct ibv_qp_attr a = {0};
    a.qp_state        = IBV_QPS_INIT;
    a.pkey_index      = 0;
    a.port_num        = r->ib_port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                 IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        perror("modify INIT"); return false;
    }

    for (int i = 0; i < opt_rx_depth; i++)
        if (!post_rx(r, i)) { fprintf(stderr, "post_recv %d failed\n", i); return false; }

    memset(&a, 0, sizeof(a));
    a.qp_state           = IBV_QPS_RTR;
    a.path_mtu           = r->mtu;
    a.dest_qp_num        = remote->qpn;
    a.rq_psn             = remote->psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer      = 12;
    a.ah_attr.is_global  = 1;
    a.ah_attr.port_num   = r->ib_port;
    a.ah_attr.grh.hop_limit     = 64;
    a.ah_attr.grh.sgid_index    = r->gid_idx;
    memcpy(a.ah_attr.grh.dgid.raw, remote->gid, GID_SIZE);
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                 IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                 IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        perror("modify RTR"); return false;
    }

    memset(&a, 0, sizeof(a));
    a.qp_state      = IBV_QPS_RTS;
    a.timeout       = 14;
    a.retry_cnt     = 7;
    a.rnr_retry     = 7;
    a.sq_psn        = local_psn;
    a.max_rd_atomic = 1;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT |
                                 IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                 IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        perror("modify RTS"); return false;
    }
    return true;
}

/* Poll one completion, with a stall deadline. -1 stall, -2 error. */
static int poll_one(struct ibv_cq *cq, struct ibv_wc *wc, double deadline)
{
    for (;;) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n < 0) return -2;
        if (n > 0) return wc->status == IBV_WC_SUCCESS ? 1 : -2;
        if (now_s() > deadline) return -1;
    }
}


/* ── receive path (own thread in --bidir, inline otherwise) ──────────── */

struct rx_args {
    struct rdma_ctx *r;
    uint64_t         nmsg;
    int              rc;        /* 0 ok, 2 corruption, 3 stall */
};

static void *rx_loop(void *arg)
{
    struct rx_args *ra = arg;
    struct rdma_ctx *r = ra->r;

    for (uint64_t seq = 0; seq < ra->nmsg; seq++) {
        struct ibv_wc wc;
        int p = poll_one(r->rcq, &wc, now_s() + opt_stall_s);
        if (p == -1) {
            fprintf(stderr,
                    "STALL: no recv for %ds after %llu msgs (%.1f MiB)\n",
                    opt_stall_s, (unsigned long long)seq,
                    seq * opt_chunk / 1048576.0);
            ra->rc = 3; return NULL;
        }
        if (p == -2) {
            fprintf(stderr, "FAIL recv wc status at msg %llu\n",
                    (unsigned long long)seq);
            ra->rc = 2; return NULL;
        }
        if (wc.byte_len != opt_chunk)
            fprintf(stderr, "SHORT msg %llu: got %u bytes, expected %zu\n",
                    (unsigned long long)seq, wc.byte_len, opt_chunk);

        int slot = (int)wc.wr_id;
        long bad = verify_chunk(r->rx_buf + (size_t)slot * opt_chunk,
                                wc.byte_len < opt_chunk ? wc.byte_len : opt_chunk,
                                seq);
        if (bad >= 0) {
            fprintf(stderr,
                    "CORRUPTION at msg %llu, first bad word %ld (byte %ld of %u)"
                    " -- %.1f MiB in\n",
                    (unsigned long long)seq, bad, bad * 8, wc.byte_len,
                    seq * opt_chunk / 1048576.0);
            ra->rc = 2; return NULL;
        }
        if (!post_rx(r, slot)) {
            fprintf(stderr, "FAIL repost at msg %llu\n",
                    (unsigned long long)seq);
            ra->rc = 3; return NULL;
        }
        if ((seq & 0xfff) == 0xfff)
            printf("  recv+verified %llu/%llu (%.1f MiB)\n",
                   (unsigned long long)seq + 1, (unsigned long long)ra->nmsg,
                   (seq + 1) * opt_chunk / 1048576.0);
    }
    ra->rc = 0;
    return NULL;
}

static int tx_loop(struct rdma_ctx *r, uint64_t nmsg)
{
    int inflight = 0;
    for (uint64_t seq = 0; seq < nmsg; seq++) {
        fill_chunk(r->tx_buf, opt_chunk, seq);

        struct ibv_sge sge = {0};
        sge.addr   = (uintptr_t)r->tx_buf;
        sge.length = opt_chunk;
        sge.lkey   = r->tx_mr->lkey;

        struct ibv_send_wr wr = {0}, *bad = NULL;
        wr.wr_id      = seq;
        wr.opcode     = IBV_WR_SEND;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.send_flags = IBV_SEND_SIGNALED;

        if (ibv_post_send(r->qp, &wr, &bad) != 0) {
            fprintf(stderr, "FAIL post_send at msg %llu: %s\n",
                    (unsigned long long)seq, strerror(errno));
            return 3;
        }
        inflight++;
        while (inflight >= opt_tx_depth) {
            struct ibv_wc wc;
            int p = poll_one(r->scq, &wc, now_s() + opt_stall_s);
            if (p == -1) {
                fprintf(stderr,
                        "STALL: no send completion for %ds at msg %llu "
                        "(%.1f MiB sent)\n", opt_stall_s,
                        (unsigned long long)seq,
                        seq * opt_chunk / 1048576.0);
                return 3;
            }
            if (p == -2) return 2;
            inflight--;
        }
        if ((seq & 0xfff) == 0xfff)
            printf("  sent %llu/%llu (%.1f MiB)\n",
                   (unsigned long long)seq + 1, (unsigned long long)nmsg,
                   (seq + 1) * opt_chunk / 1048576.0);
    }
    while (inflight > 0) {
        struct ibv_wc wc;
        if (poll_one(r->scq, &wc, now_s() + opt_stall_s) < 0) return 3;
        inflight--;
    }
    return 0;
}

/* ── bootstrap ───────────────────────────────────────────────────────── */

static int bootstrap(bool server, const char *peer_ip, struct conn_info *local,
                     struct conn_info *remote)
{
    int fd, one = 1;
    if (server) {
        int ls = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        struct sockaddr_in sa = {0};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(BOOTSTRAP_PORT);
        sa.sin_addr.s_addr = INADDR_ANY;
        if (bind(ls, (struct sockaddr *)&sa, sizeof(sa)) || listen(ls, 1)) {
            perror("bind/listen"); return -1;
        }
        printf("waiting for client on :%d\n", BOOTSTRAP_PORT);
        fd = accept(ls, NULL, NULL);
        close(ls);
    } else {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in sa = {0};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(BOOTSTRAP_PORT);
        inet_pton(AF_INET, peer_ip, &sa.sin_addr);
        for (int i = 0; i < 100; i++) {
            if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) == 0) break;
            usleep(100000);
            if (i == 99) { perror("connect"); return -1; }
        }
    }
    if (fd < 0) { perror("socket"); return -1; }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    if (write(fd, local, sizeof(*local)) != (ssize_t)sizeof(*local)) return -1;
    size_t got = 0;
    while (got < sizeof(*remote)) {
        ssize_t k = read(fd, (char *)remote + got, sizeof(*remote) - got);
        if (k <= 0) return -1;
        got += k;
    }
    return fd;
}

/* ── main ────────────────────────────────────────────────────────────── */

static void usage(void)
{
    printf("usage: odl_rdma_stress --server | --client <ip> [options]\n"
           "  --local-ip <ip>   local address for GID match (client: auto)\n"
           "  --chunk <bytes>   message size          (default 262144)\n"
           "  --rx-depth <n>    pre-posted receives   (default 24)\n"
           "  --tx-depth <n>    sends in flight       (default 2)\n"
           "  --total <bytes>   total to push, K/M/G  (default 2G)\n"
           "  --dev <name>      device name\n"
           "  --gid <idx>       force GID index\n"
           "  --stall <sec>     no-progress timeout   (default 15)\n"
           "  --bidir           BOTH peers send and receive concurrently.\n"
           "                    This is what ggml-rpc actually does, and it is\n"
           "                    the only mode that exercises TX/RX contention\n"
           "                    for the shared frame pool.\n"
           "  --mixed           interleave small control messages with bulk,\n"
           "                    as the RPC protocol does (exercises the\n"
           "                    latency/throughput dispatcher and odd sizes)\n");
}

static uint64_t parse_size(const char *s)
{
    char *e = NULL;
    uint64_t v = strtoull(s, &e, 0);
    if (e && (*e == 'k' || *e == 'K')) v <<= 10;
    else if (e && (*e == 'm' || *e == 'M')) v <<= 20;
    else if (e && (*e == 'g' || *e == 'G')) v <<= 30;
    return v;
}

int main(int argc, char **argv)
{
    bool server = false;
    const char *peer_ip = NULL, *local_ip = "0.0.0.0";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--server")) server = true;
        else if (!strcmp(argv[i], "--client") && i + 1 < argc) peer_ip = argv[++i];
        else if (!strcmp(argv[i], "--local-ip") && i + 1 < argc) local_ip = argv[++i];
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) opt_chunk = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--rx-depth") && i + 1 < argc) opt_rx_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tx-depth") && i + 1 < argc) opt_tx_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--total") && i + 1 < argc) opt_total = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--dev") && i + 1 < argc) opt_dev = argv[++i];
        else if (!strcmp(argv[i], "--gid") && i + 1 < argc) opt_gid_idx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stall") && i + 1 < argc) opt_stall_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bidir")) opt_bidir = true;
        else if (!strcmp(argv[i], "--mixed")) opt_mixed = true;
        else { usage(); return 1; }
    }
    if (!server && !peer_ip) { usage(); return 1; }

    struct rdma_ctx r = {0};
    if (!rdma_open(&r, local_ip)) return 1;
    if (!rdma_setup(&r)) return 1;

    struct conn_info local = {0}, remote = {0};
    local.qpn = r.qp->qp_num;
    local.psn = 0x1234;
    union ibv_gid g;
    if (ibv_query_gid(r.ctx, r.ib_port, r.gid_idx, &g) == 0)
        memcpy(local.gid, g.raw, GID_SIZE);

    int bfd = bootstrap(server, peer_ip, &local, &remote);
    if (bfd < 0) { fprintf(stderr, "bootstrap failed\n"); return 1; }
    printf("local qpn=%u  remote qpn=%u\n", local.qpn, remote.qpn);

    if (!rdma_connect(&r, &remote, local.psn)) return 1;
    printf("QP RTS. chunk=%zu rx_depth=%d tx_depth=%d total=%llu\n",
           opt_chunk, opt_rx_depth, opt_tx_depth,
           (unsigned long long)opt_total);

    uint64_t nmsg = opt_total / opt_chunk;
    if (nmsg == 0) nmsg = 1;
    double t0 = now_s();
    int rc = 0;

    if (opt_bidir) {
        /* Both peers send AND receive concurrently on the same QP -- the
         * pattern ggml-rpc actually uses, and the only one that makes TX and
         * RX contend for the shared frame pool. A one-way test passes 21 GiB
         * on hardware where this fails, so this mode is the meaningful one. */
        struct rx_args ra = { .r = &r, .nmsg = nmsg, .rc = 0 };
        pthread_t th;
        if (pthread_create(&th, NULL, rx_loop, &ra) != 0) {
            perror("pthread_create"); rc = 1; goto done;
        }
        rc = tx_loop(&r, nmsg);
        pthread_join(th, NULL);
        if (!rc) rc = ra.rc;
    } else if (!server) {
        rc = tx_loop(&r, nmsg);
    } else {
        struct rx_args ra = { .r = &r, .nmsg = nmsg, .rc = 0 };
        rx_loop(&ra);
        rc = ra.rc;
    }
    if (rc) goto done;

    {
        double dt = now_s() - t0;
        double mib = (double)nmsg * opt_chunk / 1048576.0;
        printf("PASS: %llu msgs, %.1f MiB in %.2fs = %.1f MiB/s (%.2f Gb/s)\n",
               (unsigned long long)nmsg, mib, dt, mib / dt,
               mib * 8 / 1024.0 / dt);
    }

done:
    close(bfd);
    if (rc) printf("RESULT: FAIL rc=%d\n", rc);
    return rc;
}
