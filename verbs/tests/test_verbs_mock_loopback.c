/*
 * OdinLink — Verbs Loopback Test (No Hardware Needed)
 *
 * Uses the mock library (libodl_tb5_mock.so) so you can test the
 * full verbs API without a Thunderbolt cable or even the kernel
 * module. Two simulated "peers" on the same machine talk to each
 * other via shared memory.
 *
 * Full lifecycle: open device → PD → MR → CQ → QP → post_send →
 * poll_cq → cleanup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include "odl_tb5_verbs_wrapper.h"

static int failures = 0;

#define TEST(name) do { \
    printf("  TEST: %-45s ", name); \
    fflush(stdout); \
} while (0)

#define PASS() do { \
    printf("PASS\n"); \
} while (0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    failures++; \
} while (0)

int main(void)
{
    printf("\n=== OdinLink Verbs Provider Mock Loopback Test ===\n\n");

    /* ── Device Discovery ──────────────────────────────────────── */

    TEST("odl_num_tb5_devices()");
    int ndev = odl_num_tb5_devices();
    if (ndev > 0) PASS(); else FAIL("no mock devices");

    TEST("odl_find_tb5_device(0)");
    struct ibv_device *dev = odl_find_tb5_device(0);
    if (dev) PASS(); else FAIL("device not found");

    TEST("odl_is_tb5_device(dev)");
    if (odl_is_tb5_device(dev)) PASS(); else FAIL("not an ODL device");

    TEST("odl_tb5_device_index(dev)");
    if (odl_tb5_device_index(dev) == 0) PASS(); else FAIL("wrong index");

    /* ── Context ───────────────────────────────────────────────── */

    TEST("ibv_open_device(dev)");
    struct ibv_context *ctx = ibv_open_device(dev);
    if (ctx) PASS(); else FAIL("open failed");

    TEST("ibv_query_device(ctx, &attr)");
    struct ibv_device_attr dev_attr;
    if (ibv_query_device(ctx, &dev_attr) == 0) PASS();
    else FAIL("query_device failed");

    printf("    max_qp=%d max_mr=%d max_cq=%d max_pd=%d\n",
           dev_attr.max_qp, dev_attr.max_mr,
           dev_attr.max_cq, dev_attr.max_pd);

    TEST("ibv_query_port(ctx, 1, &attr)");
    struct ibv_port_attr port_attr;
    if (ibv_query_port(ctx, 1, &port_attr) == 0) PASS();
    else FAIL("query_port failed");

    printf("    port_state=%s mtu=%d\n",
           port_attr.state == IBV_PORT_ACTIVE ? "ACTIVE" : "DOWN",
           port_attr.active_mtu);

    /* ── PD ────────────────────────────────────────────────────── */

    TEST("ibv_alloc_pd(ctx)");
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (pd) PASS(); else FAIL("alloc_pd failed");

    /* ── MR ────────────────────────────────────────────────────── */

    char send_buf[2048] __attribute__((aligned(4096)));
    memset(send_buf, 0x42, sizeof(send_buf));

    TEST("ibv_reg_mr(pd, send_buf, LOCAL_WRITE)");
    struct ibv_mr *mr = ibv_reg_mr(pd, send_buf, sizeof(send_buf),
                                    IBV_ACCESS_LOCAL_WRITE);
    if (mr) PASS(); else FAIL("reg_mr failed");

    printf("    lkey=%06x rkey=%06x len=%zu\n",
           mr->lkey, mr->rkey, mr->length);

    /* ── CQ ────────────────────────────────────────────────────── */

    TEST("ibv_create_cq(ctx, 256, NULL, NULL, 0)");
    struct ibv_cq *cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);
    if (cq) PASS(); else FAIL("create_cq failed");

    printf("    cqe=%d\n", cq->cqe);

    /* ── QP ────────────────────────────────────────────────────── */

    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = 128;
    qp_attr.cap.max_recv_wr = 8;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;

    TEST("ibv_create_qp(pd, RC)");
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);
    if (qp) PASS(); else FAIL("create_qp failed");

    TEST("QP reports requested 128-entry send queue");
    if (qp_attr.cap.max_send_wr == 128) PASS();
    else FAIL("provider did not report the real send queue depth");

    printf("    qp_num=%u\n", qp->qp_num);

    /* ── Modify QP: RESET → INIT ───────────────────────────────── */

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;

    TEST("ibv_modify_qp(RESET -> INIT)");
    int ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                            IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret == 0) PASS(); else FAIL("modify_qp INIT failed");

    /* ── Modify QP: INIT → RTR ─────────────────────────────────── */

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = 1;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    TEST("ibv_modify_qp(INIT -> RTR)");
    ret = ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret == 0) PASS(); else FAIL("modify_qp RTR failed");

    /* ── Modify QP: RTR → RTS ──────────────────────────────────── */

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;

    TEST("ibv_modify_qp(RTR -> RTS)");
    ret = ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_SQ_PSN);
    if (ret == 0) PASS(); else FAIL("modify_qp RTS failed");

    /* ── Post Send ─────────────────────────────────────────────── */

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)send_buf;
    sge.length = sizeof(send_buf);
    sge.lkey = mr->lkey;

    struct ibv_send_wr send_wr;
    struct ibv_send_wr *bad_wr;
    memset(&send_wr, 0, sizeof(send_wr));
    send_wr.wr_id = 42;
    send_wr.next = NULL;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    TEST("ibv_post_send(qp, SEND)");
    ret = ibv_post_send(qp, &send_wr, &bad_wr);
    if (ret == 0) PASS(); else FAIL("post_send failed");

    /* ── Poll CQ ───────────────────────────────────────────────── */

    struct ibv_wc wc;
    TEST("ibv_poll_cq(cq, 1, &wc)");

    int polled = 0;
    for (int i = 0; i < 100 && polled == 0; i++) {
        polled = ibv_poll_cq(cq, 1, &wc);
        if (polled == 0) usleep(1000);
    }

    if (polled == 1) {
        PASS();
        printf("    wr_id=%lu status=%s opcode=%d byte_len=%u qp_num=%u\n",
               (unsigned long)wc.wr_id,
               wc.status == IBV_WC_SUCCESS ? "SUCCESS" : "ERROR",
               wc.opcode, wc.byte_len, wc.qp_num);
    } else {
        FAIL("no completion received");
    }

    /* Post a full, ordered batch. With ODL_MOCK_SEND_EAGAIN_AT=2 the first
     * request in this batch fails once in the worker. The historical
     * tail requeue completed it last; a concurrent refill could also drop it.
     * Requiring every completion in order catches both failures. */
    enum { EAGAIN_BATCH = 128 };
    struct ibv_sge batch_sge[EAGAIN_BATCH];
    struct ibv_send_wr batch_wr[EAGAIN_BATCH];
    memset(batch_sge, 0, sizeof(batch_sge));
    memset(batch_wr, 0, sizeof(batch_wr));
    for (int i = 0; i < EAGAIN_BATCH; i++) {
        batch_sge[i] = sge;
        batch_wr[i].wr_id = 1000 + (uint64_t)i;
        batch_wr[i].sg_list = &batch_sge[i];
        batch_wr[i].num_sge = 1;
        batch_wr[i].opcode = IBV_WR_SEND;
        batch_wr[i].send_flags = IBV_SEND_SIGNALED;
        batch_wr[i].next = i + 1 < EAGAIN_BATCH ? &batch_wr[i + 1] : NULL;
    }

    TEST("EAGAIN preserves a full SQ batch without reorder or drop");
    bad_wr = NULL;
    ret = ibv_post_send(qp, &batch_wr[0], &bad_wr);
    int completed = 0;
    bool ordered = ret == 0;
    for (int waits = 0; ordered && completed < EAGAIN_BATCH && waits < 5000; waits++) {
        struct ibv_wc batch_wc[16];
        int n = ibv_poll_cq(cq, 16, batch_wc);
        if (n < 0) {
            ordered = false;
            break;
        }
        if (n == 0) {
            usleep(1000);
            continue;
        }
        for (int j = 0; j < n; j++, completed++) {
            if (batch_wc[j].status != IBV_WC_SUCCESS ||
                batch_wc[j].wr_id != 1000 + (uint64_t)completed) {
                ordered = false;
                break;
            }
        }
    }
    if (ordered && completed == EAGAIN_BATCH) PASS();
    else FAIL("send completions were reordered, dropped, or timed out");

    /* ── Cleanup ────────────────────────────────────────────────── */

    TEST("ibv_destroy_qp(qp)");
    if (ibv_destroy_qp(qp) == 0) PASS(); else FAIL("destroy_qp failed");

    TEST("ibv_destroy_cq(cq)");
    if (ibv_destroy_cq(cq) == 0) PASS(); else FAIL("destroy_cq failed");

    TEST("ibv_dereg_mr(mr)");
    if (ibv_dereg_mr(mr) == 0) PASS(); else FAIL("dereg_mr failed");

    TEST("ibv_dealloc_pd(pd)");
    if (ibv_dealloc_pd(pd) == 0) PASS(); else FAIL("dealloc_pd failed");

    TEST("ibv_close_device(ctx)");
    if (ibv_close_device(ctx) == 0) PASS(); else FAIL("close_device failed");

    /* ── Summary ────────────────────────────────────────────────── */

    printf("\n=== Results: ");
    if (failures == 0) {
        printf("ALL TESTS PASSED ===\n");
        return 0;
    } else {
        printf("%d TEST(S) FAILED ===\n", failures);
        return 1;
    }
}
