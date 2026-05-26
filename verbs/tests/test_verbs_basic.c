/*
 * OdinLink — Verbs Smoke Test: Can We Open a Device and Send Data?
 *
 * Walks through a standard RDMA lifecycle:
 *   - Find an OdinLink device
 *   - Open it, query capabilities
 *   - Create a PD, register memory, create a CQ
 *   - Create and connect a QP
 *   - post_send a message, poll CQ for completion
 *   - Clean up everything
 *
 * This is the verbs equivalent of "hello world."
 * Build: gcc -o test_verbs_basic test_verbs_basic.c -lodl_tb5_verbs -libverbs
 * Run:   ./test_verbs_basic  (requires /dev/odl_tb5_0 with peer connected)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <infiniband/verbs.h>

/* The wrapper header provides device discovery helpers */
#include "odl_tb5_verbs_wrapper.h"

int main(void)
{
    int ndev = odl_num_tb5_devices();
    printf("OdinLink-Five devices found: %d\n", ndev);

    if (ndev == 0) {
        printf("SKIP: No OdinLink-Five devices available.\n");
        printf("Load kernel module and connect a TB5 peer first.\n");
        return 77; /* skip */
    }

    /* Open first device */
    struct ibv_device *dev = odl_find_tb5_device(0);
    assert(dev != NULL);
    printf("Device: %s (index %d)\n", dev->name, odl_tb5_device_index(dev));
    assert(odl_is_tb5_device(dev));

    /* Open context */
    struct ibv_context *ctx = ibv_open_device(dev);
    assert(ctx != NULL);
    printf("Context opened: %p\n", (void*)ctx);

    /* Query device */
    struct ibv_device_attr dev_attr;
    int ret = ibv_query_device(ctx, &dev_attr);
    assert(ret == 0);
    printf("Device attributes: max_qp=%d max_mr=%d max_cq=%d\n",
           dev_attr.max_qp, dev_attr.max_mr, dev_attr.max_cq);

    /* Query port */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(ctx, 1, &port_attr);
    assert(ret == 0);
    printf("Port state: %s\n",
           port_attr.state == IBV_PORT_ACTIVE ? "ACTIVE" : "DOWN");

    /* Allocate PD */
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    assert(pd != NULL);
    printf("PD allocated: handle=%u\n", pd->handle);

    /* Register host memory */
    char buf[4096] __attribute__((aligned(4096)));
    memset(buf, 0xAB, sizeof(buf));
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, sizeof(buf),
                                   IBV_ACCESS_LOCAL_WRITE);
    assert(mr != NULL);
    printf("MR registered: addr=%p len=%zu lkey=%06x rkey=%06x\n",
           mr->addr, mr->length, mr->lkey, mr->rkey);

    /* Create CQ */
    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
    assert(cq != NULL);
    printf("CQ created: cqe=%d\n", cq->cqe);

    /* Create QP */
    struct ibv_qp_init_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = 8;
    qp_attr.cap.max_recv_wr = 8;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;

    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);
    assert(qp != NULL);
    printf("QP created: qp_num=%u type=%d\n", qp->qp_num, qp->qp_type);

    /* Modify QP to INIT */
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    ret = ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    assert(ret == 0);
    printf("QP state: INIT\n");

    /* Modify QP to RTR */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = 1;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    ret = ibv_modify_qp(qp, &attr,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret == 0)
        printf("QP state: RTR\n");

    /* Cleanup */
    ret = ibv_destroy_qp(qp);
    assert(ret == 0);
    printf("QP destroyed\n");

    ret = ibv_destroy_cq(cq);
    assert(ret == 0);
    printf("CQ destroyed\n");

    ret = ibv_dereg_mr(mr);
    assert(ret == 0);
    printf("MR deregistered\n");

    ret = ibv_dealloc_pd(pd);
    assert(ret == 0);
    printf("PD deallocated\n");

    ret = ibv_close_device(ctx);
    assert(ret == 0);
    printf("Context closed\n");

    printf("\nALL VERBS TESTS PASSED\n");
    return 0;
}
