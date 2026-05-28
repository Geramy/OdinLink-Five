/* SPDX-License-Identifier: MIT */
/*
 * OdinLink — Transport Backend Interface
 *
 * A "transport" is the hardware-specific layer that moves bytes over
 * Thunderbolt.  The NHI (Intel's DMA engine) is one transport.  A future
 * Apple-Silicon backend would be another.
 *
 * The rest of the driver — character device, streams, frame pools,
 * protocol handshake — is transport-agnostic.  It calls these ops to
 * allocate rings, submit frames, and get a DMA device for coherent
 * memory.
 *
 * Adding a new backend:
 *   1. Implement every op in this struct
 *   2. Add a transport_type enum value
 *   3. Wire it in odl_tb5_service.c probe() or loopback init
 */
#ifndef ODL_TB5_TRANSPORT_H
#define ODL_TB5_TRANSPORT_H

#include <linux/types.h>

struct odl_tb5_device;
struct ring_frame;
struct device;

enum odl_tb5_transport_type {
	ODL_TB5_TRANSPORT_NHI = 0,
	ODL_TB5_TRANSPORT_LOOPBACK = 1,
	ODL_TB5_TRANSPORT_APPLE = 2,
};

#define ODL_TB5_ENABLE_RETRIES 5
#define ODL_TB5_ENABLE_DELAY   200

struct odl_tb5_transport_ring_info {
	int hop;
};

struct odl_tb5_transport_ops {
	enum odl_tb5_transport_type type;
	const char *name;

	int  (*ring_alloc)(struct odl_tb5_device *dev);
	void (*ring_free)(struct odl_tb5_device *dev);
	int  (*ring_start)(struct odl_tb5_device *dev);
	void (*ring_stop)(struct odl_tb5_device *dev);
	void (*ring_reset)(struct odl_tb5_device *dev);

	int  (*ring_tx)(struct odl_tb5_device *dev, struct ring_frame *frame);
	int  (*ring_rx)(struct odl_tb5_device *dev, struct ring_frame *frame);

	struct device *(*dma_device)(struct odl_tb5_device *dev);
	struct odl_tb5_transport_ring_info (*tx_ring_info)(
		struct odl_tb5_device *dev);
	struct odl_tb5_transport_ring_info (*rx_ring_info)(
		struct odl_tb5_device *dev);

	int  (*local_tx_hopid)(struct odl_tb5_device *dev);

	int  (*path_enable)(struct odl_tb5_device *dev);
	void (*path_disable)(struct odl_tb5_device *dev);

	int  (*peer_send_login)(struct odl_tb5_device *dev);
	int  (*peer_send_logout)(struct odl_tb5_device *dev);

	void (*kick_tx)(struct odl_tb5_device *dev);
	void (*kick_rx)(struct odl_tb5_device *dev);
};

#endif
