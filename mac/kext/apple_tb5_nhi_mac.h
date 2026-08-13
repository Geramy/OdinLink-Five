/* SPDX-License-Identifier: MIT */
/*
 * Apple ACIO NHI register map — Mac kext copy.
 *
 * Same numbers as driver/apple_tb5_nhi_regs.h, without Linux headers.
 * Unverified on silicon. The kext will not write these unless the user
 * explicitly arms hardware (kOdinLinkArmHardware).
 */
#ifndef APPLE_TB5_NHI_MAC_H
#define APPLE_TB5_NHI_MAC_H

#include <stdint.h>

#define APPLE_TB5_BIT(n)		(1u << (n))
#define APPLE_TB5_GENMASK(h, l)	\
	(((~0u) << (l)) & (~0u >> (31 - (h))))

#define APPLE_TB5_RING_DESC_ADDR_LO	0x00
#define APPLE_TB5_RING_DESC_ADDR_HI	0x04
#define APPLE_TB5_RING_DESC_INDEX	0x08
#define APPLE_TB5_RING_DESC_SIZE_HOPID	0x0C
#define APPLE_TB5_RING_DESC_STRIDE	0x10

#define APPLE_TB5_INDEX_MASK		0xFFFFu
#define APPLE_TB5_SIZE_RING_MASK	APPLE_TB5_GENMASK(11, 0)
#define APPLE_TB5_SIZE_HOPID_SHIFT	16
#define APPLE_TB5_SIZE_HOPID_MASK	APPLE_TB5_GENMASK(27, 16)

#define APPLE_TB5_HOP_CTRL_STRIDE	0x10
#define APPLE_TB5_CTRL_ENABLE		APPLE_TB5_BIT(31)
#define APPLE_TB5_CTRL_INT_ON_DESC	APPLE_TB5_BIT(29)
#define APPLE_TB5_CTRL_HOPID_SHIFT	16

#define APPLE_TB5_INT_RX_MASK_BASE	0xD100
#define APPLE_TB5_INT_RX_ROUTE_BASE	0xD16C

#define APPLE_TB5_PDF_SOF_BASE		0x1000
#define APPLE_TB5_PDF_SOF_HOP_STRIDE	0x20
#define APPLE_TB5_PDF_EOF_BASE		0x1008
#define APPLE_TB5_PDF_EOF_HOP_STRIDE	0x20

struct apple_tb5_dma_desc {
	uint32_t addr_lo;
	uint32_t addr_hi;
	uint32_t control;
	uint32_t reserved;
};

#define APPLE_TB5_DESC_CTRL_SOF		APPLE_TB5_BIT(0)
#define APPLE_TB5_DESC_CTRL_EOF		APPLE_TB5_BIT(1)
#define APPLE_TB5_DESC_CTRL_INT_EN	APPLE_TB5_BIT(2)
#define APPLE_TB5_DESC_CTRL_LEN_SHIFT	16
#define APPLE_TB5_DESC_CTRL_LEN_MASK	APPLE_TB5_GENMASK(31, 16)

#define APPLE_TB5_ACIO_TX_DESC_BASE	0x0000
#define APPLE_TB5_ACIO_TX_HOP_CTRL	0x4000
#define APPLE_TB5_ACIO_RX_DESC_BASE	0x8000
#define APPLE_TB5_ACIO_RX_HOP_CTRL	0xC000

#define APPLE_TB5_ACIO_PHYS_BASE	0x40100000ull
#define APPLE_TB5_ACIO_SIZE		0x100000u
#define APPLE_TB5_DMA_BITS		32

#endif /* APPLE_TB5_NHI_MAC_H */
