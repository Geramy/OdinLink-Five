/* SPDX-License-Identifier: MIT */
/*
 * Apple Silicon Thunderbolt NHI Register Map
 *
 * Hardware register definitions for the Apple ACIO Thunderbolt fabric,
 * determined from analysis of the macOS AppleThunderboltNHI kext
 * (7.2.81, arm64e, macOS 26.5, t8132/M4 Pro).
 *
 * This header is standalone — it has no dependency on the OdinLink
 * driver internals. Any kernel driver for Apple Silicon Thunderbolt
 * can include it.
 *
 * Sources:
 *   - AppleThunderboltNHIReceiveRing::startDMA
 *   - AppleThunderboltNHIReceiveRing::stopDMA
 *   - AppleThunderboltNHITransmitRing::writeProducerIndexInternal
 *   - AppleThunderboltNHIReceiveRing::writeConsumerIndexInternal
 *   - IOThunderboltTransmitCommand::setDescCache
 *   - AppleThunderboltHALGenericACIO::initRegisterLayout
 *   - AppleThunderboltHALGenericACIO::setupRegisterRanges
 *   - AppleThunderboltNHITransmitRingManagerGenericACIO::enableInterrupt
 *   - AppleThunderboltNHIReceiveRingManagerGenericACIO::enableInterrupt
 *   - AppleThunderboltNHIReceiveRing::setPDFBitmasks
 */
#ifndef APPLE_TB5_NHI_REGS_H
#define APPLE_TB5_NHI_REGS_H

#include <linux/bits.h>
#include <linux/bitfield.h>
#include <linux/types.h>

/* ── Register layout version ────────────────────────────────────────── */

/*
 * initRegisterLayout returns a version constant:
 *   - AppleThunderboltGenericHAL:      0xE000_02C7
 *   - AppleThunderboltHALGenericACIO:   writes 128-bit layout blob
 *     to offsets +0x134, +0x144, +0x164 of the HAL object.
 *
 * The layout blob is a firmware tunable that tells the HAL where each
 * ring's descriptor and control registers live in MMIO space. See
 * apple_tb5_acio_layout below.
 */
#define APPLE_TB5_REG_LAYOUT_ID		0xE00002C7

/* ── Per-ring descriptor registers (4 x u32, stride 0x10) ─────────── */

/*
 * Each DMA ring has a 16-byte register set. The base address is
 * determined by the ACIO firmware tunable layout blob (see
 * apple_tb5_acio_layout). The stride between consecutive ring register
 * sets is 0x10 (one row of 4 x u32).
 *
 * From startDMA:
 *   registerWrite32(ring_desc_base + 0x00, addr_lo)
 *   registerWrite32(ring_desc_base + 0x04, addr_hi)
 *   registerWrite32(ring_desc_base + 0x08, credit_count / prod_idx)
 *   registerWrite32(ring_desc_base + 0x0C, size | (hopid << 16))
 */
#define APPLE_TB5_RING_DESC_ADDR_LO	0x00
#define APPLE_TB5_RING_DESC_ADDR_HI	0x04
#define APPLE_TB5_RING_DESC_INDEX	0x08
#define APPLE_TB5_RING_DESC_SIZE_HOPID	0x0C
#define APPLE_TB5_RING_DESC_STRIDE	0x10

/* Index register: if isIndexWriteValid (vtable +0x998), write
 * index & 0xFFFF. Otherwise the ring is full and the hardware
 * uses bits 16+ as an overflow indicator. */
#define APPLE_TB5_INDEX_MASK		0xFFFF
#define APPLE_TB5_INDEX_SHIFT		16

/* Size register: ring size in descriptors (bits 11:0),
 * HopID packed into bits 27:16.
 * From startDMA: bfi at bit 0x10, width 0xc. */
#define APPLE_TB5_SIZE_RING_MASK	GENMASK(11, 0)
#define APPLE_TB5_SIZE_HOPID_SHIFT	16
#define APPLE_TB5_SIZE_HOPID_MASK	GENMASK(27, 16)

/* ── Per-hop control registers (stride 0x10) ──────────────────────── */

/*
 * Each HopID path has a control register at hop_ctrl_base +
 * (hop_id * stride). The control word determines whether the hop
 * path is active, whether it generates interrupts, etc.
 *
 * From startDMA and stopDMA:
 *   startDMA: reads ctrl, ORs in ENABLE | COALESCE? | INT_ON_DESC? |
 *             TWO_PAGE? | (hopid << 16), writes back
 *   stopDMA:  reads ctrl, ANDs with ~ENABLE (clears bit 31), writes back
 */
#define APPLE_TB5_HOP_CTRL_REG		0x00
#define APPLE_TB5_HOP_CTRL_STRIDE	0x10

#define APPLE_TB5_CTRL_ENABLE		BIT(31)
#define APPLE_TB5_CTRL_COALESCE	BIT(30)
#define APPLE_TB5_CTRL_INT_ON_DESC	BIT(29)
#define APPLE_TB5_CTRL_TWO_PAGE		BIT(28)
#define APPLE_TB5_CTRL_HOPID_SHIFT	16
#define APPLE_TB5_CTRL_HOPID_MASK	GENMASK(27, 16)
#define APPLE_TB5_CTRL_FRAME_SZ_SHIFT	12
#define APPLE_TB5_CTRL_FRAME_SZ_MASK	GENMASK(22, 12)

/* ── NHI-level interrupt registers ────────────────────────────────── */

/*
 * From getInterruptThrottlingRate:  offset +0xD4
 * From getInterruptVacancyControl:  offset +0xE4
 * From getEnabledTxRingMask:        0xFFFFFF00 (generic), +0xE8 (ACIO)
 * From getEnabledRxRingMask:        0xFFFFFF00 (generic), +0xE8 (ACIO)
 *
 * The interrupt routing model uses per-HopID interrupt mask registers.
 * enableInterrupt writes to:
 *   - For HopID <= 11: registerWrite32(base + 0xD000 + (hopid * 4), mask)
 *   - For HopID > 11:  same formula but the mask is shifted
 *
 * The interrupt mask is a 32-bit value:
 *   - Bit enable: (1 << (hopid & 0x1F))
 *   - Bit disable: ~(1 << (hopid & 0x1F))
 *   - Quad select: hopid >> 5 (each 32-bit register covers 32 HopIDs)
 *   - Register offset: 0xD000 + (quad * 4) for TX, 0xD06C + (quad * 4) for RX
 *
 * From enableInterrupt (TX ACIO variant):
 *   - Read current mask: registerRead32(NHI_BASE + 0xD000 + (hopid >> 5) * 4)
 *   - Enable:  mask |=  (1 << (hopid & 0x1F))
 *   - Disable: mask &= ~(1 << (hopid & 0x1F))
 *   - Write:   registerWrite32(NHI_BASE + 0xD000 + (hopid >> 5) * 4, mask)
 *   - Then write interrupt routing: registerWrite32(NHI_BASE + 0xD000 + 0x6C + hopid*4, vector & 0xFFFF)
 *
 * The 0xD000 area is the TX interrupt mask array.
 * The 0xD06C area is the TX interrupt routing array (per-hopid to MSI-X vector mapping).
 * The RX equivalents start at different offsets.
 */
#define APPLE_TB5_INT_THROTTLE_REG	0x00D4
#define APPLE_TB5_INT_VACANCY_REG	0x00E4
#define APPLE_TB5_INT_ENABLED_MASK	0x00E8

#define APPLE_TB5_INT_TX_MASK_BASE	0xD000
#define APPLE_TB5_INT_TX_MASK_STRIDE	4
#define APPLE_TB5_INT_TX_ROUTE_BASE	0xD06C
#define APPLE_TB5_INT_TX_ROUTE_STRIDE	4

#define APPLE_TB5_INT_RX_MASK_BASE	0xD100
#define APPLE_TB5_INT_RX_MASK_STRIDE	4
#define APPLE_TB5_INT_RX_ROUTE_BASE	0xD16C
#define APPLE_TB5_INT_RX_ROUTE_STRIDE	4

/* Each 32-bit mask register covers 32 HopIDs.
 * Quad = hopid / 32, bit = hopid % 32. */
#define APPLE_TB5_INT_QUAD_SHIFT	5
#define APPLE_TB5_INT_QUAD_MASK		GENMASK(2, 0)
#define APPLE_TB5_INT_BIT_MASK		GENMASK(4, 0)

/* Maximum HopID (from enableInterrupt: "cmp w20, #0xb" = 11 max
 * for the fast-path; HopIDs > 11 use a different code path).
 * The actual max depends on the hardware generation. */
#define APPLE_TB5_MAX_HOPID_FAST	11
#define APPLE_TB5_MAX_HOPID		63

/* Ring interrupt enable/disable sentinel values.
 * enableInterrupt is called with (hopid, enable) where enable is 0 or 1.
 * When enable=0 and the ring's DART mapping fails (isIndexWriteValid
 * returns false at vtable +0xB40), the mask bit is cleared with
 * (1 << 12) << hopid rather than (1 << hopid). */
#define APPLE_TB5_INT_DISABLE_SHIFT	12

/* ── PDF (Packet Descriptor Format) bitmask registers ─────────────── */

/*
 * setPDFBitmasks writes per-HopID PDF filter masks. The PDF bitmask
 * determines which packet types (SOF/EOF codes) are delivered to this
 * ring. Two 32-bit masks are written:
 *   - registerWrite32(NHI_BASE + 0x1000 + (hopid * 0x20), sof_mask)
 *   - registerWrite32(NHI_BASE + 0x1008 + (hopid * 0x20), eof_mask)
 */
#define APPLE_TB5_PDF_SOF_BASE		0x1000
#define APPLE_TB5_PDF_SOF_HOP_STRIDE	0x20
#define APPLE_TB5_PDF_EOF_BASE		0x1008
#define APPLE_TB5_PDF_EOF_HOP_STRIDE	0x20

/* ── DMA descriptor format (16 bytes) ──────────────────────────────── */

/*
 * From setDescCache analysis:
 *
 *   IOThunderboltTransmitCommand::setDescCache(TxBufferDescriptor*, uint64):
 *     ldr  x9, [x1]         // load 8 bytes = DMA addr from descriptor object
 *     str  x9, [ring + idx*16]  // store as first 8 bytes of ring entry
 *     ldr  w9, [x1, #8]     // load 4 bytes = control word
 *     str  w9, [ring + idx*16 + 8]  // store as bytes 8-11
 *     // word[3] (bytes 12-15) is never written — reserved/zero
 *
 * The layout is:
 *   bytes 0-3:   DMA address low 32 bits
 *   bytes 4-7:   DMA address high 32 bits
 *   bytes 8-11:  control word
 *   bytes 12-15: reserved (zero)
 *
 * Control word bits (inferred from Intel NHI conventions and the
 * logRings debug format string "desc[%d] 0x%08x 0x%08x 0x%08x 0x%08x"):
 *   bit 0:   SOF (start of frame)
 *   bit 1:   EOF (end of frame)
 *   bit 2:   INT_EN (interrupt on completion)
 *   bits 16-31: frame byte length
 *
 * NOTE: The exact Apple control word bit positions are NOT confirmed
 * from analysis. The setDescCache function copies the control word
 * verbatim from the BufferDescriptor object without decoding it.
 * The bit assignments below match Intel NHI conventions and may
 * differ on Apple hardware. The logRings format string confirms
 * 4 x 32-bit words per descriptor, but not the bit layout of word[2].
 * These will need to be verified on real hardware.
 */
struct apple_tb5_dma_desc {
	__le32	addr_lo;
	__le32	addr_hi;
	__le32	control;
	__le32	reserved;
};

#define APPLE_TB5_DESC_CTRL_SOF		BIT(0)
#define APPLE_TB5_DESC_CTRL_EOF		BIT(1)
#define APPLE_TB5_DESC_CTRL_INT_EN		BIT(2)
#define APPLE_TB5_DESC_CTRL_LEN_SHIFT		16
#define APPLE_TB5_DESC_CTRL_LEN_MASK		GENMASK(31, 16)

/* ── kdebug trace IDs (from symbol table) ──────────────────────────── */

#define APPLE_TB5_KDBG_CLASS		0x534
#define APPLE_TB5_KDBG_START_DMA	0x05344034
#define APPLE_TB5_KDBG_TX_PRODUCER	0x053440B4
#define APPLE_TB5_KDBG_RX_CONSUMER	0x05344050
#define APPLE_TB5_KDBG_COMPLETE		0x05344024
#define APPLE_TB5_KDBG_STOP_DMA		0x05344024

/* ── ACIO register layout blob ────────────────────────────────────── */

/*
 * initRegisterLayout (AppleThunderboltHALGenericACIO) writes a 128-bit
 * layout blob to the HAL object at offsets +0x134, +0x144, +0x164.
 * The blob is loaded from a firmware tunable at a fixed address.
 *
 * setupRegisterRanges reads the layout blob and computes the
 * ring_desc_base and hop_ctrl_base for each ring. The HAL object
 * stores rings at offset +0x290, with stride 0x20 per ring entry.
 * Each ring entry (0x20 bytes) contains:
 *   +0x00: ring object pointer
 *   +0x08: hop_id (from HAL vtable +0x708, ring count from +0x710)
 *   +0x10: ring_desc_base (from registerRead32 +0x138)
 *   +0x18: hop_ctrl_base (from registerRead32 +0x138 of another area)
 *
 * The layout is hardware-generation specific. For the ACIO fabric
 * (M4 Pro, t8132), the defaults are:
 *   TX ring descriptors: 0x0000
 *   TX hop control:      0x4000
 *   RX ring descriptors: 0x8000
 *   RX hop control:      0xC000
 *
 * These defaults may be overridden by the tunable blob. The
 * apple_tb5_acio_layout struct represents the parsed blob.
 */
struct apple_tb5_acio_layout {
	u32	tx_desc_base;
	u32	tx_hop_ctrl_base;
	u32	rx_desc_base;
	u32	rx_hop_ctrl_base;
	u32	ring_desc_stride;
	u32	hop_ctrl_stride;
	u32	max_tx_rings;
	u32	max_rx_rings;
};

/* Default ACIO layout (M4 Pro, t8132) */
#define APPLE_TB5_ACIO_DEFAULTS() {				\
	.tx_desc_base		= 0x0000,			\
	.tx_hop_ctrl_base	= 0x4000,			\
	.rx_desc_base		= 0x8000,			\
	.rx_hop_ctrl_base	= 0xC000,			\
	.ring_desc_stride	= APPLE_TB5_RING_DESC_STRIDE,	\
	.hop_ctrl_stride	= APPLE_TB5_HOP_CTRL_STRIDE,	\
	.max_tx_rings		= 24,				\
	.max_rx_rings		= 24,				\
}

/* ── HAL object vtable offsets (from arm64e analysis) ─────────────── */

/*
 * The AppleThunderboltGenericHAL vtable offsets used by the NHI
 * driver. Documented here for reference; Linux won't use the
 * C++ vtable directly, but these offsets are where the register
 * read/write and buffer allocation virtual methods live.
 *
 * These are useful for anyone building a compatible HAL layer
 * or verifying behavior against the macOS driver.
 */
#define APPLE_TB5_HAL_VT_GET_PHYS_ADDR		0x140
#define APPLE_TB5_HAL_VT_GET_BUF_SIZE		0x148
#define APPLE_TB5_HAL_VT_REG_READ32		0x8A0
#define APPLE_TB5_HAL_VT_REG_WRITE32		0x8A8
#define APPLE_TB5_HAL_VT_INDEX_VALID		0x998
#define APPLE_TB5_HAL_VT_RING_ENABLE		0x208
#define APPLE_TB5_HAL_VT_RING_DISABLE		0x318
#define APPLE_TB5_HAL_VT_ENABLE_INTR		0x218
#define APPLE_TB5_HAL_VT_DISABLE_INTR		0x1C8

/* ── Ring object layout (from arm64e analysis) ───────────────────── */

/*
 * Offsets within the AppleThunderboltNHI*Ring C++ object.
 * Useful for understanding the macOS driver's memory layout.
 * Linux won't use these directly.
 */
#define APPLE_TB5_RING_OBJ_NHI		0x18
#define APPLE_TB5_RING_OBJ_HAL		0x20
#define APPLE_TB5_RING_OBJ_RING_ID	0x2C
#define APPLE_TB5_RING_OBJ_DESC_BASE	0x30
#define APPLE_TB5_RING_OBJ_HOP_CTRL	0x34
#define APPLE_TB5_RING_OBJ_SIZE	0x38
#define APPLE_TB5_RING_OBJ_HOP_ID	0x40
#define APPLE_TB5_RING_OBJ_CONS_IDX	0x48

/* ── DART IOMMU ───────────────────────────────────────────────────── */

/*
 * The Apple NHI uses DART (Device Address Resolution Table) for IOMMU.
 * From ioreg: DART at 0x40024000 with SIDs 0x00-0x0B and 0x10-0x1B.
 * Each SID maps a HopID's DMA address space. The Linux DMA API
 * handles DART translation transparently when dma_dev is the
 * platform device associated with the ACIO node.
 */
#define APPLE_TB5_DART_BASE		0x40024000
#define APPLE_TB5_DART_SID_MIN		0x00
#define APPLE_TB5_DART_SID_MAX		0x0B
#define APPLE_TB5_DART_SID_ALT_MIN	0x10
#define APPLE_TB5_DART_SID_ALT_MAX	0x1B

/* ── ACIO MMIO region ────────────────────────────────────────────── */

/* From ioreg: ACIO0 at 0x40100000, 1MB */
#define APPLE_TB5_ACIO_PHYS_BASE	0x40100000
#define APPLE_TB5_ACIO_SIZE		0x100000

/* Number of MSI-X interrupts (from ioreg: 24) */
#define APPLE_TB5_NUM_MSIX		24

#endif /* APPLE_TB5_NHI_REGS_H */
