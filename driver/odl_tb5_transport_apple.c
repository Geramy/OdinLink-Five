// SPDX-License-Identifier: MIT
/*
 * OdinLink — Apple Silicon Transport Backend
 *
 * Hardware-specific transport for Apple Silicon Thunderbolt (ACIO fabric).
 * Uses the Apple NHI DMA engine determined from analysis of the macOS
 * AppleThunderboltNHI kext (7.2.81, arm64e, macOS 26.5).
 *
 * Register map and descriptor format from arm64e analysis of:
 *   - AppleThunderboltNHIReceiveRing::startDMA
 *   - AppleThunderboltNHIReceiveRing::stopDMA
 *   - AppleThunderboltNHITransmitRing::writeProducerIndexInternal
 *   - AppleThunderboltNHIReceiveRing::writeConsumerIndexInternal
 *   - IOThunderboltTransmitCommand::setDescCache (descriptor layout)
 *
 * Key findings:
 *   - 16-byte descriptors: words[0:1] = 64-bit DMA addr, word[2] = control,
 *     word[3] = reserved
 *   - Per-ring registers: +0x00 DMA addr lo, +0x04 DMA addr hi,
 *     +0x08 producer/consumer index, +0x0C ring_size + HopID packed
 *   - Per-hop control register: bit 31 = enable, bit 30 = coalescing,
 *     bit 29 = interrupt-on-desc, bit 28 = two-page buffer,
 *     bits 22:12 = max frame size, bits 27:16 = HopID
 *   - Doorbell = write32 of producer/consumer index
 *   - DART IOMMU provides per-HopID address spaces
 *
 * This driver will probe when a compatible "apple,thunderbolt-nhi" device
 * appears in the device tree (provided by a future Apple NHI platform driver
 * or when the ACIO device is enabled by the ATCPHY USB4 pipehandler).
 *
 * Until the upstream Apple NHI platform driver exists, this file is
 * compile-test only. The ATCPHY USB4 pipehandler state
 * (ATCPHY_PIPEHANDLER_STATE_USB4) must also be implemented for the
 * physical lanes to carry USB4/TBT traffic.
 */

#include "odl_tb5_core.h"
#include "odl_tb5_xd_proto.h"
#include <linux/platform_device.h>
#include <linux/io.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/of.h>
#include <linux/of_platform.h>

#undef pr_fmt
#define pr_fmt(fmt) "odl_tb5 apple: " fmt

#define APPLE_DBG_REG   0x01
#define APPLE_DBG_RING  0x02
#define APPLE_DBG_DESC  0x04
#define APPLE_DBG_IRQ   0x08
#define APPLE_DBG_PROBE 0x10
#define APPLE_DBG_ALL   0x1F

static unsigned int apple_debug;
module_param_named(apple_debug, apple_debug, uint, 0644);
MODULE_PARM_DESC(apple_debug,
	"Apple transport debug bitmask: 0x01=reg, 0x02=ring, 0x04=desc, "
	"0x08=irq, 0x10=probe (default 0)");

#define apple_dbg(mask, fmt, ...) \
	do { \
		if (apple_debug & (mask)) \
			pr_debug(fmt, ##__VA_ARGS__); \
	} while (0)

#define apple_info(fmt, ...)  pr_info(fmt, ##__VA_ARGS__)
#define apple_warn(fmt, ...)  pr_warn(fmt, ##__VA_ARGS__)
#define apple_err(fmt, ...)   pr_err(fmt, ##__VA_ARGS__)

/* ── Register map (from arm64e analysis) ────────────────────────── */

/*
 * Each ring has a "ring descriptor base" and a "hop control base".
 * These are u32 offsets into the ACIO MMIO space, set during ring alloc.
 * The startDMA function writes 4 registers per ring:
 *   ring_desc_base + 0x00: DMA buffer address (low 32 bits)
 *   ring_desc_base + 0x04: DMA buffer address (high 32 bits)
 *   ring_desc_base + 0x08: producer/consumer index / credit count
 *   ring_desc_base + 0x0C: ring_size (bits 11:0) + HopID (bits 27:16)
 *
 * The hop control register is a separate base:
 *   hop_ctrl_base + 0x00: control word (see APPLE_RING_CTRL_* below)
 */

#define APPLE_RING_DESC_ADDR_LO	0x00
#define APPLE_RING_DESC_ADDR_HI	0x04
#define APPLE_RING_DESC_INDEX		0x08
#define APPLE_RING_DESC_SIZE_HOPID	0x0C
#define APPLE_RING_DESC_STRIDE		0x10

#define APPLE_HOP_CTRL_REG		0x00
#define APPLE_HOP_CTRL_STRIDE		0x10

/* Control word bits (from startDMA bit manipulation) */
#define APPLE_RING_CTRL_ENABLE		BIT(31)
#define APPLE_RING_CTRL_COALESCE	BIT(30)
#define APPLE_RING_CTRL_INT_ON_DESC	BIT(29)
#define APPLE_RING_CTRL_TWO_PAGE	BIT(28)
#define APPLE_RING_CTRL_HOPID_SHIFT	16
#define APPLE_RING_CTRL_HOPID_MASK	GENMASK(27, 16)
#define APPLE_RING_CTRL_FRAME_SZ_SHIFT	12
#define APPLE_RING_CTRL_FRAME_SZ_MASK	GENMASK(22, 12)

/* Index register format:
 *   If isIndexWriteValid (ring has space): index & 0xFFFF
 *   If not valid: index << 16 (overflow indicator)
 */
#define APPLE_INDEX_MASK		0xFFFF
#define APPLE_INDEX_SHIFT		16

/* HopID packed into size register */
#define APPLE_SIZE_HOPID_SHIFT		16
#define APPLE_SIZE_HOPID_MASK		GENMASK(27, 16)
#define APPLE_SIZE_RING_MASK		GENMASK(11, 0)

/* kdebug trace IDs (from symbol table) */
#define APPLE_KDBG_CLASS		0x534
#define APPLE_KDBG_START_DMA		0x5344034
#define APPLE_KDBG_TX_PRODUCER		0x53440B4
#define APPLE_KDBG_RX_CONSUMER		0x5344050

/* Interrupt management (from debug strings) */
#define APPLE_INT_MASK_QUAD_SHIFT	5
#define APPLE_INT_MASK_BIT_MASK		0x1F
#define APPLE_MAX_HOPID			12

/* ── Apple NHI DMA descriptor (16 bytes) ────────────────────────────── */

/*
 * From setDescCache analysis:
 *   bytes 0-7:  64-bit DMA buffer physical address
 *   bytes 8-11: control/length word (SOF, EOF, frame length, flags)
 *   bytes 12-15: reserved (zero)
 *
 * The control word (word[2]) bit fields are not fully known from
 * analysis alone. Based on Intel NHI conventions and
 * the logRings format string ("desc[%d] 0x%08x 0x%08x 0x%08x 0x%08x"),
 * we use the same ring_frame callback mechanism. The Apple hardware
 * fills the descriptor on RX completion; we write it on TX submit.
 */
struct apple_tb_desc {
	__le32	addr_lo;
	__le32	addr_hi;
	__le32	control;
	__le32	reserved;
};

/* Control word bits (inferred from Intel NHI + debug strings) */
#define APPLE_DESC_CTRL_SOF		BIT(0)
#define APPLE_DESC_CTRL_EOF		BIT(1)
#define APPLE_DESC_CTRL_INT_EN		BIT(2)
#define APPLE_DESC_CTRL_LEN_SHIFT	16
#define APPLE_DESC_CTRL_LEN_MASK	GENMASK(31, 16)

/* ── Private data ──────────────────────────────────────────────────── */

struct apple_ring_state {
	void		*desc_ring;	/* DMA coherent descriptor ring */
	dma_addr_t	desc_ring_phys;	/* physical address of desc ring */
	unsigned int	desc_count;	/* number of descriptors */
	unsigned int	prod_idx;	/* TX: producer index */
	unsigned int	cons_idx;	/* RX: consumer index */
	u32		ring_desc_base;	/* register offset for ring descriptors */
	u32		hop_ctrl_base;	/* register offset for hop control */
	int		hop_id;	/* HopID for this ring */
	bool		started;
};

struct apple_priv {
	struct platform_device	*pdev;
	void __iomem		*mmio;		/* ACIO register base (1MB) */
	struct device		*dma_dev;	/* DART-mapped DMA device */
	int			irq;		/* NHI interrupt */
	int			local_tx_hopid;

	struct apple_ring_state	tx;
	struct apple_ring_state	rx;

	/* Spinlock for register access */
	spinlock_t		reg_lock;
};

static inline struct apple_priv *apple_priv(struct odl_tb5_device *dev)
{
	return dev->transport_priv;
}

/* ── MMIO helpers ───────────────────────────────────────────────────── */

static inline void apple_reg_write(struct apple_priv *priv,
				   u32 offset, u32 value)
{
	apple_dbg(APPLE_DBG_REG, "reg_write offset=0x%04x value=0x%08x\n",
		  offset, value);
	writel(value, priv->mmio + offset);
}

static inline u32 apple_reg_read(struct apple_priv *priv, u32 offset)
{
	u32 val = readl(priv->mmio + offset);
	apple_dbg(APPLE_DBG_REG, "reg_read  offset=0x%04x value=0x%08x\n",
		  offset, val);
	return val;
}

/*
 * Write a 64-bit DMA address as two 32-bit register writes.
 * From startDMA: registerWrite32(base, addr64), then
 * registerWrite32(base + 4, addr_high32).
 * The HAL vtable's registerWrite32 takes the offset as first arg
 * and the value as second. The HAL internally adds the ring base.
 */
static void apple_write_dma_addr(struct apple_priv *priv,
				 u32 base, dma_addr_t addr)
{
	apple_dbg(APPLE_DBG_REG, "write_dma_addr base=0x%04x addr=0x%016llx\n",
		  base, (unsigned long long)addr);
	apple_reg_write(priv, base + APPLE_RING_DESC_ADDR_LO,
			lower_32_bits(addr));
	apple_reg_write(priv, base + APPLE_RING_DESC_ADDR_HI,
			upper_32_bits(addr));
}

/* ── Ring alloc/free ───────────────────────────────────────────────── */

static int apple_ring_alloc(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned int rs = odl_ring_size;
	size_t desc_ring_bytes;
	int ret;

	if (rs < ODL_TB5_RING_SIZE_MIN)
		rs = ODL_TB5_RING_SIZE_MIN;
	if (rs > ODL_TB5_RING_SIZE_MAX)
		rs = ODL_TB5_RING_SIZE_MAX;
	rs = roundup_pow_of_two(rs);

	dev->tx.ring_size = rs;
	dev->rx.ring_size = rs;

	apple_info("ring_size=%u (%u MB per batch, %u MB total)\n",
		   rs,
		   (rs * ODL_TB5_FRAME_SIZE) >> 20,
		   (rs * ODL_TB5_FRAME_SIZE * ODL_TB5_NUM_BUFFERS * 2) >> 20);

	dev->tx.frames = kvzalloc(rs * sizeof(struct ring_frame), GFP_KERNEL);
	if (!dev->tx.frames)
		return -ENOMEM;

	dev->rx.frames = kvzalloc(rs * sizeof(struct ring_frame), GFP_KERNEL);
	if (!dev->rx.frames) {
		ret = -ENOMEM;
		goto err_free_tx_frames;
	}

	desc_ring_bytes = rs * sizeof(struct apple_tb_desc);
	apple_dbg(APPLE_DBG_RING, "allocating %zu bytes for TX desc ring "
		  "(%u descriptors x %zu bytes each)\n",
		  desc_ring_bytes, rs, sizeof(struct apple_tb_desc));

	priv->tx.desc_ring = dma_alloc_coherent(priv->dma_dev,
						 desc_ring_bytes,
						 &priv->tx.desc_ring_phys,
						 GFP_KERNEL);
	if (!priv->tx.desc_ring) {
		apple_err("failed to allocate TX descriptor ring "
			  "(%zu bytes, dma_dev=%p)\n",
			  desc_ring_bytes, priv->dma_dev);
		ret = -ENOMEM;
		goto err_free_rx_frames;
	}
	priv->tx.desc_count = rs;
	priv->tx.prod_idx = 0;
	apple_dbg(APPLE_DBG_RING, "TX desc ring: virt=%p phys=0x%016llx\n",
		  priv->tx.desc_ring,
		  (unsigned long long)priv->tx.desc_ring_phys);

	priv->rx.desc_ring = dma_alloc_coherent(priv->dma_dev,
						 desc_ring_bytes,
						 &priv->rx.desc_ring_phys,
						 GFP_KERNEL);
	if (!priv->rx.desc_ring) {
		apple_err("failed to allocate RX descriptor ring "
			  "(%zu bytes, dma_dev=%p)\n",
			  desc_ring_bytes, priv->dma_dev);
		ret = -ENOMEM;
		goto err_free_tx_desc;
	}
	priv->rx.desc_count = rs;
	priv->rx.cons_idx = 0;
	apple_dbg(APPLE_DBG_RING, "RX desc ring: virt=%p phys=0x%016llx\n",
		  priv->rx.desc_ring,
		  (unsigned long long)priv->rx.desc_ring_phys);

	/*
	 * Register base offsets for each ring.
	 * The Apple HAL uses ring_desc_base and hop_ctrl_base stored
	 * in the ring object (offsets 0x30 and 0x34). For the ACIO
	 * hardware, these are computed from the HopID and ring type.
	 *
	 * The exact stride between consecutive ring register sets is
	 * 0x10 (16 bytes = 4 x u32 registers), matching the 4 register
	 * offsets 0x00..0x0C. The hop control uses the same stride.
	 *
	 * Base offsets are determined by the firmware tunable blobs
	 * applied to the ACIO fabric. For now we use fixed offsets
	 * that will be overridden by tunable application.
	 *
	 * TX rings start at a different base than RX rings (Apple's
	 * HAL separates TX and RX fabric paths).
	 */
	priv->tx.ring_desc_base = 0x0000;
	priv->tx.hop_ctrl_base = 0x4000;
	priv->rx.ring_desc_base = 0x8000;
	priv->rx.hop_ctrl_base = 0xC000;

	priv->tx.hop_id = 0;
	priv->rx.hop_id = 0;
	priv->local_tx_hopid = 0;

	apple_info("rings allocated: tx_desc_base=0x%x tx_hop_ctrl=0x%x "
		   "rx_desc_base=0x%x rx_hop_ctrl=0x%x\n",
		   priv->tx.ring_desc_base, priv->tx.hop_ctrl_base,
		   priv->rx.ring_desc_base, priv->rx.hop_ctrl_base);

	return 0;

err_free_tx_desc:
	dma_free_coherent(priv->dma_dev,
			  priv->tx.desc_count * sizeof(struct apple_tb_desc),
			  priv->tx.desc_ring, priv->tx.desc_ring_phys);
	priv->tx.desc_ring = NULL;
err_free_rx_frames:
	kvfree(dev->rx.frames);
	dev->rx.frames = NULL;
err_free_tx_frames:
	kvfree(dev->tx.frames);
	dev->tx.frames = NULL;
	return ret;
}

static void apple_ring_free(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);

	apple_dbg(APPLE_DBG_RING, "freeing rings\n");

	if (priv->rx.desc_ring) {
		apple_dbg(APPLE_DBG_RING, "freeing RX desc ring: "
			  "virt=%p phys=0x%016llx count=%u\n",
			  priv->rx.desc_ring,
			  (unsigned long long)priv->rx.desc_ring_phys,
			  priv->rx.desc_count);
		dma_free_coherent(priv->dma_dev,
				  priv->rx.desc_count * sizeof(struct apple_tb_desc),
				  priv->rx.desc_ring, priv->rx.desc_ring_phys);
		priv->rx.desc_ring = NULL;
	}

	if (priv->tx.desc_ring) {
		apple_dbg(APPLE_DBG_RING, "freeing TX desc ring: "
			  "virt=%p phys=0x%016llx count=%u\n",
			  priv->tx.desc_ring,
			  (unsigned long long)priv->tx.desc_ring_phys,
			  priv->tx.desc_count);
		dma_free_coherent(priv->dma_dev,
				  priv->tx.desc_count * sizeof(struct apple_tb_desc),
				  priv->tx.desc_ring, priv->tx.desc_ring_phys);
		priv->tx.desc_ring = NULL;
	}

	kvfree(dev->tx.frames);
	dev->tx.frames = NULL;
	kvfree(dev->rx.frames);
	dev->rx.frames = NULL;
}

/* ── Ring start/stop/reset ──────────────────────────────────────────── */

/*
 * startDMA (from ReceiveRing::startDMA analysis):
 *
 * 1. Get DMA buffer physical address (64-bit)
 * 2. Write buffer address to ring_desc_base + 0x00 (lo) and +0x04 (hi)
 * 3. Write buffer address high 32 bits to ring_desc_base + 0x04
 * 4. Write ring size + HopID packed to ring_desc_base + 0x0C
 * 5. Write credit count to ring_desc_base + 0x08
 * 6. Build control word: ENABLE | COALESCE? | INT_ON_DESC? | TWO_PAGE? | HopID
 * 7. Write control word to hop_ctrl_base + 0x00
 * 8. Enable interrupts
 */
static int apple_ring_start(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned long flags;
	u32 size_hopid;
	u32 ctrl;

	apple_info("starting rings (tx_hop=%d, rx_hop=%d, "
		   "tx_desc_base=0x%x, rx_desc_base=0x%x)\n",
		   priv->tx.hop_id, priv->rx.hop_id,
		   priv->tx.ring_desc_base, priv->rx.ring_desc_base);

	spin_lock_irqsave(&priv->reg_lock, flags);

	/* TX ring setup */
	apple_dbg(APPLE_DBG_RING, "TX: writing DMA addr 0x%016llx "
		  "to desc_base 0x%04x\n",
		  (unsigned long long)priv->tx.desc_ring_phys,
		  priv->tx.ring_desc_base);
	apple_write_dma_addr(priv, priv->tx.ring_desc_base,
			     priv->tx.desc_ring_phys);

	size_hopid = (dev->tx.ring_size & APPLE_SIZE_RING_MASK) |
		     (priv->tx.hop_id << APPLE_SIZE_HOPID_SHIFT);
	apple_dbg(APPLE_DBG_RING, "TX: size_hopid=0x%08x "
		  "(ring_size=%u, hop_id=%d)\n",
		  size_hopid, dev->tx.ring_size, priv->tx.hop_id);
	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_RING_DESC_SIZE_HOPID,
			size_hopid);

	apple_dbg(APPLE_DBG_RING, "TX: initial prod_idx=%u\n",
		  priv->tx.prod_idx);
	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_RING_DESC_INDEX,
			priv->tx.prod_idx & APPLE_INDEX_MASK);

	ctrl = APPLE_RING_CTRL_ENABLE |
	       (priv->tx.hop_id << APPLE_RING_CTRL_HOPID_SHIFT);
	apple_dbg(APPLE_DBG_RING, "TX: control word=0x%08x "
		  "(ENABLE|HOPID=%d)\n", ctrl, priv->tx.hop_id);
	apple_reg_write(priv, priv->tx.hop_ctrl_base, ctrl);
	priv->tx.started = true;

	/* RX ring setup */
	apple_dbg(APPLE_DBG_RING, "RX: writing DMA addr 0x%016llx "
		  "to desc_base 0x%04x\n",
		  (unsigned long long)priv->rx.desc_ring_phys,
		  priv->rx.ring_desc_base);
	apple_write_dma_addr(priv, priv->rx.ring_desc_base,
			     priv->rx.desc_ring_phys);

	size_hopid = (dev->rx.ring_size & APPLE_SIZE_RING_MASK) |
		     (priv->rx.hop_id << APPLE_SIZE_HOPID_SHIFT);
	apple_dbg(APPLE_DBG_RING, "RX: size_hopid=0x%08x "
		  "(ring_size=%u, hop_id=%d)\n",
		  size_hopid, dev->rx.ring_size, priv->rx.hop_id);
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_RING_DESC_SIZE_HOPID,
			size_hopid);

	apple_dbg(APPLE_DBG_RING, "RX: initial cons_idx=%u\n",
		  priv->rx.cons_idx);
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_RING_DESC_INDEX,
			priv->rx.cons_idx & APPLE_INDEX_MASK);

	ctrl = APPLE_RING_CTRL_ENABLE |
	       APPLE_RING_CTRL_INT_ON_DESC |
	       (priv->rx.hop_id << APPLE_RING_CTRL_HOPID_SHIFT);
	apple_dbg(APPLE_DBG_RING, "RX: control word=0x%08x "
		  "(ENABLE|INT_ON_DESC|HOPID=%d)\n", ctrl, priv->rx.hop_id);
	apple_reg_write(priv, priv->rx.hop_ctrl_base, ctrl);
	priv->rx.started = true;

	spin_unlock_irqrestore(&priv->reg_lock, flags);

	apple_info("rings started OK\n");

	return 0;
}

/*
 * stopDMA (from ReceiveRing::stopDMA analysis):
 *
 * 1. Check if ring is enabled
 * 2. Disable interrupt
 * 3. Read hop_ctrl_base register, clear bit 31 (ENABLE), write back
 * 4. Call ringDisable() vtable
 */
static void apple_ring_stop(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned long flags;
	u32 ctrl;

	apple_info("stopping rings (tx_started=%d, rx_started=%d)\n",
		   priv->tx.started, priv->rx.started);

	spin_lock_irqsave(&priv->reg_lock, flags);

	if (priv->tx.started) {
		ctrl = apple_reg_read(priv, priv->tx.hop_ctrl_base);
		apple_dbg(APPLE_DBG_RING, "TX: ctrl before stop=0x%08x, "
			  "clearing ENABLE\n", ctrl);
		apple_reg_write(priv, priv->tx.hop_ctrl_base,
				ctrl & ~APPLE_RING_CTRL_ENABLE);
		priv->tx.started = false;
	}

	if (priv->rx.started) {
		ctrl = apple_reg_read(priv, priv->rx.hop_ctrl_base);
		apple_dbg(APPLE_DBG_RING, "RX: ctrl before stop=0x%08x, "
			  "clearing ENABLE\n", ctrl);
		apple_reg_write(priv, priv->rx.hop_ctrl_base,
				ctrl & ~APPLE_RING_CTRL_ENABLE);
		priv->rx.started = false;
	}

	spin_unlock_irqrestore(&priv->reg_lock, flags);
	apple_info("rings stopped\n");
}

static void apple_ring_reset(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);

	apple_info("resetting rings (tx_prod=%u, rx_cons=%u)\n",
		   priv->tx.prod_idx, priv->rx.cons_idx);

	apple_ring_stop(dev);

	priv->tx.prod_idx = 0;
	priv->rx.cons_idx = 0;

	if (priv->tx.desc_ring)
		memset(priv->tx.desc_ring, 0,
		       priv->tx.desc_count * sizeof(struct apple_tb_desc));
	if (priv->rx.desc_ring)
		memset(priv->rx.desc_ring, 0,
		       priv->rx.desc_count * sizeof(struct apple_tb_desc));

	apple_dbg(APPLE_DBG_RING, "descriptor rings zeroed, restarting\n");
	apple_ring_start(dev);
}

/* ── Frame submit ──────────────────────────────────────────────────── */

/*
 * TX submit: write a 16-byte descriptor to the ring, then ring the
 * doorbell by writing the producer index.
 *
 * From writeNextDescriptor analysis:
 *   1. Build descriptor via buildTxBufferDescriptor()
 *   2. Copy 16 bytes to ring: ldr q0, [cmd]; str q0, [ring + (index * 16)]
 *   3. Call writeProducerIndexForCommand()
 *
 * From setDescCache analysis, the descriptor is:
 *   words[0:1] = 64-bit DMA buffer address
 *   words[2] = control word (SOF/EOF/length)
 *   words[3] = reserved (zero)
 */
static int apple_ring_tx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	struct apple_priv *priv = apple_priv(dev);
	struct apple_tb_desc *desc;
	unsigned int idx;
	dma_addr_t buf_phys;
	u32 control = 0;
	unsigned long flags;

	if (!priv->tx.started) {
		apple_warn("TX submit while ring not started\n");
		return -EIO;
	}

	idx = priv->tx.prod_idx % priv->tx.desc_count;
	desc = (struct apple_tb_desc *)priv->tx.desc_ring + idx;

	buf_phys = frame->buffer_phy;

	control = (frame->size << APPLE_DESC_CTRL_LEN_SHIFT) &
		  APPLE_DESC_CTRL_LEN_MASK;
	if (frame->sof)
		control |= APPLE_DESC_CTRL_SOF;
	if (frame->eof)
		control |= APPLE_DESC_CTRL_EOF;
	control |= APPLE_DESC_CTRL_INT_EN;

	apple_dbg(APPLE_DBG_DESC, "TX desc[%u]: addr=0x%016llx "
		  "ctrl=0x%08x (len=%u sof=%d eof=%d) prod_idx=%u\n",
		  idx, (unsigned long long)buf_phys, control,
		  frame->size, frame->sof, frame->eof,
		  priv->tx.prod_idx);

	spin_lock_irqsave(&priv->reg_lock, flags);

	desc->addr_lo = cpu_to_le32(lower_32_bits(buf_phys));
	desc->addr_hi = cpu_to_le32(upper_32_bits(buf_phys));
	desc->control = cpu_to_le32(control);
	desc->reserved = 0;

	priv->tx.prod_idx++;
	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_RING_DESC_INDEX,
			priv->tx.prod_idx & APPLE_INDEX_MASK);

	spin_unlock_irqrestore(&priv->reg_lock, flags);

	return 0;
}

static int apple_ring_rx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	struct apple_priv *priv = apple_priv(dev);
	struct apple_tb_desc *desc;
	unsigned int idx;
	unsigned long flags;

	if (!priv->rx.started) {
		apple_warn("RX submit while ring not started\n");
		return -EIO;
	}

	idx = priv->rx.cons_idx % priv->rx.desc_count;
	desc = (struct apple_tb_desc *)priv->rx.desc_ring + idx;

	apple_dbg(APPLE_DBG_DESC, "RX desc[%u]: addr=0x%016llx "
		  "cons_idx=%u\n",
		  idx, (unsigned long long)frame->buffer_phy,
		  priv->rx.cons_idx);

	spin_lock_irqsave(&priv->reg_lock, flags);

	desc->addr_lo = cpu_to_le32(lower_32_bits(frame->buffer_phy));
	desc->addr_hi = cpu_to_le32(upper_32_bits(frame->buffer_phy));
	desc->control = cpu_to_le32(APPLE_DESC_CTRL_INT_EN |
				    (ODL_TB5_FRAME_SIZE <<
				     APPLE_DESC_CTRL_LEN_SHIFT));
	desc->reserved = 0;

	priv->rx.cons_idx++;
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_RING_DESC_INDEX,
			priv->rx.cons_idx & APPLE_INDEX_MASK);

	spin_unlock_irqrestore(&priv->reg_lock, flags);

	return 0;
}

/* ── DMA device access ─────────────────────────────────────────────── */

/*
 * Return the DART-mapped device for coherent DMA allocation.
 * Apple Silicon uses per-HopID DART mappers (AppleThunderboltNHIDARTVMAllocator),
 * but the Linux DMA API handles DART translation transparently when
 * dma_dev is the platform device associated with the ACIO node.
 */
static struct device *apple_dma_device(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);

	if (priv->dma_dev)
		return priv->dma_dev;
	if (priv->pdev)
		return &priv->pdev->dev;
	return NULL;
}

static struct odl_tb5_transport_ring_info apple_tx_ring_info(
	struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);

	return (struct odl_tb5_transport_ring_info){
		.hop = priv->tx.hop_id
	};
}

static struct odl_tb5_transport_ring_info apple_rx_ring_info(
	struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);

	return (struct odl_tb5_transport_ring_info){
		.hop = priv->rx.hop_id
	};
}

static int apple_local_tx_hopid(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);

	return priv->local_tx_hopid;
}

/* ── Path management ───────────────────────────────────────────────── */

/*
 * Path enable: configure the ACIO fabric for the HopID path.
 * On Apple Silicon, the fabric path is set up by firmware tunables
 * applied during boot (the "hi_up_tx_desc_fabric_tunables" etc. blobs
 * from the device tree). The NHI driver just enables the rings.
 *
 * Once the ATCPHY USB4 pipehandler state is implemented upstream,
 * this function will also need to request USB4 mode from the PHY.
 */
static int apple_path_enable(struct odl_tb5_device *dev)
{
	/*
	 * No additional path setup needed for Apple — the ACIO fabric
	 * paths are configured by firmware tunables at boot. The ring
	 * start already writes the HopID to the control register.
	 *
	 * Future: request ATCPHY USB4 pipehandler state change via
	 * phy_set_mode(priv->phy, PHY_MODE_USB4) or equivalent.
	 */
	return 0;
}

static void apple_path_disable(struct odl_tb5_device *dev)
{
	/*
	 * Disable the ring (clear ENABLE bit). The ACIO fabric path
	 * teardown happens automatically when the ring is disabled.
	 */
	apple_ring_stop(dev);
}

/* ── Login/Logout ──────────────────────────────────────────────────── */

/*
 * The XDomain login/logout protocol is the same for Apple and Intel.
 * The difference is that on Apple, we don't have a tb_xdomain object
 * from the Thunderbolt bus. Instead, we send the login packet
 * directly through the DMA ring.
 *
 * For now, return -ENODEV until the Apple NHI platform driver
 * provides the XDomain route and UUID information.
 */
static int apple_peer_send_login(struct odl_tb5_device *dev)
{
	apple_warn("peer_send_login: not implemented (no XDomain on Apple yet)\n");
	return -ENODEV;
}

static int apple_peer_send_logout(struct odl_tb5_device *dev)
{
	apple_warn("peer_send_logout: not implemented (no XDomain on Apple yet)\n");
	return -ENODEV;
}

/* ── Kick (for hrtimer poll) ───────────────────────────────────────── */

/*
 * Kick = ring the doorbell by writing the producer/consumer index.
 * On Apple, the doorbell IS the index write to the register.
 * The hrtimer poll is needed for the same reason as Intel: descriptor
 * write-back can lag behind the MSI-X interrupt.
 */
static void apple_kick_tx(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned long flags;

	if (!priv->tx.started)
		return;

	spin_lock_irqsave(&priv->reg_lock, flags);
	apple_dbg(APPLE_DBG_IRQ, "kick_tx: writing prod_idx=%u to "
		  "reg 0x%04x\n",
		  priv->tx.prod_idx & APPLE_INDEX_MASK,
		  priv->tx.ring_desc_base + APPLE_RING_DESC_INDEX);
	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_RING_DESC_INDEX,
			priv->tx.prod_idx & APPLE_INDEX_MASK);
	spin_unlock_irqrestore(&priv->reg_lock, flags);
}

static void apple_kick_rx(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned long flags;

	if (!priv->rx.started)
		return;

	spin_lock_irqsave(&priv->reg_lock, flags);
	apple_dbg(APPLE_DBG_IRQ, "kick_rx: writing cons_idx=%u to "
		  "reg 0x%04x\n",
		  priv->rx.cons_idx & APPLE_INDEX_MASK,
		  priv->rx.ring_desc_base + APPLE_RING_DESC_INDEX);
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_RING_DESC_INDEX,
			priv->rx.cons_idx & APPLE_INDEX_MASK);
	spin_unlock_irqrestore(&priv->reg_lock, flags);
}

/* ── Interrupt handler ─────────────────────────────────────────────── */

static irqreturn_t apple_nhi_irq(int irq, void *data)
{
	struct odl_tb5_device *dev = data;
	struct apple_priv *priv = apple_priv(dev);
	u32 status;

	if (!priv->mmio)
		return IRQ_NONE;

	status = apple_reg_read(priv, 0);

	if (!status)
		return IRQ_NONE;

	apple_dbg(APPLE_DBG_IRQ, "IRQ %d fired, status=0x%08x\n",
		  irq, status);

	if (dev->transport->kick_tx)
		dev->transport->kick_tx(dev);
	if (dev->transport->kick_rx)
		dev->transport->kick_rx(dev);

	return IRQ_HANDLED;
}

/* ── Ops table ─────────────────────────────────────────────────────── */

const struct odl_tb5_transport_ops odl_tb5_apple_transport = {
	.type		= ODL_TB5_TRANSPORT_APPLE,
	.name		= "apple",
	.ring_alloc	= apple_ring_alloc,
	.ring_free	= apple_ring_free,
	.ring_start	= apple_ring_start,
	.ring_stop	= apple_ring_stop,
	.ring_reset	= apple_ring_reset,
	.ring_tx	= apple_ring_tx,
	.ring_rx	= apple_ring_rx,
	.dma_device	= apple_dma_device,
	.tx_ring_info	= apple_tx_ring_info,
	.rx_ring_info	= apple_rx_ring_info,
	.local_tx_hopid = apple_local_tx_hopid,
	.path_enable	= apple_path_enable,
	.path_disable	= apple_path_disable,
	.peer_send_login  = apple_peer_send_login,
	.peer_send_logout = apple_peer_send_logout,
	.kick_tx	= apple_kick_tx,
	.kick_rx	= apple_kick_rx,
};

/* ── Platform driver (for device tree probe) ────────────────────────── */

/*
 * The Apple NHI will be exposed as a platform device under the ACIO
 * node in the device tree. This driver probes when:
 *   1. The ATCPHY has set up USB4 lane mode
 *   2. The ACIO fabric is powered and clocked
 *   3. The firmware tunables have been applied
 *   4. A "apple,thunderbolt-nhi" compatible node exists
 *
 * Until the upstream Apple NHI platform driver creates this device,
 * this probe won't fire. The transport is also usable without the
 * platform driver if manually instantiated (for development).
 */
static int apple_nhi_probe(struct platform_device *pdev)
{
	struct device_node *np = pdev->dev.of_node;
	struct odl_tb5_device *dev;
	struct apple_priv *priv;
	struct resource *res;
	int ret;

	apple_info("probing platform device %s\n",
		   dev_name(&pdev->dev));

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return -ENOMEM;

	priv = kzalloc(sizeof(*priv), GFP_KERNEL);
	if (!priv) {
		kfree(dev);
		return -ENOMEM;
	}

	priv->pdev = pdev;
	priv->dma_dev = &pdev->dev;
	spin_lock_init(&priv->reg_lock);

	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res) {
		apple_err("no MMIO resource\n");
		ret = -ENODEV;
		goto err_free_priv;
	}
	apple_dbg(APPLE_DBG_PROBE, "MMIO resource: start=0x%llx end=0x%llx "
		  "size=0x%llx\n",
		  (unsigned long long)res->start,
		  (unsigned long long)res->end,
		  (unsigned long long)resource_size(res));

	priv->mmio = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(priv->mmio)) {
		apple_err("MMIO remap failed (err=%ld)\n",
			  PTR_ERR(priv->mmio));
		ret = PTR_ERR(priv->mmio);
		goto err_free_priv;
	}
	apple_dbg(APPLE_DBG_PROBE, "MMIO mapped at %p\n", priv->mmio);

	priv->irq = platform_get_irq(pdev, 0);
	apple_dbg(APPLE_DBG_PROBE, "IRQ = %d\n", priv->irq);
	if (priv->irq > 0) {
		ret = devm_request_irq(&pdev->dev, priv->irq,
				       apple_nhi_irq, 0,
				       "odl_tb5_apple", dev);
		if (ret) {
			apple_warn("IRQ request failed (%d), "
				   "falling back to poll\n", ret);
			priv->irq = -1;
		} else {
			apple_dbg(APPLE_DBG_PROBE, "IRQ %d registered\n",
				  priv->irq);
		}
	}

	if (np) {
		of_property_read_u32(np, "apple,hopid",
				     &priv->local_tx_hopid);
		of_property_read_u32(np, "apple,tx-desc-base",
				     &priv->tx.ring_desc_base);
		of_property_read_u32(np, "apple,tx-hop-ctrl-base",
				     &priv->tx.hop_ctrl_base);
		of_property_read_u32(np, "apple,rx-desc-base",
				     &priv->rx.ring_desc_base);
		of_property_read_u32(np, "apple,rx-hop-ctrl-base",
				     &priv->rx.hop_ctrl_base);
		of_property_read_u32(np, "apple,tx-hopid",
				     &priv->tx.hop_id);
		of_property_read_u32(np, "apple,rx-hopid",
				     &priv->rx.hop_id);
		apple_dbg(APPLE_DBG_PROBE, "DT props: hopid=%d "
			  "tx_desc_base=0x%x tx_hop_ctrl=0x%x "
			  "rx_desc_base=0x%x rx_hop_ctrl=0x%x "
			  "tx_hopid=%d rx_hopid=%d\n",
			  priv->local_tx_hopid,
			  priv->tx.ring_desc_base, priv->tx.hop_ctrl_base,
			  priv->rx.ring_desc_base, priv->rx.hop_ctrl_base,
			  priv->tx.hop_id, priv->rx.hop_id);
	} else {
		apple_dbg(APPLE_DBG_PROBE, "no OF node, using defaults\n");
	}

	dev->transport = &odl_tb5_apple_transport;
	dev->transport_priv = priv;
	dev->state = ODL_TB5_STATE_DISCONNECTED;
	mutex_init(&dev->state_lock);
	init_waitqueue_head(&dev->state_waitq);
	spin_lock_init(&dev->tx.lock);
	spin_lock_init(&dev->rx.lock);
	init_waitqueue_head(&dev->tx.waitq);
	init_waitqueue_head(&dev->rx.waitq);
	atomic_set(&dev->tx.completed, 0);
	atomic_set(&dev->tx.submitted, 0);
	atomic_set(&dev->rx.completed, 0);
	atomic_set(&dev->rx.submitted, 0);
	atomic_set(&dev->open_count, 0);
	atomic_set(&dev->removing, 0);

	hash_init(dev->streams);
	ida_init(&dev->stream_ida);
	mutex_init(&dev->stream_lock);
	INIT_WORK(&dev->tx_drain_work, odl_tb5_tx_drain_work_fn);
	atomic_set(&dev->rx_posted, 0);
	dev->rx_target = 0;

	dev->tx_adaptive.mode = ODL_TB5_TX_LATENCY;
	dev->tx_adaptive.consecutive_low = 0;
	dev->tx_adaptive.high_watermark = odl_ring_size * 3 / 4;
	dev->tx_adaptive.low_watermark  = odl_ring_size / 4;

	ret = ida_alloc_max(&odl_tb5_ida, ODL_TB5_MAX_DEVICES - 1,
			    GFP_KERNEL);
	if (ret < 0)
		goto err_free_priv;
	dev->index = ret;

	ret = odl_tb5_chardev_create(dev);
	if (ret)
		goto err_ida;

	ret = odl_tb5_rings_alloc(dev);
	if (ret)
		goto err_chardev;

	ret = odl_tb5_dma_bufs_alloc(dev);
	if (ret)
		goto err_rings;

	mutex_lock(&odl_tb5_devices_lock);
	list_add_tail(&dev->list, &odl_tb5_devices_list);
	mutex_unlock(&odl_tb5_devices_lock);

	platform_set_drvdata(pdev, dev);

	apple_info("probed (hopid=%d, mmio=%p, irq=%d)\n",
		   priv->local_tx_hopid, priv->mmio, priv->irq);

	return 0;

err_rings:
	apple_err("ring alloc failed during probe, cleaning up\n");
	odl_tb5_rings_free(dev);
err_chardev:
	odl_tb5_chardev_destroy(dev);
err_ida:
	ida_free(&odl_tb5_ida, dev->index);
err_free_priv:
	kfree(priv);
	kfree(dev);
	return ret;
}

static int apple_nhi_remove(struct platform_device *pdev)
{
	struct odl_tb5_device *dev = platform_get_drvdata(pdev);
	struct apple_priv *priv;

	if (!dev)
		return 0;

	priv = apple_priv(dev);
	apple_info("removing device index %d\n", dev->index);
	atomic_set(&dev->removing, 1);

	hrtimer_cancel(&dev->rx_poll_timer);
	cancel_work_sync(&dev->tx_drain_work);

	odl_tb5_rings_stop(dev);
	synchronize_rcu();

	mutex_lock(&odl_tb5_devices_lock);
	list_del_rcu(&dev->list);
	mutex_unlock(&odl_tb5_devices_lock);

	odl_tb5_streams_destroy_all(dev);
	ida_destroy(&dev->stream_ida);
	odl_tb5_frame_pool_free(dev);
	odl_tb5_batch_pool_free(dev);
	odl_tb5_dma_bufs_free(dev);
	odl_tb5_rings_free(dev);
	odl_tb5_chardev_destroy(dev);

	apple_info("removed device index %d\n", dev->index);

	kfree(priv);
	ida_free(&odl_tb5_ida, dev->index);
	kfree(dev);
	return 0;
}

static const struct of_device_id apple_nhi_of_match[] = {
	{ .compatible = "apple,thunderbolt-nhi" },
	{ }
};
MODULE_DEVICE_TABLE(of, apple_nhi_of_match);

struct platform_driver odl_tb5_apple_driver = {
	.probe	= apple_nhi_probe,
	.remove	= apple_nhi_remove,
	.driver	= {
		.name		= "odl_tb5_apple",
		.of_match_table	= apple_nhi_of_match,
	},
};

/* ── Module init/exit for Apple transport ───────────────────────────── */

int odl_tb5_apple_init(void)
{
	apple_info("registering platform driver\n");
	return platform_driver_register(&odl_tb5_apple_driver);
}

void odl_tb5_apple_exit(void)
{
	apple_info("unregistering platform driver\n");
	platform_driver_unregister(&odl_tb5_apple_driver);
}
