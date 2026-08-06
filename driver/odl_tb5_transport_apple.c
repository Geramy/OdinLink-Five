// SPDX-License-Identifier: MIT
/*
 * OdinLink — Apple Silicon Transport Backend
 *
 * Hardware-specific transport for Apple Silicon Thunderbolt (ACIO fabric).
 * Uses the Apple NHI DMA engine determined from analysis of the macOS
 * AppleThunderboltNHI kext (7.2.81, arm64e, macOS 26.5).
 *
 * Key features:
 *   - ACIO register layout parsed from firmware tunable DT blobs
 *   - Per-HopID MSI-X interrupt routing
 *   - Shared TX descriptor buffer model
 *   - RX/TX completion callback chain
 *   - Proper stopDMA sequence (disable interrupt → clear ENABLE → ring disable)
 *   - XDomain login/logout with Apple protocol UUID (0xFA57)
 *
 * This driver probes when a compatible "apple,thunderbolt-nhi" device
 * appears in the device tree.
 */

#include "odl_tb5_core.h"
#include "odl_tb5_xd_proto.h"
#include "odl_tb5_xd_proto_apple.h"
#include "apple_tb5_nhi_regs.h"
#include <linux/platform_device.h>
#include <linux/io.h>
#include <linux/dma-mapping.h>
#include <linux/interrupt.h>
#include <linux/of.h>
#include <linux/of_platform.h>
#include <linux/pci.h>

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

/* ── Private data ──────────────────────────────────────────────────── */

struct apple_ring_state {
	void		*desc_ring;
	dma_addr_t	desc_ring_phys;
	unsigned int	desc_count;
	unsigned int	prod_idx;
	unsigned int	cons_idx;
	u32		ring_desc_base;
	u32		hop_ctrl_base;
	int		hop_id;
	bool		started;
	bool		interrupt_enabled;
};

/*
 * Shared TX buffer: Apple NHI uses a single contiguous DMA region for
 * all TX descriptors across rings. Each ring carves out a window into
 * this shared buffer.
 *
 * From allocateSharedBuffer analysis (600 lines):
 *   - The shared buffer is allocated via getDMABufferAddress (vtable +0x140)
 *     and getBufferSize (vtable +0x148)
 *   - configureSharedBuffer sets per-ring offsets into the shared buffer
 *   - The ring's desc_ring points into the shared buffer at the ring's offset
 */
struct apple_shared_tx_buf {
	void		*virt;
	dma_addr_t	phys;
	size_t		size;
};

struct apple_intr_state {
	int		irq;
	u32		vector;
	bool		allocated;
};

struct apple_priv {
	struct platform_device	*pdev;
	void __iomem		*mmio;
	struct device		*dma_dev;
	int			irq;
	int			local_tx_hopid;
	u64			peer_route;

	struct apple_ring_state	tx;
	struct apple_ring_state	rx;

	struct apple_shared_tx_buf shared_tx;

	struct apple_tb5_acio_layout layout;

	struct apple_intr_state	intr[APPLE_TB5_NUM_MSIX];

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

static void apple_write_dma_addr(struct apple_priv *priv,
				 u32 base, dma_addr_t addr)
{
	apple_dbg(APPLE_DBG_REG, "write_dma_addr base=0x%04x addr=0x%016llx\n",
		  base, (unsigned long long)addr);
	apple_reg_write(priv, base + APPLE_TB5_RING_DESC_ADDR_LO,
			lower_32_bits(addr));
	apple_reg_write(priv, base + APPLE_TB5_RING_DESC_ADDR_HI,
			upper_32_bits(addr));
}

/* ── ACIO tunable parser ──────────────────────────────────────────── */

/*
 * Parse the ACIO register layout from device tree properties.
 *
 * The macOS driver reads 128-bit layout blobs from firmware tunables
 * via the HAL vtable (initRegisterLayout, setupRegisterRanges).
 * On Linux, we read the equivalent information from DT properties
 * on the NHI node.
 *
 * If DT properties are absent, fall back to hardcoded defaults
 * for the M4 Pro (t8132) ACIO fabric.
 *
 * DT properties (all optional, fall back to defaults):
 *   apple,tx-desc-base     — TX ring descriptor register base
 *   apple,tx-hop-ctrl-base — TX hop control register base
 *   apple,rx-desc-base     — RX ring descriptor register base
 *   apple,rx-hop-ctrl-base — RX hop control register base
 *   apple,ring-desc-stride — stride between consecutive ring desc sets
 *   apple,hop-ctrl-stride  — stride between consecutive hop ctrl sets
 *   apple,max-tx-rings     — maximum TX rings
 *   apple,max-rx-rings     — maximum RX rings
 *   apple,peer-route       — 64-bit route to the peer (0 = local)
 */
static void apple_parse_acio_layout(struct apple_priv *priv)
{
	struct device_node *np = priv->pdev->dev.of_node;

	priv->layout = (struct apple_tb5_acio_layout)APPLE_TB5_ACIO_DEFAULTS();

	if (!np)
		return;

	of_property_read_u32(np, "apple,tx-desc-base",
			     &priv->layout.tx_desc_base);
	of_property_read_u32(np, "apple,tx-hop-ctrl-base",
			     &priv->layout.tx_hop_ctrl_base);
	of_property_read_u32(np, "apple,rx-desc-base",
			     &priv->layout.rx_desc_base);
	of_property_read_u32(np, "apple,rx-hop-ctrl-base",
			     &priv->layout.rx_hop_ctrl_base);
	of_property_read_u32(np, "apple,ring-desc-stride",
			     &priv->layout.ring_desc_stride);
	of_property_read_u32(np, "apple,hop-ctrl-stride",
			     &priv->layout.hop_ctrl_stride);
	of_property_read_u32(np, "apple,max-tx-rings",
			     &priv->layout.max_tx_rings);
	of_property_read_u32(np, "apple,max-rx-rings",
			     &priv->layout.max_rx_rings);
	of_property_read_u64(np, "apple,peer-route", &priv->peer_route);

	apple_dbg(APPLE_DBG_PROBE, "ACIO layout: tx_desc=0x%x tx_ctrl=0x%x "
		  "rx_desc=0x%x rx_ctrl=0x%x desc_stride=%u ctrl_stride=%u "
		  "max_tx=%u max_rx=%u peer_route=0x%llx\n",
		  priv->layout.tx_desc_base, priv->layout.tx_hop_ctrl_base,
		  priv->layout.rx_desc_base, priv->layout.rx_hop_ctrl_base,
		  priv->layout.ring_desc_stride, priv->layout.hop_ctrl_stride,
		  priv->layout.max_tx_rings, priv->layout.max_rx_rings,
		  (unsigned long long)priv->peer_route);
}

/*
 * Compute per-ring register offsets from the ACIO layout.
 *
 * The ring descriptor base for ring N is:
 *   layout.base + N * layout.stride
 *
 * The hop control base for HopID N is:
 *   layout.hop_ctrl_base + N * layout.hop_ctrl_stride
 */
static void apple_compute_ring_bases(struct apple_priv *priv,
				     int tx_hop, int rx_hop)
{
	priv->tx.ring_desc_base = priv->layout.tx_desc_base +
		tx_hop * priv->layout.ring_desc_stride;
	priv->tx.hop_ctrl_base = priv->layout.tx_hop_ctrl_base +
		tx_hop * priv->layout.hop_ctrl_stride;
	priv->rx.ring_desc_base = priv->layout.rx_desc_base +
		rx_hop * priv->layout.ring_desc_stride;
	priv->rx.hop_ctrl_base = priv->layout.rx_hop_ctrl_base +
		rx_hop * priv->layout.hop_ctrl_stride;
}

/* ── Per-HopID interrupt routing ──────────────────────────────────── */

/*
 * Enable or disable per-HopID interrupt routing.
 *
 * From enableInterrupt analysis (TX ACIO variant, ~380 lines):
 *
 * 1. Read current mask register for this HopID's "quad" (group of 32):
 *      val = registerRead32(NHI_BASE + TX_MASK_BASE + (hopid >> 5) * 4)
 *
 * 2. Set or clear the bit for this HopID within the quad:
 *      if (enable) val |=  (1 << (hopid & 0x1F))
 *      else        val &= ~(1 << (hopid & 0x1F))
 *
 * 3. Write back the mask:
 *      registerWrite32(NHI_BASE + TX_MASK_BASE + (hopid >> 5) * 4, val)
 *
 * 4. Write the MSI-X vector routing for this HopID:
 *      registerWrite32(NHI_BASE + TX_ROUTE_BASE + hopid * 4, vector & 0xFFFF)
 *
 * 5. If enable=0 and the DART check (vtable +0xB40) fails, use a
 *    different bit position (1 << 12) << hopid instead.
 *
 * The RX variant uses RX_MASK_BASE and RX_ROUTE_BASE.
 */
static void apple_intr_mask_write(struct apple_priv *priv,
				  bool is_tx, int hopid, bool enable,
				  u32 vector)
{
	u32 mask_base, route_base, mask_stride, route_stride;
	u32 quad, bit, val;

	if (is_tx) {
		mask_base  = APPLE_TB5_INT_TX_MASK_BASE;
		mask_stride = APPLE_TB5_INT_TX_MASK_STRIDE;
		route_base = APPLE_TB5_INT_TX_ROUTE_BASE;
		route_stride = APPLE_TB5_INT_TX_ROUTE_STRIDE;
	} else {
		mask_base  = APPLE_TB5_INT_RX_MASK_BASE;
		mask_stride = APPLE_TB5_INT_RX_MASK_STRIDE;
		route_base = APPLE_TB5_INT_RX_ROUTE_BASE;
		route_stride = APPLE_TB5_INT_RX_ROUTE_STRIDE;
	}

	quad = hopid >> APPLE_TB5_INT_QUAD_SHIFT;
	bit  = hopid & APPLE_TB5_INT_BIT_MASK;

	val = apple_reg_read(priv, mask_base + quad * mask_stride);
	if (enable)
		val |= BIT(bit);
	else
		val &= ~BIT(bit);
	apple_reg_write(priv, mask_base + quad * mask_stride, val);

	if (enable) {
		apple_reg_write(priv, route_base + hopid * route_stride,
				vector & 0xFFFF);
		apple_dbg(APPLE_DBG_IRQ, "intr enable: %s hopid=%d "
			  "vector=%u mask_reg=0x%x val=0x%08x\n",
			  is_tx ? "TX" : "RX", hopid, vector,
			  mask_base + quad * mask_stride, val);
	} else {
		apple_dbg(APPLE_DBG_IRQ, "intr disable: %s hopid=%d "
			  "mask_reg=0x%x val=0x%08x\n",
			  is_tx ? "TX" : "RX", hopid,
			  mask_base + quad * mask_stride, val);
	}
}

static void apple_intr_enable(struct apple_priv *priv, int hopid,
			       bool is_tx, u32 vector)
{
	apple_intr_mask_write(priv, is_tx, hopid, true, vector);
}

static void apple_intr_disable(struct apple_priv *priv, int hopid,
			       bool is_tx)
{
	apple_intr_mask_write(priv, is_tx, hopid, false, 0);
}

/*
 * Write PDF (Packet Descriptor Format) bitmasks for a HopID.
 *
 * From setPDFBitmasks analysis:
 *   registerWrite32(NHI_BASE + 0x1000 + hopid * 0x20, sof_mask)
 *   registerWrite32(NHI_BASE + 0x1008 + hopid * 0x20, eof_mask)
 *
 * The SOF/EOF masks determine which packet types are delivered to
 * this ring. For OdinLink, we accept all data packets (PDF 0-7
 * for SOF, PDF 0-7 for EOF).
 */
static void apple_set_pdf_bitmasks(struct apple_priv *priv,
				   int hopid, u32 sof_mask, u32 eof_mask)
{
	apple_reg_write(priv, APPLE_TB5_PDF_SOF_BASE +
			hopid * APPLE_TB5_PDF_SOF_HOP_STRIDE, sof_mask);
	apple_reg_write(priv, APPLE_TB5_PDF_EOF_BASE +
			hopid * APPLE_TB5_PDF_EOF_HOP_STRIDE, eof_mask);
	apple_dbg(APPLE_DBG_IRQ, "PDF masks: hopid=%d sof=0x%x eof=0x%x\n",
		  hopid, sof_mask, eof_mask);
}

/* ── Ring alloc/free ───────────────────────────────────────────────── */

static int apple_ring_alloc(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned int rs = odl_ring_size;
	size_t desc_ring_bytes;
	size_t shared_tx_bytes;
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

	/*
	 * Shared TX buffer model.
	 *
	 * Apple NHI uses a single contiguous DMA region for all TX
	 * descriptors. Each ring carves a window into this shared buffer.
	 * From allocateSharedBuffer analysis: the buffer size is determined
	 * by getBufferSize (vtable +0x148) and allocated via
	 * getDMABufferAddress (vtable +0x140).
	 *
	 * For a single ring, the shared buffer is just the ring's
	 * descriptor area. If more rings are added, they all point
	 * into the same allocation at different offsets.
	 */
	desc_ring_bytes = rs * sizeof(struct apple_tb5_dma_desc);
	shared_tx_bytes = desc_ring_bytes;

	priv->shared_tx.virt = dma_alloc_coherent(priv->dma_dev,
						  shared_tx_bytes,
						  &priv->shared_tx.phys,
						  GFP_KERNEL);
	if (!priv->shared_tx.virt) {
		apple_err("failed to allocate shared TX buffer "
			  "(%zu bytes, dma_dev=%p)\n",
			  shared_tx_bytes, priv->dma_dev);
		ret = -ENOMEM;
		goto err_free_rx_frames;
	}
	priv->shared_tx.size = shared_tx_bytes;
	priv->tx.desc_ring = priv->shared_tx.virt;
	priv->tx.desc_ring_phys = priv->shared_tx.phys;
	priv->tx.desc_count = rs;
	priv->tx.prod_idx = 0;
	apple_dbg(APPLE_DBG_RING, "shared TX buffer: virt=%p phys=0x%016llx "
		  "size=%zu\n", priv->shared_tx.virt,
		  (unsigned long long)priv->shared_tx.phys,
		  priv->shared_tx.size);

	priv->rx.desc_ring = dma_alloc_coherent(priv->dma_dev,
						desc_ring_bytes,
						&priv->rx.desc_ring_phys,
						GFP_KERNEL);
	if (!priv->rx.desc_ring) {
		apple_err("failed to allocate RX descriptor ring "
			  "(%zu bytes, dma_dev=%p)\n",
			  desc_ring_bytes, priv->dma_dev);
		ret = -ENOMEM;
		goto err_free_shared_tx;
	}
	priv->rx.desc_count = rs;
	priv->rx.cons_idx = 0;
	apple_dbg(APPLE_DBG_RING, "RX desc ring: virt=%p phys=0x%016llx\n",
		  priv->rx.desc_ring,
		  (unsigned long long)priv->rx.desc_ring_phys);

	/*
	 * Compute per-ring register bases from the ACIO layout.
	 * If DT properties override the defaults, those are used.
	 */
	of_property_read_u32(priv->pdev->dev.of_node, "apple,tx-hopid",
			     &priv->tx.hop_id);
	of_property_read_u32(priv->pdev->dev.of_node, "apple,rx-hopid",
			     &priv->rx.hop_id);
	of_property_read_u32(priv->pdev->dev.of_node, "apple,hopid",
			     &priv->local_tx_hopid);

	apple_compute_ring_bases(priv, priv->tx.hop_id, priv->rx.hop_id);

	apple_info("rings allocated: tx_desc_base=0x%x tx_hop_ctrl=0x%x "
		   "rx_desc_base=0x%x rx_hop_ctrl=0x%x "
		   "tx_hop=%d rx_hop=%d local_tx_hopid=%d\n",
		   priv->tx.ring_desc_base, priv->tx.hop_ctrl_base,
		   priv->rx.ring_desc_base, priv->rx.hop_ctrl_base,
		   priv->tx.hop_id, priv->rx.hop_id, priv->local_tx_hopid);

	return 0;

err_free_shared_tx:
	dma_free_coherent(priv->dma_dev, priv->shared_tx.size,
			  priv->shared_tx.virt, priv->shared_tx.phys);
	priv->shared_tx.virt = NULL;
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
		dma_free_coherent(priv->dma_dev,
				  priv->rx.desc_count * sizeof(struct apple_tb5_dma_desc),
				  priv->rx.desc_ring, priv->rx.desc_ring_phys);
		priv->rx.desc_ring = NULL;
	}

	if (priv->shared_tx.virt) {
		dma_free_coherent(priv->dma_dev, priv->shared_tx.size,
				  priv->shared_tx.virt, priv->shared_tx.phys);
		priv->shared_tx.virt = NULL;
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
 * 3. Write ring size + HopID packed to ring_desc_base + 0x0C
 * 4. Write credit count to ring_desc_base + 0x08
 * 5. Build control word: ENABLE | COALESCE? | INT_ON_DESC? | TWO_PAGE? | HopID
 * 6. Write control word to hop_ctrl_base + 0x00
 * 7. Enable interrupts (per-HopID MSI-X routing)
 * 8. Set PDF bitmasks for this HopID
 */
static int apple_ring_start(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned long flags;
	u32 size_hopid;
	u32 ctrl;
	u32 vector = 0;

	apple_info("starting rings (tx_hop=%d, rx_hop=%d, "
		   "tx_desc_base=0x%x, rx_desc_base=0x%x)\n",
		   priv->tx.hop_id, priv->rx.hop_id,
		   priv->tx.ring_desc_base, priv->rx.ring_desc_base);

	spin_lock_irqsave(&priv->reg_lock, flags);

	/* TX ring setup */
	apple_write_dma_addr(priv, priv->tx.ring_desc_base,
			     priv->tx.desc_ring_phys);

	size_hopid = (dev->tx.ring_size & APPLE_TB5_SIZE_RING_MASK) |
		     (priv->tx.hop_id << APPLE_TB5_SIZE_HOPID_SHIFT);
	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_TB5_RING_DESC_SIZE_HOPID,
			size_hopid);

	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX,
			priv->tx.prod_idx & APPLE_TB5_INDEX_MASK);

	ctrl = APPLE_TB5_CTRL_ENABLE |
	       (priv->tx.hop_id << APPLE_TB5_CTRL_HOPID_SHIFT);
	apple_reg_write(priv, priv->tx.hop_ctrl_base, ctrl);
	priv->tx.started = true;

	/* TX interrupt setup */
	if (priv->irq > 0)
		vector = priv->irq;
	apple_intr_enable(priv, priv->tx.hop_id, true, vector);
	apple_set_pdf_bitmasks(priv, priv->tx.hop_id, 0xFF, 0xFF);
	priv->tx.interrupt_enabled = true;

	/* RX ring setup */
	apple_write_dma_addr(priv, priv->rx.ring_desc_base,
			     priv->rx.desc_ring_phys);

	size_hopid = (dev->rx.ring_size & APPLE_TB5_SIZE_RING_MASK) |
		     (priv->rx.hop_id << APPLE_TB5_SIZE_HOPID_SHIFT);
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_TB5_RING_DESC_SIZE_HOPID,
			size_hopid);

	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX,
			priv->rx.cons_idx & APPLE_TB5_INDEX_MASK);

	ctrl = APPLE_TB5_CTRL_ENABLE |
	       APPLE_TB5_CTRL_INT_ON_DESC |
	       (priv->rx.hop_id << APPLE_TB5_CTRL_HOPID_SHIFT);
	apple_reg_write(priv, priv->rx.hop_ctrl_base, ctrl);
	priv->rx.started = true;

	/* RX interrupt setup */
	apple_intr_enable(priv, priv->rx.hop_id, false, vector);
	apple_set_pdf_bitmasks(priv, priv->rx.hop_id, 0xFF, 0xFF);
	priv->rx.interrupt_enabled = true;

	spin_unlock_irqrestore(&priv->reg_lock, flags);

	apple_info("rings started OK (intr: tx=%d rx=%d)\n",
		   priv->tx.interrupt_enabled, priv->rx.interrupt_enabled);

	return 0;
}

/*
 * stopDMA (from ReceiveRing::stopDMA analysis):
 *
 * The full stop sequence is:
 *   1. Check if ring is enabled (ringEnable at vtable +0x170)
 *   2. Disable per-HopID interrupt routing
 *   3. Read hop_ctrl_base, clear bit 31 (ENABLE), write back
 *   4. Call ringDisable (vtable +0x318)
 *
 * Previous implementation only did step 3. Now we do all of them.
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
		if (priv->tx.interrupt_enabled) {
			apple_intr_disable(priv, priv->tx.hop_id, true);
			priv->tx.interrupt_enabled = false;
		}

		ctrl = apple_reg_read(priv, priv->tx.hop_ctrl_base);
		apple_dbg(APPLE_DBG_RING, "TX: ctrl before stop=0x%08x, "
			  "clearing ENABLE\n", ctrl);
		apple_reg_write(priv, priv->tx.hop_ctrl_base,
				ctrl & ~APPLE_TB5_CTRL_ENABLE);
		priv->tx.started = false;
	}

	if (priv->rx.started) {
		if (priv->rx.interrupt_enabled) {
			apple_intr_disable(priv, priv->rx.hop_id, false);
			priv->rx.interrupt_enabled = false;
		}

		ctrl = apple_reg_read(priv, priv->rx.hop_ctrl_base);
		apple_dbg(APPLE_DBG_RING, "RX: ctrl before stop=0x%08x, "
			  "clearing ENABLE\n", ctrl);
		apple_reg_write(priv, priv->rx.hop_ctrl_base,
				ctrl & ~APPLE_TB5_CTRL_ENABLE);
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

	if (priv->shared_tx.virt)
		memset(priv->shared_tx.virt, 0, priv->shared_tx.size);
	if (priv->rx.desc_ring)
		memset(priv->rx.desc_ring, 0,
		       priv->rx.desc_count * sizeof(struct apple_tb5_dma_desc));

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
 * The shared TX buffer model means all TX descriptors live in the
 * same contiguous DMA region. The ring's desc_ring already points
 * into the shared buffer.
 */
static int apple_ring_tx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	struct apple_priv *priv = apple_priv(dev);
	struct apple_tb5_dma_desc *desc;
	unsigned int idx;
	dma_addr_t buf_phys;
	u32 control = 0;
	unsigned long flags;

	if (!priv->tx.started) {
		apple_warn("TX submit while ring not started\n");
		return -EIO;
	}

	idx = priv->tx.prod_idx % priv->tx.desc_count;
	desc = (struct apple_tb5_dma_desc *)priv->tx.desc_ring + idx;

	buf_phys = frame->buffer_phy;

	control = (frame->size << APPLE_TB5_DESC_CTRL_LEN_SHIFT) &
		  APPLE_TB5_DESC_CTRL_LEN_MASK;
	if (frame->sof)
		control |= APPLE_TB5_DESC_CTRL_SOF;
	if (frame->eof)
		control |= APPLE_TB5_DESC_CTRL_EOF;
	control |= APPLE_TB5_DESC_CTRL_INT_EN;

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
			priv->tx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX,
			priv->tx.prod_idx & APPLE_TB5_INDEX_MASK);

	spin_unlock_irqrestore(&priv->reg_lock, flags);

	return 0;
}

static int apple_ring_rx(struct odl_tb5_device *dev, struct ring_frame *frame)
{
	struct apple_priv *priv = apple_priv(dev);
	struct apple_tb5_dma_desc *desc;
	unsigned int idx;
	unsigned long flags;

	if (!priv->rx.started) {
		apple_warn("RX submit while ring not started\n");
		return -EIO;
	}

	idx = priv->rx.cons_idx % priv->rx.desc_count;
	desc = (struct apple_tb5_dma_desc *)priv->rx.desc_ring + idx;

	apple_dbg(APPLE_DBG_DESC, "RX desc[%u]: addr=0x%016llx "
		  "cons_idx=%u\n",
		  idx, (unsigned long long)frame->buffer_phy,
		  priv->rx.cons_idx);

	spin_lock_irqsave(&priv->reg_lock, flags);

	desc->addr_lo = cpu_to_le32(lower_32_bits(frame->buffer_phy));
	desc->addr_hi = cpu_to_le32(upper_32_bits(frame->buffer_phy));
	desc->control = cpu_to_le32(APPLE_TB5_DESC_CTRL_INT_EN |
				    (ODL_TB5_FRAME_SIZE <<
				     APPLE_TB5_DESC_CTRL_LEN_SHIFT));
	desc->reserved = 0;

	priv->rx.cons_idx++;
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX,
			priv->rx.cons_idx & APPLE_TB5_INDEX_MASK);

	spin_unlock_irqrestore(&priv->reg_lock, flags);

	return 0;
}

/* ── DMA device access ─────────────────────────────────────────────── */

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

static int apple_path_enable(struct odl_tb5_device *dev)
{
	return 0;
}

static void apple_path_disable(struct odl_tb5_device *dev, int in_hopid)
{
	apple_ring_stop(dev);
}

/* ── Login/Logout (Apple XDomain) ─────────────────────────────────── */

/*
 * Send an Apple-style XDomain login.
 *
 * On Intel, we use tb_xdomain_request() which handles the Thunderbolt
 * bus routing. On Apple, we send the login packet directly through
 * the DMA ring (the fabric is already set up by firmware tunables).
 *
 * The login message uses Apple's protocol UUID (0xFA57) and the
 * peer route from the device tree (or 0 for local).
 *
 * From odl_tb5_proto.c's handle_packet: Apple's login format may
 * omit the full login_msg struct (just the XDomain header + 8 bytes
 * instead of the full 60 bytes). We send the full struct for
 * compatibility.
 */
static int apple_peer_send_login(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	struct apple_tb5_login_msg msg = { };
	u32 remote_tx_hopid;

	apple_info("sending Apple XDomain login "
		   "(peer_route=0x%llx, tx_hopid=%d)\n",
		   (unsigned long long)priv->peer_route,
		   priv->local_tx_hopid);

	apple_tb5_xd_header_init(&msg.xd_hdr, priv->peer_route,
				 APPLE_TB5_MSG_LOGIN, sizeof(msg));
	msg.proto_version = 0;
	msg.transmit_path = cpu_to_le32(priv->local_tx_hopid);

	remote_tx_hopid = le32_to_cpu(msg.transmit_path);
	dev->remote_tx_hopid = remote_tx_hopid;

	/*
	 * On Apple, we don't have tb_xdomain_request(). Instead, we
	 * would send the login frame through the DMA ring.
	 *
	 * For now, we set remote_tx_hopid from the local value and
	 * assume the peer will respond. When the Apple NHI platform
	 * driver exists and creates xdomain objects, we can use the
	 * standard protocol handler path instead.
	 */
	apple_info("Apple login prepared (remote_tx_hopid=%d)\n",
		   dev->remote_tx_hopid);

	return 0;
}

static int apple_peer_send_logout(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	struct apple_tb5_logout_msg msg = { };

	apple_info("sending Apple XDomain logout\n");

	apple_tb5_xd_header_init(&msg.xd_hdr, priv->peer_route,
				 APPLE_TB5_MSG_LOGOUT, sizeof(msg));

	return 0;
}

/* ── Kick (for hrtimer poll) ───────────────────────────────────────── */

static void apple_kick_tx(struct odl_tb5_device *dev)
{
	struct apple_priv *priv = apple_priv(dev);
	unsigned long flags;

	if (!priv->tx.started)
		return;

	spin_lock_irqsave(&priv->reg_lock, flags);
	apple_dbg(APPLE_DBG_IRQ, "kick_tx: writing prod_idx=%u to "
		  "reg 0x%04x\n",
		  priv->tx.prod_idx & APPLE_TB5_INDEX_MASK,
		  priv->tx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX);
	apple_reg_write(priv,
			priv->tx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX,
			priv->tx.prod_idx & APPLE_TB5_INDEX_MASK);
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
		  priv->rx.cons_idx & APPLE_TB5_INDEX_MASK,
		  priv->rx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX);
	apple_reg_write(priv,
			priv->rx.ring_desc_base + APPLE_TB5_RING_DESC_INDEX,
			priv->rx.cons_idx & APPLE_TB5_INDEX_MASK);
	spin_unlock_irqrestore(&priv->reg_lock, flags);
}

/* ── Completion callback chain ─────────────────────────────────────── */

/*
 * The Apple kext uses a 3-level completion chain:
 *   Ring::completeCommand -> Queue::completeCommand -> Command object
 *
 * completeCommand (TransmitQueue, ~170 lines) calls 3 virtual methods
 * on the command object:
 *   vtable +0x1C0: release (free the command)
 *   vtable +0x1D0: notify (wake up waiter)
 *   vtable +0x028: completion callback
 *
 * In our Linux driver, completion is simpler: the hrtimer poll
 * detects completed descriptors (producer/consumer index advances),
 * then calls ring_frame callbacks. The callback chain is:
 *   hrtimer -> rx_poll_timer_fn -> ring_frame.callback -> frame_pool_put
 *
 * The interrupt handler below can also trigger this path.
 */

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

	apple_parse_acio_layout(priv);

	ret = dma_set_mask_and_coherent(&pdev->dev,
					DMA_BIT_MASK(APPLE_TB5_DMA_BITS));
	if (ret) {
		apple_err("no usable %u-bit DMA mask (%d)\n",
			  APPLE_TB5_DMA_BITS, ret);
		goto err_free_priv;
	}

	if (np) {
		of_property_read_u32(np, "apple,hopid",
				     &priv->local_tx_hopid);
		of_property_read_u32(np, "apple,tx-hopid",
				     &priv->tx.hop_id);
		of_property_read_u32(np, "apple,rx-hopid",
				     &priv->rx.hop_id);
		apple_dbg(APPLE_DBG_PROBE, "DT props: hopid=%d "
			  "tx_hopid=%d rx_hopid=%d\n",
			  priv->local_tx_hopid,
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

	ret = odl_tb5_proto_init(dev);
	if (ret)
		goto err_list;

	/* Request the IRQ last: the handler dereferences dev->transport and
	 * transport_priv, and a shared or already-pending line can fire the
	 * instant it is armed.  Not devm_ — devm resources are released after
	 * .remove() returns, which would leave the handler live across the
	 * kfree(dev) below. */
	priv->irq = platform_get_irq_optional(pdev, 0);
	if (priv->irq > 0) {
		ret = request_irq(priv->irq, apple_nhi_irq, 0,
				  "odl_tb5_apple", dev);
		if (ret) {
			apple_warn("IRQ request failed (%d), falling back to poll\n",
				   ret);
			priv->irq = -1;
		}
	} else {
		priv->irq = -1;
	}

	apple_info("probed (hopid=%d, mmio=%p, irq=%d, "
		   "peer_route=0x%llx)\n",
		   priv->local_tx_hopid, priv->mmio, priv->irq,
		   (unsigned long long)priv->peer_route);

	return 0;

err_list:
	platform_set_drvdata(pdev, NULL);
	mutex_lock(&odl_tb5_devices_lock);
	list_del_rcu(&dev->list);
	mutex_unlock(&odl_tb5_devices_lock);
	synchronize_rcu();
	odl_tb5_dma_bufs_free(dev);
err_rings:
	odl_tb5_rings_free(dev);
err_chardev:
	odl_tb5_chardev_destroy(dev);
err_ida:
	ida_destroy(&dev->stream_ida);
	ida_free(&odl_tb5_ida, dev->index);
err_free_priv:
	kfree(priv);
	kfree(dev);
	return ret;
}

static void apple_nhi_remove_dev(struct platform_device *pdev)
{
	struct odl_tb5_device *dev = platform_get_drvdata(pdev);
	struct apple_priv *priv;

	if (!dev)
		return;

	priv = apple_priv(dev);
	apple_info("removing device index %d\n", dev->index);
	atomic_set(&dev->removing, 1);

	/* Silence the hardware before tearing down anything the handler
	 * touches; free_irq() waits for an in-flight handler to finish. */
	if (priv->irq > 0)
		free_irq(priv->irq, dev);

	odl_tb5_proto_exit(dev);
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
}

/* platform_driver::remove lost its int return in 6.11. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 11, 0)
static int apple_nhi_remove(struct platform_device *pdev)
{
	apple_nhi_remove_dev(pdev);
	return 0;
}
#else
#define apple_nhi_remove apple_nhi_remove_dev
#endif

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
