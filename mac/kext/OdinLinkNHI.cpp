/*
 * OdinLink — Apple NHI RX ring + XDomain watch (Mac kext)
 *
 * Maps the ACIO NHI, posts 4 KB RX descriptors into the shared buffer,
 * and polls the consumer index. Register writes are gated: nothing is
 * armed until userspace calls kOdinLinkArmHardware. The map itself is
 * inferred from AppleThunderboltNHI and has not been proven on silicon.
 */

#include <IOKit/IOLib.h>
#include <IOKit/IODeviceMemory.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <libkern/c++/OSSymbol.h>
#include <libkern/libkern.h>

#include "OdinLinkRDMA.h"
#include "apple_tb5_nhi_mac.h"

void OdinLinkRDMA::writeNHI(uint32_t offset, uint32_t value)
{
	if (!fNHIRegs || offset + 4 > fNHISize)
		return;
	fNHIRegs[offset / 4] = value;
	OSSynchronizeIO();
}

uint32_t OdinLinkRDMA::readNHI(uint32_t offset)
{
	if (!fNHIRegs || offset + 4 > fNHISize)
		return 0;
	OSSynchronizeIO();
	return fNHIRegs[offset / 4];
}

bool OdinLinkRDMA::mapNHI(IOService *provider)
{
	IODeviceMemory *mem;

	fNHIMap = NULL;
	fNHIRegs = NULL;
	fNHISize = 0;

	mem = provider->getDeviceMemoryWithIndex(0);
	if (!mem) {
		IOLog("OdinLinkRDMA: provider has no MMIO range 0 — "
		      "NHI rings unavailable\n");
		return false;
	}

	fNHIMap = mem->map();
	if (!fNHIMap) {
		IOLog("OdinLinkRDMA: failed to map NHI MMIO\n");
		return false;
	}

	fNHIRegs = (volatile uint32_t *)fNHIMap->getVirtualAddress();
	fNHISize = (uint32_t)fNHIMap->getLength();
	if (!fNHIRegs || fNHISize < 0x10000) {
		IOLog("OdinLinkRDMA: NHI map too small (%u)\n", fNHISize);
		unmapNHI();
		return false;
	}

	IOLog("OdinLinkRDMA: NHI MMIO mapped at %p, %u bytes\n",
	      fNHIRegs, fNHISize);
	return true;
}

void OdinLinkRDMA::unmapNHI(void)
{
	if (fNHIMap) {
		fNHIMap->release();
		fNHIMap = NULL;
	}
	fNHIRegs = NULL;
	fNHISize = 0;
}

IOReturn OdinLinkRDMA::startRxRing(void)
{
	IODMACommand::Segment64 seg;
	UInt64 offset;
	UInt32 numSeg;
	IOReturn ret;
	unsigned int i;
	struct apple_tb5_dma_desc *desc;
	uint32_t size_hopid;
	uint32_t ctrl;
	uint32_t desc_base;
	uint32_t hop_ctrl;

	if (fArmed)
		return kIOReturnSuccess;
	if (!fNHIRegs || !fBufferPhysAddr)
		return kIOReturnNotReady;

	fDescMemory = IOBufferMemoryDescriptor::withOptions(
		kIODirectionInOut | kIOMemoryPhysicallyContiguous,
		ODL_MAC_RX_SLOTS * sizeof(struct apple_tb5_dma_desc),
		16);
	if (!fDescMemory)
		return kIOReturnNoMemory;

	fDescDMA = IODMACommand::withSpecification(
		kIODMACommandOutputHost64, APPLE_TB5_DMA_BITS, 0,
		IODMACommand::kMapped, 0, 1);
	if (!fDescDMA) {
		fDescMemory->release();
		fDescMemory = NULL;
		return kIOReturnNoMemory;
	}

	ret = fDescDMA->setMemoryDescriptor(fDescMemory);
	if (ret != kIOReturnSuccess)
		goto fail_desc;

	offset = 0;
	numSeg = 1;
	ret = fDescDMA->gen64IOVMSegments(&offset, &seg, &numSeg);
	if (ret != kIOReturnSuccess || numSeg != 1)
		goto fail_desc;
	fDescPhys = seg.fIOVMAddr;
	fDescVirt = fDescMemory->getBytesNoCopy();
	if (!fDescVirt)
		goto fail_desc;

	desc = (struct apple_tb5_dma_desc *)fDescVirt;
	memset(desc, 0, ODL_MAC_RX_SLOTS * sizeof(*desc));
	for (i = 0; i < ODL_MAC_RX_SLOTS; i++) {
		uint64_t slot = fBufferPhysAddr +
				(uint64_t)i * ODL_MAC_SLOT_BYTES;
		uint32_t control;

		desc[i].addr_lo = (uint32_t)slot;
		desc[i].addr_hi = (uint32_t)(slot >> 32);
		control = (ODL_MAC_SLOT_BYTES << APPLE_TB5_DESC_CTRL_LEN_SHIFT) &
			  APPLE_TB5_DESC_CTRL_LEN_MASK;
		control |= APPLE_TB5_DESC_CTRL_SOF | APPLE_TB5_DESC_CTRL_EOF |
			   APPLE_TB5_DESC_CTRL_INT_EN;
		desc[i].control = control;
	}
	OSSynchronizeIO();

	desc_base = APPLE_TB5_ACIO_RX_DESC_BASE +
		    (uint32_t)fRxHop * APPLE_TB5_RING_DESC_STRIDE;
	hop_ctrl = APPLE_TB5_ACIO_RX_HOP_CTRL +
		   (uint32_t)fRxHop * APPLE_TB5_HOP_CTRL_STRIDE;

	writeNHI(desc_base + APPLE_TB5_RING_DESC_ADDR_LO,
		 (uint32_t)fDescPhys);
	writeNHI(desc_base + APPLE_TB5_RING_DESC_ADDR_HI,
		 (uint32_t)(fDescPhys >> 32));
	size_hopid = (ODL_MAC_RX_SLOTS & APPLE_TB5_SIZE_RING_MASK) |
		     ((uint32_t)fRxHop << APPLE_TB5_SIZE_HOPID_SHIFT);
	writeNHI(desc_base + APPLE_TB5_RING_DESC_SIZE_HOPID, size_hopid);
	writeNHI(desc_base + APPLE_TB5_RING_DESC_INDEX, 0);

	writeNHI(APPLE_TB5_PDF_SOF_BASE +
		 (uint32_t)fRxHop * APPLE_TB5_PDF_SOF_HOP_STRIDE, 0xFFu);
	writeNHI(APPLE_TB5_PDF_EOF_BASE +
		 (uint32_t)fRxHop * APPLE_TB5_PDF_EOF_HOP_STRIDE, 0xFFu);

	ctrl = APPLE_TB5_CTRL_ENABLE | APPLE_TB5_CTRL_INT_ON_DESC |
	       ((uint32_t)fRxHop << APPLE_TB5_CTRL_HOPID_SHIFT);
	writeNHI(hop_ctrl, ctrl);

	fLastIdx = 0;
	fArmed = true;
	IOLog("OdinLinkRDMA: RX ring ARMED hop=%d desc_phys=0x%llx "
	      "slots=%u (unverified register map)\n",
	      fRxHop, (unsigned long long)fDescPhys, ODL_MAC_RX_SLOTS);
	return kIOReturnSuccess;

fail_desc:
	if (fDescDMA) {
		fDescDMA->clearMemoryDescriptor();
		fDescDMA->release();
		fDescDMA = NULL;
	}
	if (fDescMemory) {
		fDescMemory->release();
		fDescMemory = NULL;
	}
	return kIOReturnNoMemory;
}

void OdinLinkRDMA::stopRxRing(void)
{
	uint32_t hop_ctrl;

	if (!fArmed)
		return;

	hop_ctrl = APPLE_TB5_ACIO_RX_HOP_CTRL +
		   (uint32_t)fRxHop * APPLE_TB5_HOP_CTRL_STRIDE;
	if (fNHIRegs) {
		uint32_t ctrl = readNHI(hop_ctrl);
		writeNHI(hop_ctrl, ctrl & ~APPLE_TB5_CTRL_ENABLE);
	}

	if (fDescDMA) {
		fDescDMA->clearMemoryDescriptor();
		fDescDMA->release();
		fDescDMA = NULL;
	}
	if (fDescMemory) {
		fDescMemory->release();
		fDescMemory = NULL;
	}
	fDescVirt = NULL;
	fDescPhys = 0;
	fArmed = false;
	IOLog("OdinLinkRDMA: RX ring disarmed\n");
}

void OdinLinkRDMA::pollRx(void)
{
	uint32_t desc_base;
	uint32_t idx;
	uint32_t delta;

	if (!fArmed || !fNHIRegs)
		return;

	desc_base = APPLE_TB5_ACIO_RX_DESC_BASE +
		    (uint32_t)fRxHop * APPLE_TB5_RING_DESC_STRIDE;
	idx = readNHI(desc_base + APPLE_TB5_RING_DESC_INDEX) &
	      APPLE_TB5_INDEX_MASK;

	delta = (idx - fLastIdx) & APPLE_TB5_INDEX_MASK;
	if (delta == 0 || delta > ODL_MAC_RX_SLOTS)
		return;

	if (fLock)
		IOSimpleLockLock(fLock);
	fRxDone += delta;
	fFrameCount = fRxDone;
	fLastIdx = idx;
	fBufferReady = true;
	if (fLock)
		IOSimpleLockUnlock(fLock);
}

void OdinLinkRDMA::timerFired(OSObject *owner, IOTimerEventSource *sender)
{
	OdinLinkRDMA *self = OSDynamicCast(OdinLinkRDMA, owner);

	if (!self)
		return;
	self->pollRx();
	if (sender)
		sender->setTimeoutMS(1);
}

void OdinLinkRDMA::logXDomain(IOService *svc)
{
	OSObject *obj;
	OSNumber *num;
	OSString *str;

	if (!svc)
		return;

	IOLog("OdinLinkRDMA: XDomain service %s\n", svc->getName());

	obj = svc->getProperty("Protocol ID");
	num = OSDynamicCast(OSNumber, obj);
	if (num)
		IOLog("OdinLinkRDMA:   Protocol ID = %u (0x%x)\n",
		      num->unsigned32BitValue(),
		      num->unsigned32BitValue());

	str = OSDynamicCast(OSString, svc->getProperty("Protocol Key"));
	if (str)
		IOLog("OdinLinkRDMA:   Protocol Key = %s\n",
		      str->getCStringNoCopy());

	str = OSDynamicCast(OSString, svc->getProperty("Route String"));
	if (str)
		IOLog("OdinLinkRDMA:   Route = %s\n",
		      str->getCStringNoCopy());
}

bool OdinLinkRDMA::xdomainAppeared(void *target, void *refCon,
				   IOService *newService, IONotifier *notifier)
{
	OdinLinkRDMA *self = (OdinLinkRDMA *)target;
	OSNumber *num;
	uint32_t proto;

	(void)refCon;
	(void)notifier;
	if (!self || !newService)
		return true;

	num = OSDynamicCast(OSNumber, newService->getProperty("Protocol ID"));
	if (!num)
		return true;
	proto = num->unsigned32BitValue();
	if (proto != ODL_MAC_PROTOCOL_ID &&
	    proto != ODL_MAC_PROTOCOL_ID_APPLE)
		return true;

	self->logXDomain(newService);
	self->claimXDomain(newService);
	self->tryXDomainRespond(newService);
	IOLog("OdinLinkRDMA: peer advertised protocol %u. "
	      "Linux: bind_any/skip_login still works if this response "
	      "is ignored. Arm RX with odl_rdma_client -a for data.\n",
	      proto);
	return true;
}

void OdinLinkRDMA::claimXDomain(IOService *svc)
{
	OSNumber *pid;
	OSString *key;

	if (!svc)
		return;
	if (fXdService)
		fXdService->release();
	svc->retain();
	fXdService = svc;

	pid = OSNumber::withNumber((unsigned long long)ODL_MAC_PROTOCOL_ID, 32);
	key = OSString::withCString("odinlink");
	if (pid) {
		setProperty("Protocol ID", pid);
		pid->release();
	}
	if (key) {
		setProperty("Protocol Key", key);
		key->release();
	}
	setProperty("OdinLinkPeer", kOSBooleanTrue);
	IOLog("OdinLinkRDMA: claimed XDomain service, published "
	      "Protocol ID %u / key odinlink\n", ODL_MAC_PROTOCOL_ID);
}

void OdinLinkRDMA::tryXDomainRespond(IOService *svc)
{
	static const char *names[] = {
		"response",
		"sendResponse",
		"xdomainResponse",
		"completeRequest",
		NULL,
	};
	unsigned int i;
	IOReturn ret;

	if (!svc)
		return;

	/*
	 * IOThunderboltXDomainService has no public header. Try the
	 * symbols Apple uses internally; a miss is fine — Linux
	 * skip_login still brings the data path up.
	 */
	for (i = 0; names[i]; i++) {
		const OSSymbol *sym = OSSymbol::withCString(names[i]);
		if (!sym)
			continue;
		ret = svc->callPlatformFunction(sym, false, NULL, NULL,
						NULL, NULL);
		sym->release();
		if (ret != kIOReturnUnsupported) {
			IOLog("OdinLinkRDMA: XDomain %s -> 0x%x\n",
			      names[i], ret);
			if (ret == kIOReturnSuccess)
				return;
		}
	}
	IOLog("OdinLinkRDMA: no public XDomain response entry — "
	      "Linux should use skip_login=1 / bind_any\n");
}

IOReturn OdinLinkRDMA::armHardware(bool enable)
{
	if (!enable) {
		stopRxRing();
		return kIOReturnSuccess;
	}
	return startRxRing();
}

void OdinLinkRDMA::getLinkInfo(uint64_t *hop, uint64_t *armed,
			       uint64_t *rxDone, uint64_t *lastIdx)
{
	if (hop)
		*hop = (uint64_t)fRxHop;
	if (armed)
		*armed = fArmed ? 1 : 0;
	if (fLock)
		IOSimpleLockLock(fLock);
	if (rxDone)
		*rxDone = fRxDone;
	if (lastIdx)
		*lastIdx = fLastIdx;
	if (fLock)
		IOSimpleLockUnlock(fLock);
}
