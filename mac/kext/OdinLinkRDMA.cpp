/*
 * OdinLink RDMA — IOKit Kernel Extension
 *
 * Receives DMA writes from a Linux peer over Thunderbolt 5 and stores
 * them in a DART-mapped buffer that userspace can mmap.
 *
 * Architecture:
 *
 *   Linux (OdinLink driver)                        Mac (this kext)
 *   ──────────────────────                         ────────────────
 *   ibv_post_send(tensor_data)  ───TB5 DMA───►   ACIO DMA ring
 *                                                   → DMA write lands in buffer
 *                                                   → IRQ / doorbell notifies kext
 *                                                   → Userspace mmaps the buffer
 *                                                   → App reads tensor data
 *
 * The buffer is allocated via IOMemoryDescriptor + IODMACommand so DART
 * maps it for the NHI's DMA engine. The same physical pages are exposed
 * to userspace via IOUserClient shared memory — zero copy.
 *
 * Build:
 *   xcrun -sdk macosx.internal make   (or xcodebuild with appropriate SDK)
 *
 * Load (SIP must be disabled):
 *   sudo cp -r OdinLinkRDMA.kext /tmp/
 *   sudo kextutil /tmp/OdinLinkRDMA.kext
 *
 * Unload:
 *   sudo kextunload -b com.odinlink.rdma
 */

#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IODMACommand.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IODeviceTreeSupport.h>
#include <libkern/OSAtomic.h>
#include <libkern/libkern.h>

#include "OdinLinkRDMA.h"

#define super IOService

OSDefineMetaClassAndStructors(OdinLinkRDMA, IOService)

bool OdinLinkRDMA::start(IOService *provider)
{
	if (!super::start(provider))
		return false;

	OSString *prop;
	OSData *data;

	prop = OSDynamicCast(OSString, provider->getProperty("apple,peer-route"));
	if (prop) {
		IOLog("OdinLinkRDMA: peer-route = %s\n", prop->getCStringNoCopy());
	}

	data = OSDynamicCast(OSData, provider->getProperty("reg"));
	if (data && data->getLength() >= 2 * sizeof(uint32_t)) {
		const uint32_t *reg = (const uint32_t *)data->getBytesNoCopy();
		IOLog("OdinLinkRDMA: reg = 0x%08x 0x%08x\n", reg[0], reg[1]);
	}

	fBufferBytes = ODINLINK_RDMA_BUFFER_SIZE;
	fFrameSize = ODINLINK_RDMA_FRAME_SIZE;

	IOLog("OdinLinkRDMA: allocating %llu byte DMA buffer "
	      "(%llu frames x %llu bytes)\n",
	      fBufferBytes,
	      fBufferBytes / fFrameSize,
	      fFrameSize);

	fBufferMemory = IOBufferMemoryDescriptor::withOptions(
		kIODirectionInOut | kIOMemoryPhysicallyContiguous,
		fBufferBytes,
		4096);

	if (!fBufferMemory) {
		IOLog("OdinLinkRDMA: failed to allocate %llu byte buffer\n",
		      fBufferBytes);
		return false;
	}

	/* 32 address bits: the Apple DART aperture, matching APPLE_TB5_DMA_BITS
	 * on the Linux side. */
	fDMACommand = IODMACommand::withSpecification(
		kIODMACommandOutputHost64,
		32,
		0,
		IODMACommand::kMapped,
		0,
		1);

	if (!fDMACommand) {
		IOLog("OdinLinkRDMA: failed to create IODMACommand\n");
		goto err_free_buffer;
	}

	{
		IOReturn ret = fDMACommand->setMemoryDescriptor(fBufferMemory);
		if (ret != kIOReturnSuccess) {
			IOLog("OdinLinkRDMA: setMemoryDescriptor failed: 0x%x\n", ret);
			goto err_free_dma;
		}
	}

	{
		IODMACommand::Segment64 seg;
		UInt64 offset = 0;
		UInt32 numSeg = 1;

		/* (offset, segments, numSegments) — the previous call had the
		 * first and last arguments transposed. */
		IOReturn ret = fDMACommand->gen64IOVMSegments(
			&offset, &seg, &numSeg);

		if (ret != kIOReturnSuccess || numSeg != 1) {
			IOLog("OdinLinkRDMA: gen64IOVMSegments failed: 0x%x\n", ret);
			goto err_free_dma;
		}

		/* A single segment must cover the whole buffer, or the peer's
		 * RDMA writes would run off the end of the first one. */
		if (seg.fLength < fBufferBytes) {
			IOLog("OdinLinkRDMA: DART gave a %llu byte segment for a "
			      "%llu byte buffer\n",
			      (unsigned long long)seg.fLength,
			      (unsigned long long)fBufferBytes);
			goto err_free_dma;
		}

		fBufferPhysAddr = seg.fIOVMAddr;
		IOLog("OdinLinkRDMA: buffer phys (DART-translated) = 0x%016llx, "
		      "size = %llu\n",
		      (unsigned long long)fBufferPhysAddr,
		      (unsigned long long)fBufferBytes);
	}

	fBufferReady = false;
	fFrameCount = 0;
	fRxHop = ODL_MAC_DEFAULT_RX_HOP;
	fArmed = false;
	fLastIdx = 0;
	fRxDone = 0;
	fNHIMap = NULL;
	fNHIRegs = NULL;
	fNHISize = 0;
	fDescMemory = NULL;
	fDescDMA = NULL;
	fDescPhys = 0;
	fDescVirt = NULL;
	fWorkLoop = NULL;
	fTimer = NULL;
	fXdNotifier = NULL;
	fXdService = NULL;
	fLock = IOSimpleLockAlloc();
	if (fLock)
		IOSimpleLockInit(fLock);

	mapNHI(provider);

	fWorkLoop = IOWorkLoop::workLoop();
	if (fWorkLoop) {
		fTimer = IOTimerEventSource::timerEventSource(this, timerFired);
		if (fTimer) {
			fWorkLoop->addEventSource(fTimer);
			fTimer->setTimeoutMS(1);
		}
	}

	{
		OSDictionary *matching =
			IOService::serviceMatching("IOThunderboltXDomainService");
		if (matching)
			fXdNotifier = addMatchingNotification(
				gIOFirstMatchNotification, matching,
				xdomainAppeared, this);
	}

	registerService();

	IOLog("OdinLinkRDMA: started — buffer DART 0x%016llx (%llu x %u), "
	      "NHI %s, hardware NOT armed. This kext does not yet publish "
	      "an XDomain directory (Linux /dev appears only when the Mac "
	      "advertises 0x4F4C). We watch for the Linux advertisement. "
	      "Arm with odl_rdma_client -a only if you accept the "
	      "unverified ACIO map.\n",
	      (unsigned long long)fBufferPhysAddr,
	      (unsigned long long)ODL_MAC_RX_SLOTS, ODL_MAC_SLOT_BYTES,
	      fNHIRegs ? "mapped" : "missing");

	return true;

err_free_dma:
	if (fDMACommand) {
		fDMACommand->clearMemoryDescriptor();
		fDMACommand->release();
		fDMACommand = NULL;
	}
err_free_buffer:
	if (fBufferMemory) {
		fBufferMemory->release();
		fBufferMemory = NULL;
	}
	return false;
}

void OdinLinkRDMA::stop(IOService *provider)
{
	IOLog("OdinLinkRDMA: stopping\n");

	if (fXdNotifier) {
		fXdNotifier->remove();
		fXdNotifier = NULL;
	}
	if (fXdService) {
		fXdService->release();
		fXdService = NULL;
	}
	if (fTimer) {
		fTimer->cancelTimeout();
		if (fWorkLoop)
			fWorkLoop->removeEventSource(fTimer);
		fTimer->release();
		fTimer = NULL;
	}
	if (fWorkLoop) {
		fWorkLoop->release();
		fWorkLoop = NULL;
	}
	stopRxRing();
	unmapNHI();

	if (fDMACommand) {
		fDMACommand->clearMemoryDescriptor();
		fDMACommand->release();
		fDMACommand = NULL;
	}

	if (fBufferMemory) {
		fBufferMemory->release();
		fBufferMemory = NULL;
	}

	if (fLock) {
		IOSimpleLockFree(fLock);
		fLock = NULL;
	}

	super::stop(provider);
}

IOReturn OdinLinkRDMA::setProperties(OSObject *properties)
{
	OSDictionary *dict = OSDynamicCast(OSDictionary, properties);
	if (!dict || !fLock)
		return kIOReturnBadArgument;

	OSNumber *frameCount = OSDynamicCast(OSNumber,
					     dict->getObject("FrameCount"));
	if (frameCount) {
		IOSimpleLockLock(fLock);
		fFrameCount = frameCount->unsigned64BitValue();
		fBufferReady = true;
		IOSimpleLockUnlock(fLock);
		return kIOReturnSuccess;
	}

	return kIOReturnBadArgument;
}

IODMACommand *OdinLinkRDMA::getDMACommand() const
{
	return fDMACommand;
}

IOMemoryDescriptor *OdinLinkRDMA::getBufferMemory() const
{
	return fBufferMemory;
}

uint64_t OdinLinkRDMA::getBufferPhysAddr() const
{
	return fBufferPhysAddr;
}

uint64_t OdinLinkRDMA::getBufferSize() const
{
	return fBufferBytes;
}

bool OdinLinkRDMA::getFrameInfo(uint64_t *frameCount, uint64_t *frameSize)
{
	if (!fLock)
		return false;

	IOSimpleLockLock(fLock);
	if (frameCount)
		*frameCount = fFrameCount;
	if (frameSize)
		*frameSize = fFrameSize;
	bool ready = fBufferReady;
	fBufferReady = false;
	IOSimpleLockUnlock(fLock);

	return ready;
}

/*
 * Positional, in enum order — array designators are a C99 construct that C++
 * only accepts as a compiler extension.
 */
IOExternalMethodDispatch OdinLinkRDMAUserClient::sMethods[kOdinLinkClientNumMethods] = {
	{	/* kOdinLinkGetBufferInfo */
		(IOExternalMethodAction)&OdinLinkRDMAUserClient::externalGetBufferInfo,
		0, 0, 4, 0,
	},
	{	/* kOdinLinkGetFrameInfo */
		(IOExternalMethodAction)&OdinLinkRDMAUserClient::externalGetFrameInfo,
		0, 0, 2, 0,
	},
	{	/* kOdinLinkGetLinkInfo */
		(IOExternalMethodAction)&OdinLinkRDMAUserClient::externalGetLinkInfo,
		0, 0, 4, 0,
	},
	{	/* kOdinLinkArmHardware */
		(IOExternalMethodAction)&OdinLinkRDMAUserClient::externalArmHardware,
		1, 0, 1, 0,
	},
};

OSDefineMetaClassAndStructors(OdinLinkRDMAUserClient, IOUserClient)

/* The user client's base class is IOUserClient, not IOService. */
#undef super
#define super IOUserClient

bool OdinLinkRDMAUserClient::start(IOService *provider)
{
	if (!super::start(provider))
		return false;

	fProvider = OSDynamicCast(OdinLinkRDMA, provider);
	if (!fProvider)
		return false;

	IOLog("OdinLinkRDMAUserClient: started\n");
	return true;
}

void OdinLinkRDMAUserClient::stop(IOService *provider)
{
	IOLog("OdinLinkRDMAUserClient: stopping\n");
	super::stop(provider);
}

IOReturn OdinLinkRDMAUserClient::clientClose(void)
{
	IOLog("OdinLinkRDMAUserClient: client closed\n");

	terminate();
	return kIOReturnSuccess;
}

IOReturn OdinLinkRDMAUserClient::externalMethod(
	uint32_t selector, IOExternalMethodArguments *arguments,
	IOExternalMethodDispatch *dispatch, OSObject *target, void *reference)
{
	if (selector >= kOdinLinkClientNumMethods)
		return kIOReturnBadArgument;

	dispatch = &sMethods[selector];
	target = this;

	return super::externalMethod(selector, arguments, dispatch,
				     target, reference);
}

IOReturn OdinLinkRDMAUserClient::externalGetBufferInfo(
	OdinLinkRDMAUserClient *target, void *reference,
	IOExternalMethodArguments *arguments)
{
	OdinLinkRDMA *prov = target->fProvider;
	if (!prov)
		return kIOReturnNotReady;

	arguments->scalarOutput[0] = prov->getBufferPhysAddr();
	arguments->scalarOutput[1] = prov->getBufferSize();
	arguments->scalarOutput[2] = ODL_MAC_SLOT_BYTES;
	arguments->scalarOutput[3] = ODL_MAC_RX_SLOTS;

	return kIOReturnSuccess;
}

/*
 * Backs IOConnectMapMemory64().  IOKit releases the descriptor once the
 * mapping is built, so hand back a retained reference rather than the
 * provider's only one.
 */
IOReturn OdinLinkRDMAUserClient::clientMemoryForType(
	UInt32 type, IOOptionBits *options, IOMemoryDescriptor **memory)
{
	if (type != kOdinLinkSharedBufferType)
		return kIOReturnUnsupported;

	if (!fProvider)
		return kIOReturnNotReady;

	IOMemoryDescriptor *bufMem = fProvider->getBufferMemory();
	if (!bufMem)
		return kIOReturnNoMemory;

	bufMem->retain();

	*options = 0;
	*memory = bufMem;

	return kIOReturnSuccess;
}

IOReturn OdinLinkRDMAUserClient::externalGetFrameInfo(
	OdinLinkRDMAUserClient *target, void *reference,
	IOExternalMethodArguments *arguments)
{
	OdinLinkRDMA *prov = target->fProvider;
	if (!prov)
		return kIOReturnNotReady;

	uint64_t frameCount = 0;
	uint64_t frameSize = 0;

	bool ready = prov->getFrameInfo(&frameCount, &frameSize);

	arguments->scalarOutput[0] = frameCount;
	arguments->scalarOutput[1] = frameSize;

	return ready ? kIOReturnSuccess : kIOReturnNoResources;
}

IOReturn OdinLinkRDMAUserClient::externalGetLinkInfo(
	OdinLinkRDMAUserClient *target, void *reference,
	IOExternalMethodArguments *arguments)
{
	OdinLinkRDMA *prov = target->fProvider;

	(void)reference;
	if (!prov)
		return kIOReturnNotReady;

	prov->getLinkInfo(&arguments->scalarOutput[0],
			  &arguments->scalarOutput[1],
			  &arguments->scalarOutput[2],
			  &arguments->scalarOutput[3]);
	return kIOReturnSuccess;
}

IOReturn OdinLinkRDMAUserClient::externalArmHardware(
	OdinLinkRDMAUserClient *target, void *reference,
	IOExternalMethodArguments *arguments)
{
	OdinLinkRDMA *prov = target->fProvider;

	(void)reference;
	if (!prov)
		return kIOReturnNotReady;
	if (arguments->scalarInputCount < 1)
		return kIOReturnBadArgument;

	IOReturn ret = prov->armHardware(arguments->scalarInput[0] != 0);
	arguments->scalarOutput[0] = (ret == kIOReturnSuccess) ? 1 : 0;
	return ret;
}
