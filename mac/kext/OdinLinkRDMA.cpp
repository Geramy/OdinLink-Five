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

#include "OdinLinkRDMA.h"

#define super IOService

OSDefineMetaClassAndStructors(OdinLinkRDMA, IOService)

static const IOPhysicalAddress kInvalidPhysAddr = 0xFFFFFFFFFFFFFFFFULL;

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
	if (data) {
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

	fDMACommand = IODMACommand::withSpecification(
		kIODMACommandOutputHost,
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
		UInt32 numSeg = 1;
		IOReturn ret = fDMACommand->gen64IOVMSegments(
			&numSeg, &seg, 0);

		if (ret != kIOReturnSuccess || numSeg != 1) {
			IOLog("OdinLinkRDMA: gen64IOVMSegments failed: 0x%x\n", ret);
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
	fLock = IOSimpleLockAlloc();
	if (fLock)
		IOSimpleLockInit(fLock);

	registerService();

	IOLog("OdinLinkRDMA: started — buffer at DART addr 0x%016llx, "
	      "ready for DMA from Linux peer\n",
	      (unsigned long long)fBufferPhysAddr);

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
	if (!dict)
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

IOExternalMethodDispatch OdinLinkRDMAUserClient::sMethods[kOdinLinkClientNumMethods] = {
	[kOdinLinkGetBufferInfo] = {
		.function = (IOExternalMethodAction)&OdinLinkRDMAUserClient::externalGetBufferInfo,
		.checkScalarInputCount = 0,
		.checkStructureInputSize = 0,
		.checkScalarOutputCount = 4,
		.checkStructureOutputSize = 0,
	},
	[kOdinLinkSharedBufferCreate] = {
		.function = (IOExternalMethodAction)&OdinLinkRDMAUserClient::externalSharedBufferCreate,
		.checkScalarInputCount = 0,
		.checkStructureInputSize = 0,
		.checkScalarOutputCount = 1,
		.checkStructureOutputSize = 0,
	},
	[kOdinLinkGetFrameInfo] = {
		.function = (IOExternalMethodAction)&OdinLinkRDMAUserClient::externalGetFrameInfo,
		.checkScalarInputCount = 0,
		.checkStructureInputSize = 0,
		.checkScalarOutputCount = 2,
		.checkStructureOutputSize = 0,
	},
};

OSDefineMetaClassAndStructors(OdinLinkRDMAUserClient, IOUserClient)

bool OdinLinkRDMAUserClient::start(IOService *provider)
{
	if (!super::start(provider))
		return false;

	fProvider = OSDynamicCast(OdinLinkRDMA, provider);
	if (!fProvider)
		return false;

	fSharedMemory = NULL;

	IOLog("OdinLinkRDMAUserClient: started\n");
	return true;
}

void OdinLinkRDMAUserClient::stop(IOService *provider)
{
	IOLog("OdinLinkRDMAUserClient: stopping\n");

	if (fSharedMemory) {
		fSharedMemory->release();
		fSharedMemory = NULL;
	}

	super::stop(provider);
}

IOReturn OdinLinkRDMAUserClient::clientClose(void)
{
	IOLog("OdinLinkRDMAUserClient: client closed\n");

	if (fSharedMemory) {
		fSharedMemory->release();
		fSharedMemory = NULL;
	}

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
	arguments->scalarOutput[2] = ODINLINK_RDMA_FRAME_SIZE;
	arguments->scalarOutput[3] = ODINLINK_RDMA_FRAME_COUNT;

	return kIOReturnSuccess;
}

IOReturn OdinLinkRDMAUserClient::externalSharedBufferCreate(
	OdinLinkRDMAUserClient *target, void *reference,
	IOExternalMethodArguments *arguments)
{
	OdinLinkRDMA *prov = target->fProvider;
	if (!prov)
		return kIOReturnNotReady;

	IOMemoryDescriptor *bufMem = prov->getBufferMemory();
	if (!bufMem)
		return kIOReturnNoMemory;

	if (target->fSharedMemory) {
		target->fSharedMemory->release();
		target->fSharedMemory = NULL;
	}

	IOReturn ret = target->copyClientMemoryForVirtualAddress(
		(mach_vm_address_t)0, &target->fSharedMemory, 0, bufMem);

	if (ret != kIOReturnSuccess) {
		IOLog("OdinLinkRDMAUserClient: shared buffer create failed: 0x%x\n", ret);
		return ret;
	}

	arguments->scalarOutput[0] = (uint64_t)target->fSharedMemory;

	IOLog("OdinLinkRDMAUserClient: shared buffer created\n");
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
