/*
 * OdinLink RDMA — IOKit Kext Header
 *
 * Shared definitions between the kext and the userspace client.
 */

#ifndef ODINLINK_RDMA_H
#define ODINLINK_RDMA_H

/*
 * Everything above the KERNEL guard below is the userspace ABI — the buffer
 * geometry and the method selectors.  The kext's own class declarations pull
 * in C++ IOKit headers that only exist in the Kernel framework, so they must
 * stay out of a userland translation unit.
 */

#ifdef KERNEL
#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IODMACommand.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOTimerEventSource.h>
#include <IOKit/IOWorkLoop.h>
#include <libkern/OSAtomic.h>
#endif

#include "../include/odinlink_mac_proto.h"

/* ── Buffer layout ──────────────────────────────────────────────── */

/*
 * The DMA buffer is a simple ring of frames. Each frame is one
 * tensor / image from the Linux peer. The buffer is allocated
 * as physically contiguous memory so DART can map it for the
 * NHI's DMA engine.
 *
 * The Linux peer writes into this buffer via OdinLink's DMA rings.
 * The kext exposes it to userspace via IOUserClient shared memory.
 * Both sides see the same physical pages — zero copy.
 *
 * The buffer is sized for double-buffering: while userspace reads
 * frame N, the Linux peer can write frame N+1 into the other half.
 */

#define ODINLINK_RDMA_FRAME_SIZE       ODL_MAC_SLOT_BYTES
#define ODINLINK_RDMA_FRAME_COUNT      ODL_MAC_RX_SLOTS
#define ODINLINK_RDMA_BUFFER_SIZE      ODL_MAC_RX_WINDOW_BYTES

/* ── User client methods ─────────────────────────────────────────── */

enum OdinLinkRDMAClientMethods {
	kOdinLinkGetBufferInfo       = 0,  /* Out: phys, bytes, slot, slots */
	kOdinLinkGetFrameInfo        = 1,  /* Out: completed, slot_bytes */
	kOdinLinkGetLinkInfo         = 2,  /* Out: hop, armed, rx_done, last_idx */
	kOdinLinkArmHardware         = 3,  /* In: 1=arm RX ring (dangerous) */
	kOdinLinkClientNumMethods    = 4,
};

/*
 * Memory type for IOConnectMapMemory64().  The buffer is mapped through the
 * IOKit mapping path rather than handed out as a mach memory-entry name: an
 * io_connect_t is not a memory entry, so mach_vm_map() cannot consume one.
 */
enum OdinLinkRDMAMemoryTypes {
	kOdinLinkSharedBufferType    = 0,
};

/* ── Userspace connection interface ──────────────────────────────── */

/*
 * How to use from userspace:
 *
 *   1. io_connect_t conn = IOServiceOpen(..., "OdinLinkRDMAUserClient")
 *   2. IOConnectCallScalarMethod(conn, kOdinLinkGetBufferInfo, ...)
 *      → returns phys_addr, buffer_size, frame_size, frame_count
 *   3. IOConnectMapMemory64(conn, kOdinLinkSharedBufferType, mach_task_self(),
 *                           &addr, &size, kIOMapAnywhere)
 *      → now you can read the tensor data at addr
 *   4. Poll kOdinLinkGetFrameInfo to check for new frames
 */

/* ── Kext class declarations ────────────────────────────────────── */

#ifdef KERNEL

class OdinLinkRDMA : public IOService
{
	OSDeclareDefaultStructors(OdinLinkRDMA)

public:
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	IOReturn setProperties(OSObject *properties) override;

	IODMACommand     *getDMACommand() const;
	IOMemoryDescriptor *getBufferMemory() const;
	uint64_t          getBufferPhysAddr() const;
	uint64_t          getBufferSize() const;
	bool              getFrameInfo(uint64_t *frameCount, uint64_t *frameSize);
	void              getLinkInfo(uint64_t *hop, uint64_t *armed,
				      uint64_t *rxDone, uint64_t *lastIdx);
	IOReturn          armHardware(bool enable);

	void              pollRx(void);
	static void       timerFired(OSObject *owner, IOTimerEventSource *sender);
	static bool       xdomainAppeared(void *target, void *refCon,
					  IOService *newService,
					  IONotifier *notifier);

private:
	bool              mapNHI(IOService *provider);
	void              unmapNHI(void);
	IOReturn          startRxRing(void);
	void              stopRxRing(void);
	void              writeNHI(uint32_t offset, uint32_t value);
	uint32_t          readNHI(uint32_t offset);
	void              logXDomain(IOService *svc);

	IOBufferMemoryDescriptor  *fBufferMemory;
	IODMACommand              *fDMACommand;
	uint64_t                   fBufferPhysAddr;
	uint64_t                   fBufferBytes;
	uint64_t                   fFrameSize;
	uint64_t                   fFrameCount;
	bool                       fBufferReady;
	IOSimpleLock              *fLock;

	IOMemoryMap               *fNHIMap;
	volatile uint32_t         *fNHIRegs;
	uint32_t                   fNHISize;
	IOBufferMemoryDescriptor  *fDescMemory;
	IODMACommand              *fDescDMA;
	uint64_t                   fDescPhys;
	void                      *fDescVirt;
	int                        fRxHop;
	bool                       fArmed;
	uint32_t                   fLastIdx;
	uint64_t                   fRxDone;

	IOWorkLoop                *fWorkLoop;
	IOTimerEventSource        *fTimer;
	IONotifier                *fXdNotifier;
};

class OdinLinkRDMAUserClient : public IOUserClient
{
	OSDeclareDefaultStructors(OdinLinkRDMAUserClient)

public:
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	IOReturn clientClose(void) override;

	IOReturn externalMethod(uint32_t selector,
				IOExternalMethodArguments *arguments,
				IOExternalMethodDispatch *dispatch,
				OSObject *target,
				void *reference) override;

	IOReturn clientMemoryForType(UInt32 type,
				     IOOptionBits *options,
				     IOMemoryDescriptor **memory) override;

	static IOReturn externalGetBufferInfo(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

	static IOReturn externalGetFrameInfo(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

	static IOReturn externalGetLinkInfo(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

	static IOReturn externalArmHardware(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

private:
	OdinLinkRDMA  *fProvider;

	static IOExternalMethodDispatch sMethods[kOdinLinkClientNumMethods];
};

#endif /* KERNEL */

#endif /* ODINLINK_RDMA_H */
