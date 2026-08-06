/*
 * OdinLink RDMA — IOKit Kext Header
 *
 * Shared definitions between the kext and the userspace client.
 */

#ifndef ODINLINK_RDMA_H
#define ODINLINK_RDMA_H

#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IODMACommand.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <libkern/OSAtomic.h>

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

#define ODINLINK_RDMA_FRAME_SIZE       (1920ULL * 1080 * 4)  /* 1920x1080 RGBA8 = 8.25 MB */
#define ODINLINK_RDMA_FRAME_COUNT      2                     /* double-buffered */
#define ODINLINK_RDMA_BUFFER_SIZE      (ODINLINK_RDMA_FRAME_SIZE * ODINLINK_RDMA_FRAME_COUNT)

/* ── User client methods ─────────────────────────────────────────── */

enum OdinLinkRDMAClientMethods {
	kOdinLinkGetBufferInfo       = 0,  /* Out: phys_addr, size, frame_size, frame_count */
	kOdinLinkSharedBufferCreate  = 1,  /* Out: shared_memory_handle (mmap this) */
	kOdinLinkGetFrameInfo        = 2,  /* Out: frame_count, frame_size (poll for new data) */
	kOdinLinkClientNumMethods    = 3,
};

/* ── Userspace connection interface ──────────────────────────────── */

/*
 * How to use from userspace:
 *
 *   1. io_connect_t conn = IOServiceOpen(..., "OdinLinkRDMAUserClient")
 *   2. IOConnectCallScalarMethod(conn, kOdinLinkGetBufferInfo, ...)
 *      → returns phys_addr, buffer_size, frame_size, frame_count
 *   3. IOConnectCallScalarMethod(conn, kOdinLinkSharedBufferCreate, ...)
 *      → returns a mach_vm_address_t for mmap
 *   4. mmap(0, buffer_size, PROT_READ|PROT_WRITE, MAP_SHARED, mem_fd, 0)
 *      → now you can read the tensor data
 *   5. Poll kOdinLinkGetFrameInfo to check for new frames
 */

/* ── Kext class declarations ────────────────────────────────────── */

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

private:
	IOBufferMemoryDescriptor  *fBufferMemory;
	IODMACommand              *fDMACommand;
	uint64_t                   fBufferPhysAddr;
	uint64_t                   fBufferBytes;
	uint64_t                   fFrameSize;
	uint64_t                   fFrameCount;
	bool                       fBufferReady;
	IOSimpleLock              *fLock;
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

	static IOReturn externalGetBufferInfo(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

	static IOReturn externalSharedBufferCreate(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

	static IOReturn externalGetFrameInfo(
		OdinLinkRDMAUserClient *target, void *reference,
		IOExternalMethodArguments *arguments);

private:
	OdinLinkRDMA  *fProvider;
	IOMemoryMap   *fSharedMemory;

	static IOExternalMethodDispatch sMethods[kOdinLinkClientNumMethods];
};

#endif /* ODINLINK_RDMA_H */
