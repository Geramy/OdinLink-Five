/*
 * OdinLink RDMA — Mac Userspace Client
 *
 * Connects to the OdinLinkRDMA kext, mmaps the shared DMA buffer,
 * and dumps frame data as it arrives from the Linux peer.
 *
 * Build:
 *   clang -o odl_rdma_client odl_rdma_client.c -framework IOKit -framework CoreFoundation
 *
 * Run:
 *   ./odl_rdma_client
 *   ./odl_rdma_client --dump frame0.rgba    # dump first frame to file
 *   ./odl_rdma_client --poll 100            # poll for 100 iterations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include "kext/OdinLinkRDMA.h"
#include "include/odinlink_mac_proto.h"

static volatile int g_running = 1;

static void signal_handler(int sig)
{
	(void)sig;
	g_running = 0;
}

static io_service_t find_odinlink_service(void)
{
	CFMutableDictionaryRef matching = IOServiceMatching("OdinLinkRDMA");
	if (!matching)
		return IO_OBJECT_NULL;

	io_service_t service = IOServiceGetMatchingService(
		kIOMasterPortDefault, matching);

	return service;
}

int main(int argc, char **argv)
{
	const char *dump_path = NULL;
	int poll_count = 0;
	int do_arm = 0;
	int opt;

	while ((opt = getopt(argc, argv, "d:p:a")) != -1) {
		switch (opt) {
		case 'd': dump_path = optarg; break;
		case 'p': poll_count = atoi(optarg); break;
		case 'a': do_arm = 1; break;
		default:
			fprintf(stderr,
				"Usage: %s [-d dump.bin] [-p poll_count] [-a]\n"
				"  -a  arm the unverified ACIO RX ring\n",
				argv[0]);
			return 1;
		}
	}

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	printf("OdinLink RDMA Client\n");

	/* ── Find kext service ───────────────────────────────────────── */

	io_service_t service = find_odinlink_service();
	if (service == IO_OBJECT_NULL) {
		fprintf(stderr, "OdinLinkRDMA kext not found. Is it loaded?\n");
		fprintf(stderr, "  sudo kextutil /tmp/OdinLinkRDMA.kext\n");
		return 1;
	}

	printf("Found OdinLinkRDMA service\n");

	/* ── Open connection ─────────────────────────────────────────── */

	io_connect_t conn = IO_OBJECT_NULL;
	kern_return_t kr;

	kr = IOServiceOpen(service, mach_task_self(), 0, &conn);
	IOObjectRelease(service);

	if (kr != KERN_SUCCESS) {
		fprintf(stderr, "IOServiceOpen failed: %s\n",
			mach_error_string(kr));
		return 1;
	}

	printf("Connected to kext (conn=0x%x)\n", conn);

	/* ── Get buffer info ─────────────────────────────────────────── */

	uint64_t phys_addr = 0;
	uint64_t buf_size = 0;
	uint64_t frame_size = 0;
	uint64_t frame_count = 0;
	{
		uint64_t output[4] = {};
		uint32_t output_count = 4;

		kr = IOConnectCallScalarMethod(
			conn,
			kOdinLinkGetBufferInfo,
			NULL, 0,
			output, &output_count);

		if (kr != KERN_SUCCESS) {
			fprintf(stderr, "GetBufferInfo failed: %s\n",
				mach_error_string(kr));
			goto err_close;
		}

		phys_addr   = output[0];
		buf_size    = output[1];
		frame_size  = output[2];
		frame_count = output[3];

		printf("Buffer info:\n");
		printf("  DART phys addr: 0x%016llx\n",
		       (unsigned long long)phys_addr);
		printf("  Buffer size:    %llu bytes (%.1f MB)\n",
		       (unsigned long long)buf_size,
		       (double)buf_size / (1 << 20));
		printf("  Frame size:     %llu bytes (%.1f MB)\n",
		       (unsigned long long)frame_size,
		       (double)frame_size / (1 << 20));
		printf("  Slot count:     %llu\n",
		       (unsigned long long)frame_count);
	}

	{
		uint64_t output[4] = {};
		uint32_t output_count = 4;

		kr = IOConnectCallScalarMethod(conn, kOdinLinkGetLinkInfo,
					       NULL, 0, output, &output_count);
		if (kr == KERN_SUCCESS)
			printf("  hop=%llu armed=%llu rx_done=%llu last_idx=%llu\n",
			       (unsigned long long)output[0],
			       (unsigned long long)output[1],
			       (unsigned long long)output[2],
			       (unsigned long long)output[3]);
	}

	if (do_arm) {
		uint64_t in = 1;
		uint64_t out = 0;
		uint32_t out_count = 1;

		printf("Arming ACIO RX ring (unverified map)...\n");
		kr = IOConnectCallScalarMethod(conn, kOdinLinkArmHardware,
					       &in, 1, &out, &out_count);
		if (kr != KERN_SUCCESS) {
			fprintf(stderr, "ArmHardware failed: %s\n",
				mach_error_string(kr));
			goto err_close;
		}
		printf("Armed.\n");
	}

	/*
	 * The DART phys addr is what the Linux peer needs to target
	 * with its RDMA writes. On the Linux side, you'd do:
	 *
	 *   ibv_post_send(RDMA_WRITE, remote_addr=phys_addr, rkey=...)
	 *
	 * The OdinLink driver on Linux will DMA directly into this
	 * buffer via the NHI's DMA rings. DART translates the NHI's
	 * DMA addresses to point at these physical pages.
	 */

	printf("\n  *** Share this with the Linux peer: ***\n");
	printf("  DART phys addr = 0x%016llx\n",
	       (unsigned long long)phys_addr);
	printf("  Buffer size    = %llu\n",
	       (unsigned long long)buf_size);
	printf("\n");

	/* ── Map shared buffer ───────────────────────────────────────── */

	mach_vm_address_t shared_addr = 0;
	mach_vm_size_t shared_size = 0;

	kr = IOConnectMapMemory64(conn,
				  kOdinLinkSharedBufferType,
				  mach_task_self(),
				  &shared_addr,
				  &shared_size,
				  kIOMapAnywhere);

	if (kr != KERN_SUCCESS) {
		fprintf(stderr, "IOConnectMapMemory64 failed: %s\n",
			mach_error_string(kr));
		goto err_close;
	}

	printf("Shared buffer mapped at %p (%llu bytes)\n",
	       (void *)shared_addr, (unsigned long long)shared_size);

	/* ── Poll for frames or dump ─────────────────────────────────── */

	if (dump_path) {
		printf("Waiting for first frame, then dumping to %s...\n",
		       dump_path);

		uint64_t last_count = 0;
		while (g_running) {
			uint64_t output[2] = {};
			uint32_t output_count = 2;

			kr = IOConnectCallScalarMethod(
				conn,
				kOdinLinkGetFrameInfo,
				NULL, 0,
				output, &output_count);

			if (kr == KERN_SUCCESS && output[0] > last_count) {
				printf("Frame %llu arrived (size=%llu)\n",
				       (unsigned long long)output[0],
				       (unsigned long long)output[1]);
				last_count = output[0];
				break;
			}

			usleep(10000);
		}

		FILE *f = fopen(dump_path, "wb");
		if (f) {
			fwrite((void *)shared_addr, 1, frame_size, f);
			fclose(f);
			printf("Dumped %llu bytes to %s\n",
			       (unsigned long long)frame_size, dump_path);
		} else {
			fprintf(stderr, "Failed to open %s: %s\n",
				dump_path, strerror(errno));
		}
	} else if (poll_count > 0) {
		printf("Polling for %d iterations...\n", poll_count);

		uint64_t last_count = 0;
		for (int i = 0; i < poll_count && g_running; i++) {
			uint64_t output[2] = {};
			uint32_t output_count = 2;

			kr = IOConnectCallScalarMethod(
				conn,
				kOdinLinkGetFrameInfo,
				NULL, 0,
				output, &output_count);

			if (kr == KERN_SUCCESS && output[0] > last_count) {
				uint32_t *pixels = (uint32_t *)shared_addr;
				uint32_t first_pixel = pixels[0];
				uint32_t mid_pixel = pixels[frame_size / 8];

				printf("[%d] Frame %llu: first_pixel=0x%08x "
				       "mid_pixel=0x%08x\n",
				       i,
				       (unsigned long long)output[0],
				       first_pixel, mid_pixel);
				last_count = output[0];
			}

			usleep(100000);
		}
	} else {
		printf("Monitoring for frames... (Ctrl-C to stop)\n");

		uint64_t last_count = 0;
		while (g_running) {
			uint64_t output[2] = {};
			uint32_t output_count = 2;

			kr = IOConnectCallScalarMethod(
				conn,
				kOdinLinkGetFrameInfo,
				NULL, 0,
				output, &output_count);

			if (kr == KERN_SUCCESS && output[0] > last_count) {
				printf("Frame %llu arrived (size=%llu)\n",
				       (unsigned long long)output[0],
				       (unsigned long long)output[1]);
				last_count = output[0];
			}

			usleep(10000);
		}
	}

	/* ── Cleanup ─────────────────────────────────────────────────── */

	if (shared_addr)
		IOConnectUnmapMemory64(conn, kOdinLinkSharedBufferType,
				       mach_task_self(), shared_addr);

err_close:
	IOServiceClose(conn);
	printf("Disconnected\n");
	return 0;
}
