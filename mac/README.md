# OdinLink RDMA — Mac + Linux Tensor Pipeline

Send tensors from Ubuntu → Thunderbolt 5 → Mac, stored in a shared DMA buffer.

## Architecture

```
┌─ Ubuntu PC ───────────────┐     TB5      ┌─ Mac ──────────────────────────┐
│                            │    cable    │                                 │
│  odl_tensor_send           │             │  OdinLinkRDMA.kext             │
│  ├─ ibv_open_device()     │             │  ├─ IOBufferMemoryDescriptor  │
│  ├─ ibv_reg_mr(tensor)    │  RDMA       │  ├─ IODMACommand (DART map)   │
│  └─ ibv_post_send() ──────│──write───►  │  └─ phys_addr = 0x...        │
│                            │             │                                 │
│  Tensor data flows via     │             │  odl_rdma_client               │
│  OdinLink kernel driver    │             │  ├─ IOServiceOpen()            │
│  → NHI DMA rings           │             │  ├─ mach_vm_map(shared_buf)    │
│  → Thunderbolt fabric      │             │  └─ Read tensor data          │
│  → Apple ACIO/NHI          │             │                                 │
│  → DART translates addr    │             │  (Optional) Metal:             │
│  → Writes land in buffer   │             │  MTLBuffer → same phys pages   │
└────────────────────────────┘             └─────────────────────────────────┘
```

Key insight: Apple Silicon has unified memory. CPU, GPU, and DMA all share the same physical RAM. Once DART maps the buffer for NHI DMA, Metal can wrap the same pages in a `MTLBuffer` with zero copies.

## Files

| File | Side | Purpose |
|------|------|---------|
| `kext/OdinLinkRDMA.cpp` | Mac | IOKit kext: allocates DART-mapped buffer, exposes via IOUserClient |
| `kext/OdinLinkRDMA.h` | Mac | Kext class declarations + shared constants |
| `kext/Info.plist` | Mac | Kext bundle config (matches `IOPlatformDevice` "apple,thunderbolt-nhi") |
| `odl_rdma_client.c` | Mac | Userspace client: connects to kext, mmaps buffer, polls for frames |
| `linux_test/odl_tensor_send.c` | Linux | Sends RGBA test frames via ibv verbs API |

## Mac Kext — How It Works

1. **Matches** on `IOPlatformDevice` with name `apple,thunderbolt-nhi` (the ACIO NHI node from device tree)
2. **Allocates** an `IOBufferMemoryDescriptor` (physically contiguous, 2 frames of 1920×1080 RGBA8 = ~16.5 MB)
3. **Creates** an `IODMACommand` which tells DART to map those pages for the NHI's DMA engine
4. **Exposes** the DART-translated physical address to userspace (the Linux peer needs this as the RDMA target)
5. **Shares** the buffer via `IOUserClient` shared memory — userspace mmaps it directly
6. **Polling**: userspace calls `kOdinLinkGetFrameInfo` to check for new frames (could be replaced with MSI notification)

## Linux Tensor Sender — How It Works

1. **Opens** the OdinLink verbs device (`ibv_open_device`)
2. **Creates** PD, CQ, QP (which maps to an OdinLink kernel stream)
3. **Registers** a memory region with the tensor data
4. **Posts** `IBV_WR_SEND` work requests at target FPS
5. Each send goes through: `ibv_post_send` → OdinLink stream → kernel DMA ring → Thunderbolt fabric → Mac NHI → DART → buffer

## Setup

### Mac (receiver)

```bash
# Build the kext (requires Xcode + matching SDK)
cd mac/kext
xcodebuild  # or manual: clang -arch arm64 -kernel -c OdinLinkRDMA.cpp ...

# Load (SIP must be disabled: csrutil disable in Recovery)
sudo cp -r OdinLinkRDMA.kext /tmp/
sudo kextutil /tmp/OdinLinkRDMA.kext

# Run the client to see the DART physical address
cd mac
clang -o odl_rdma_client odl_rdma_client.c -framework IOKit -framework CoreFoundation
./odl_rdma_client
# Output includes: "DART phys addr = 0x000000040XXXXXX"
# This is the address the Linux peer targets

# Monitor for frames
./odl_rdma_client --poll 1000

# Dump a frame to file
./odl_rdma_client --dump frame0.rgba
```

### Linux (sender)

```bash
# Build (from repo root)
mkdir build && cd build && cmake .. && make -j$(nproc)

# Load kernel module
sudo insmod driver/odl_tb5.ko

# Wait for peer connection (TB cable to Mac)
# The Mac's NHI kext + OdinLinkRDMA must be loaded

# Build the tensor sender
gcc -o odl_tensor_send ../mac/linux_test/odl_tensor_send.c \
    -I../lib/include -I../third_party/rccl \
    -L./lib -lodl_tb5 -libverbs -lpthread

# Send 1080p test frames at 30 FPS
LD_LIBRARY_PATH=./lib ./odl_tensor_send --width 1920 --height 1080 --fps 30
```

## Current Status

| Component | Status |
|-----------|--------|
| Mac IOKit kext | Code complete, needs compile-test on arm64 Mac |
| Mac userspace client | Code complete, needs compile-test |
| Linux tensor sender | Code complete, builds with OdinLink verbs |
| End-to-end DMA path | Blocked until Apple NHI platform driver exists on Linux (Asahi) |
| DART buffer mapping | Works via IODMACommand (standard IOKit API) |
| Metal MTLBuffer integration | Future — wrap same IOMemoryDescriptor in MTLBuffer |

## What Needs to Happen for End-to-End

1. **Apple NHI platform driver** (Linux side) — registers the ACIO MMIO and creates `/dev/odl_tb5_N`. In progress by Asahi community.
2. **ATCPHY USB4 pipehandler** — sets PHY lanes to USB4 mode. Partially upstream, USB4 state not yet functional.
3. **XDomain login** — Linux and Mac NHI kexts must exchange protocol IDs. Our `odl_tb5_xd_proto_apple.h` handles Apple UUID (0xFA57).
4. **DART address exchange** — The Mac kext prints the DART physical address. The Linux sender needs this as the RDMA remote address. Could automate via a handshake protocol on the TB link.
