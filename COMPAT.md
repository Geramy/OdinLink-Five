# Thunderbolt 5 RDMA — Mac ↔ Linux Cross-Platform Guide

## Goal

Make a Mac (Apple Silicon, macOS 26+) and a Linux machine (Ubuntu 24.04+)
talk RDMA over a Thunderbolt 5 cable, using the standard `ibv_*` verbs API.

## Architecture

```
Mac (Apple ThunderboltRDMA)              Linux (OdinLink-Five)
┌──────────────────────┐                ┌──────────────────────────┐
│ libibverbs (Apple)    │                │ libibverbs (rdma-core)   │
│ libthunderboltrdma.dyl│                │ libodl_tb5-rdmav34.so    │
│ AppleThunderboltRDMA. │                │ libodl_tb5.so            │
│   kext                │                │ odl_tb5.ko (kernel mod)  │
├──────────────────────┤                ├──────────────────────────┤
│ Intel TB5 NHI DMA     │◄═══ TB5 ════►│ Intel TB5 NHI DMA         │
└──────────────────────┘                └──────────────────────────┘
```

Both sides use the same Intel Thunderbolt NHI DMA engine. The wire
protocol is determined by the NHI hardware. What differs is:
- **Peer discovery**: XDomain property directories with protocol IDs
- **Login handshake**: messages exchanged over the XDomain control path
- **Protocol ID**: Apple uses `64087 (0xFA57)`, OdinLink uses `20236 (0x4F4C)`

## Platform Status

| Capability | macOS (Apple) | Linux (OdinLink-Five) |
|------------|---------------|----------------------|
| Kernel driver | `AppleThunderboltRDMA.kext` — ships with macOS | `odl_tb5.ko` — custom module |
| Userspace lib | `libthunderboltrdma.dylib` + `libibverbs` (Apple) | `libodl_tb5.so` |
| Verbs provider | Built into `libthunderboltrdma.dylib` | `libodl_tb5-rdmav34.so` |
| Device discovery | Via IOThunderboltXDomainService | `/dev/odl_tb5_N` scan |
| `ibv_open_device` | ✅ Yes (Apple verbs) | ✅ Yes (standalone lib + plugin) |
| `ibv_reg_mr` | ✅ Yes | ✅ Yes (host memory) |
| `ibv_reg_dmabuf_mr` | ✅ Yes (hidden but real) | ✅ Yes (via Linux DMA-buf) |
| `ibv_create_qp` | ✅ Yes | ✅ Yes (maps to streams) |
| `ibv_post_send` | ✅ Yes | ✅ Yes (async via workqueue) |
| `ibv_poll_cq` | ✅ Yes | ✅ Yes (eventfd-based) |
| `ibv_devinfo` | ✅ Yes | ✅ Yes (via provider plugin) |
| Zero-copy GPU | ✅ Metal → IOSurface dmabuf | ❌ Needs NCCL plugin |
| No-cable test | ❌ Requires TB5 cable | ✅ `loopback=1` module param |

## Apple's Protocol Details

From `AppleThunderboltRDMA.kext/Contents/Info.plist`:

```xml
<key>IOPropertyMatch</key>
<dict>
    <key>Protocol ID</key>
    <integer>64087</integer>       <!-- 0xFA57 -->
    <key>Protocol Version</key>
    <integer>1</integer>
</dict>
```

Apple's XDomain properties advertise:
- Protocol ID: `64087` (hex `0xFA57`)
- Protocol Version: `1`
- Provider match on `IOThunderboltXDomainService`

The kernel extension uses `IORDMAFamily` for NHI DMA ring management and
`IOThunderboltFamily` for XDomain control messages.

## OdinLink-Five Protocol Details

```
Protocol key:  "odinlink"
Protocol ID:   0x4F4C (ASCII "OL")
Protocol ver:  1
```

The kernel module:
- Advertises `"odinlink"` + `0x4F4C` in its XDomain property directory
- Uses standard Linux Thunderbolt NHI ring API (`tb_ring_alloc_tx/rx`)
- Custom login/logout messages over `tb_xdomain_request()`

## Making Them Talk

The protocol IDs MUST match for XDomain discovery to work. Options:

| Option | Work Required | Cross-Platform? |
|--------|---------------|-----------------|
| **A: OdinLink matches Apple** | Change OdinLink's protocol ID to 64087 + match login msg format | ✅ Linux ↔ Mac |
| **B: Implement both simultaneously** | OdinLink accepts both protocol IDs | ✅ Linux ↔ both |
| **C: Gateway mode** | Linux bridge between Apple protocol and OdinLink | ⚠️ Complex |

## Making Them Talk

The protocol IDs MUST match for XDomain discovery to work. OdinLink-Five
now supports a `protocol` module parameter:

```bash
# Default: talk to other OdinLink nodes (protocol ID 0x4F4C)
sudo insmod driver/odl_tb5.ko

# Apple mode: talk to macOS Thunderbolt RDMA (protocol ID 64087 / 0xFA57)
sudo insmod driver/odl_tb5.ko protocol=1
```

In Apple mode (protocol=1), the driver:
- Advertises Apple's protocol ID `64087` under property key `"rdma"`
- Also advertises OdinLink's original protocol under `"odinlink"`
- Responds to XDomain discovery from either side
- The login message format may still differ — see below

### Option A: Apple protocol mode (easiest)

```bash
sudo insmod driver/odl_tb5.ko protocol=1
```

The driver advertises Apple's `prtcid=64087` under `"rdma"` key, AND
advertises OdinLink's `prtcid=0x4F4C` under `"odinlink"` key.
This lets it connect to both macOS and other OdinLink nodes.

### Option B: Modify login message format

The XDomain login message format might differ between OdinLink and
Apple's ThunderboltRDMA. If ODL's login is rejected by macOS, the fix
would be in `driver/odl_tb5_proto.c` where `odl_tb5_login_msg` is
sent and received.

Apple's login format can be determined by:
- Disassembling `libthunderboltrdma.dylib` from macOS dyld shared cache
- Or running `ioreg -lw0 | grep -A20 ThunderboltRDMA` on a connected Mac
- Or capturing XDomain packets between two Macs

## Setup: macOS Side

Apple's ThunderboltRDMA ships with macOS and loads automatically when
a TB5 peer connects. No installation needed.

```bash
# Check if the kext is loaded:
kextstat | grep ThunderboltRDMA

# Check for Thunderbolt hardware:
system_profiler SPThunderboltDataType

# Check for RDMA interfaces:
ibv_devinfo 2>/dev/null || echo "No RDMA devices (no peer connected)"
```

## Setup: Linux Side (Ubuntu 24.04+)

### Prerequisites

```bash
sudo apt install build-essential cmake linux-headers-$(uname -r) \
    libibverbs-dev rdma-core pkg-config gcc-14
```

### Build

```bash
git clone https://github.com/johndpope/OdinLink-Five.git
cd OdinLink-Five
mkdir build && cd build
cmake .. -DBUILD_VERBS=ON
make -j$(nproc) odl_tb5_verbs odl_tb5_verbs_provider
```

### Install

```bash
# Install kernel module
sudo insmod driver/odl_tb5.ko

# Install verbs provider plugin
sudo cp verbs/libodl_tb5-rdmav34.so /usr/lib/aarch64-linux-gnu/libibverbs/

# Install udev rule
sudo cp driver/71-odl-tb5.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Verify

```bash
# Without a cable (loopback mode):
sudo insmod driver/odl_tb5.ko loopback=1
ibv_devinfo    # Should show odl_tb5 device

# With a cable to another Linux box:
sudo insmod driver/odl_tb5.ko
ibv_devinfo    # Shows odl_tb5 when peer connects
ls /dev/odl_tb5_*
```

## Testing

### Loopback test (no cable, Linux only)

```bash
sudo insmod driver/odl_tb5.ko loopback=1

# Test with verbs smoke test
LD_LIBRARY_PATH=build/verbs:build/lib \
build/verbs/tests/test_verbs_basic

# Or with mock (kernel module not needed):
mkfifo /dev/odl_tb5_0
LD_PRELOAD=verbs/tests/libodl_tb5_mock.so \
LD_LIBRARY_PATH=build/verbs:build/lib \
build/verbs/tests/test_verbs_mock_loopback
```

### Point-to-point (two Linux boxes)

```bash
# Machine A:
sudo insmod driver/odl_tb5.ko
build/cli/odl_tb5_cli --server --device 0

# Machine B:
sudo insmod driver/odl_tb5.ko
build/cli/odl_tb5_cli --client --device 0 --test bandwidth

# Or via verbs:
build/verbs/tests/test_verbs_basic
```

### Mac ↔ Linux (future — needs protocol matching)

```bash
# Linux side:
sudo insmod driver/odl_tb5.ko
# Module must advertise Apple's protocol ID 64087, not 0x4F4C
ibv_devinfo    # Shows peer once Mac connects
ibv_rc_pingpong -d odl_tb5_0  # Standard verbs test
```

## What Works Now (April 2026)

| Scenario | Status |
|----------|--------|
| Linux ↔ Linux (same OdinLink) | ✅ Kernel module + verbs provider tested |
| Linux loopback (no cable) | ✅ loopback=1 + mock test |
| Mac ↔ Mac (Apple native) | ✅ Ships with macOS |
| Mac ↔ Linux | 🚧 Protocol IDs differ — needs OdinLink to match Apple's 0xFA57 |
| Linux → Mac RDMA | 🚧 Protocol + login message format unknown |

## Files of Interest

```
driver/odl_tb5_proto.c        # XDomain login/logout handshake
driver/odl_tb5_core.h         # Device struct, ring contexts
driver/odl_tb5_service.c      # Module init, property directory
driver/odl_tb5_ring_dma.c     # NHI DMA ring operations
driver/odl_tb5_loopback.c     # Software loopback (no cable mode)
driver/uapi/odl_tb5_uapi.h    # Userspace ioctl interface
lib/src/odl_tb5_stream.c      # Stream-based I/O
verbs/src/odl_tb5_verbs_*.c   # Verbs provider
verbs/VERBS_PROVIDER.md       # Verbs provider manual
```

## Cross-Platform Protocol Investigation (WIP)

To determine Apple's exact XDomain login message format:

1. **Kext binary**: `AppleThunderboltRDMA.kext` — no Mach-O on macOS 26.5
   (may be built into kernel cache or DriverKit-based)
2. **User-space driver**: `libthunderboltrdma.dylib` — in dyld shared cache
3. **IORDMAFamily**: Dependency but binary not easily accessible
4. **XDomain protocol**: Uses `IOThunderboltXDomainService` for messages

The login message format can be determined by:
- Running Ghidra on an extracted copy of `libthunderboltrdma.dylib`
- Monitoring XDomain events with `ioreg` or custom IOKit tracing
- Protocol analysis: capture login packets between two Macs

## Development Notes

For protocol-level debugging on macOS:

```bash
# Dump IOKit Thunderbolt properties:
ioreg -r -c IOThunderboltXDomainService

# Trace kext activity (requires SIP-disabled or debug kernel):
log stream --predicate 'subsystem contains "Thunderbolt"'
```
