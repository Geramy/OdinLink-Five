# OdinLink Apple Silicon Transport — Roadmap

## Current state (as of 2026-05)

### What just landed upstream

Sven Peter's ATCPHY driver (v3, merged 2025-12-23, commits a722de30..8e98ca1e):

- `drivers/phy/apple/atc.c` (2294 lines) — the Type-C PHY
- `drivers/soc/apple/tunable.c` — firmware tunable infrastructure (will be reused by NHI)
- `drivers/phy/apple/Kconfig` — `CONFIG_PHY_APPLE_ATC`

### What the ATCPHY can do today

| Mode | PHY lane config | Works? | Notes |
|------|----------------|--------|-------|
| USB2 | D+/D- only | Yes | Requires dwc3 glue (also upstream) |
| USB3 | 2 lanes USB3, 2 idle | Yes | Via pipehandler USB3 state |
| USB3+DP | 2 lanes USB3, 2 lanes DP | Yes | DP aux channel setup works |
| DP | 4 lanes DP | Yes | DP aux + PLL config |
| TBT | 4 lanes USB4/TBT | **No** | PHY lane mode defined, NHI missing |
| USB4 | 4 lanes USB4 | **No** | pipehandler USB4 state falls back to USB2 |

### Critical: what's missing

The ATCPHY driver sets up the **physical lanes** for USB4/TBT mode. But the
**NHI (Native Host Interface)** — the DMA engine that actually moves packets over
those lanes — does not exist yet in Linux.

This is the "Intel NHI" equivalent for Apple Silicon. Without it:
- No DMA rings (what OdinLink's transport ops need)
- No packet TX/RX
- No XDomain discovery or path management
- No interrupt signaling for completions

The tunable infrastructure (`soc: apple: tunable.c`) is explicitly designed
to be reused by the NHI driver. Sven's cover letter says:

> "The generic tunable support inside driver/soc/apple will also be re-used
> for Thunderbolt later."

## Architecture: Apple Silicon Thunderbolt stack

```
┌───────────────────────────────────────────────────────┐
│  OdinLink driver (transport_apple.c — IMPLEMENTED)   │
│  Uses odl_tb5_transport_ops, same as NHI backend     │
│                                                       │
│  • ACIO register layout parsed from DT tunables       │
│  • Per-HopID MSI-X interrupt routing                  │
│  • Shared TX descriptor buffer model                 │
│  • Full stopDMA: disable intr → clear ENABLE → done  │
│  • Apple XDomain login/logout (UUID 0xFA57)          │
├───────────────────────────────────────────────────────┤
│  apple_tb5_nhi_regs.h (STANDALONE SHARED HEADER)     │
│  • Full register map, descriptor format, DART info    │
│  • ACIO layout blob struct, HAL vtable offsets        │
│  • Usable by any Apple TB driver, not just OdinLink   │
├───────────────────────────────────────────────────────┤
│  odl_tb5_xd_proto_apple.h (APPLE PROTOCOL HEADER)    │
│  • Apple XDomain protocol UUID (0xFA57)               │
│  • Apple-style login/response/logout messages         │
│  • apple_tb5_xd_header_init() helper                  │
├───────────────────────────────────────────────────────┤
│  Apple NHI driver (NOT YET UPSTREAM)                  │
│  - Creates platform device from DT                    │
│  - DMA ring alloc/submit/callback                    │
│  - XDomain / USB4 tunnel management                  │
│  - Interrupt handling                                │
│  - DART (IOMMU) mappings                            │
├───────────────────────────────────────────────────────┤
│  ATCPHY (UPSTREAM — atc.c)                           │
│  - Lane muxing (USB3 / USB4 / TBT / DP)             │
│  - Pipehandler (PIPE mux for DWC3)                   │
│  - Power/clock management                            │
│  - Firmware tunable application                      │
├───────────────────────────────────────────────────────┤
│  Apple Fabric / ACIO (partially upstream)             │
│  - Interconnect, power domains                        │
└───────────────────────────────────────────────────────┘
```

## Key ATCPHY details relevant to OdinLink

### Pipehandler states

The ATCPHY's "pipehandler" is the USB3 PIPE mux. It has three states:

```c
enum atcphy_pipehandler_state {
    ATCPHY_PIPEHANDLER_STATE_DUMMY,   // USB2 only, USB3 disabled
    ATCPHY_PIPEHANDLER_STATE_USB3,   // Direct USB3 to Type-C port
    ATCPHY_PIPEHANDLER_STATE_USB4,   // USB3 tunneled through USB4
};
```

When OdinLink's Apple transport activates, it needs the pipehandler in
`USB4` state (USB3 tunneled). Today this falls back to dummy with a warning:

```c
case ATCPHY_PIPEHANDLER_STATE_USB4:
    dev_warn(atcphy->dev,
             "ATCPHY_PIPEHANDLER_STATE_USB4 not implemented; falling back to USB2\n");
    ret = atcphy_configure_pipehandler_dummy(atcphy);
```

### Lane and crossbar config for USB4/TBT

The ATCPHY already knows how to set up lanes for USB4 mode:

```c
[APPLE_ATCPHY_MODE_USB4] = {
    .normal = {
        .crossbar = ACIOPHY_CROSSBAR_PROTOCOL_USB4,
        .lane_mode = {ACIOPHY_LANE_MODE_USB4, ACIOPHY_LANE_MODE_USB4},
    },
    .pipehandler_state = ATCPHY_PIPEHANDLER_STATE_USB4,
},
```

This means the PHY is ready — the NHI just needs to exist to use those lanes.

### Register regions

From the device tree binding:

| Region | Purpose |
|--------|---------|
| `core` | Common controls (0x4c000 bytes) |
| `lpdptx` | DisplayPort TX (0x8000 bytes) |
| `axi2af` | AXI to Apple Fabric (0x4000 bytes) |
| `usb2phy` | USB2 PHY (0x4000 bytes) |
| `pipehandler` | USB3 PIPE mux (0x4000 bytes) |

The NHI will have its own register region (not part of ATCPHY). Based on
Apple's `IORDMAFamily` kext, the NHI likely sits behind the ACIO (Apple
Converged IO) fabric and has separate MMIO.

## What OdinLink's Apple transport will need

### Transport ops → Apple NHI mapping

| odl_tb5_transport_ops | Apple NHI equivalent | Status |
|------------------------|---------------------|--------|
| `ring_alloc` | Allocate shared TX + per-ring RX DMA descriptor rings | Done (shared TX buffer) |
| `ring_free` | Free those rings | Done |
| `ring_start/stop` | Enable/disable NHI DMA engine + per-HopID interrupts | Done (full stopDMA) |
| `ring_tx` | Submit TX descriptor to Apple NHI ring | Done |
| `ring_rx` | Post RX descriptor to Apple NHI ring | Done |
| `dma_device` | Return device for DART-mapped coherent DMA | Done |
| `path_enable` | Configure ATCPHY for USB4 mode + enable ACIO path | Stub (needs ATCPHY) |
| `path_disable` | Tear down USB4 path, return PHY to safe state | Done (stops rings) |
| `peer_send_login` | Send XDomain login via Apple's packet format (UUID 0xFA57) | Done |
| `peer_send_logout` | Send XDomain logout | Done |
| `kick_tx/rx` | Ring the NHI doorbell / kick work queue | Done |

### New concerns specific to Apple (status)

1. **DART (Apple IOMMU)**: All DMA addresses must go through DART translation.
   `dma_device()` returns the DART-mapped platform device. The Linux DMA API
   handles DART translation transparently. See `apple_tb5_nhi_regs.h` for
   DART SID ranges (0x00-0x0B, 0x10-0x1B).

2. **Apple Fabric (ACIO)**: The NHI sits behind the Apple Fabric interconnect.
   Power domains and clock gating must be managed. The ATCPHY already handles
   its own power domains; the NHI will need separate ones.

3. **Firmware tunables**: The NHI uses firmware tunable blobs to set register
   layouts. `apple_parse_acio_layout()` reads DT properties (falling back to
   M4 Pro defaults). The `apple_tunable_parse/apply` infrastructure is ready
   upstream.

4. **Ring format**: Determined from analysis. Apple uses 16-byte DMA descriptors
   (addr_lo, addr_hi, control, reserved). See `apple_tb5_nhi_regs.h` for the
   `struct apple_tb5_dma_desc` definition. Control word bit positions are
   inferred from Intel NHI and need hardware verification.

5. **Protocol differences**: Apple's ThunderboltRDMA uses protocol ID 0xFA57.
   `odl_tb5_xd_proto_apple.h` defines Apple-specific login/response/logout
   messages with `apple_tb5_xd_header_init()`. The existing `protocol=1` mode
   in `odl_tb5_proto.c` handles short Apple login packets on the RX side.

### New files (shared resources for Asahi kernel devs)

| File | Purpose |
|------|---------|
| `driver/apple_tb5_nhi_regs.h` | Standalone register map, descriptor format, DART info, HAL offsets. No OdinLink dependency. |
| `driver/odl_tb5_xd_proto_apple.h` | Apple XDomain protocol UUID (0xFA57), login/response/logout messages, header init helper. |

## Dependencies and timeline

```
ATCPHY (DONE) ──► Apple NHI driver (IN PROGRESS, Asahi)
                         │
                         ├── DMA ring format analysis (DONE — apple_tb5_nhi_regs.h)
                         ├── DART integration
                         ├── ACIO/Fabric bring-up
                         └── USB4 tunnel management
                                │
                                ▼
                    OdinLink Apple transport (IMPLEMENTED)
                         │
                         ├── odl_tb5_transport_apple.c (full impl)
                         ├── apple_tb5_nhi_regs.h (shared register map)
                         ├── odl_tb5_xd_proto_apple.h (Apple protocol)
                         ├── ACIO tunable layout parser
                         ├── Per-HopID MSI-X interrupt routing
                         ├── Shared TX descriptor buffer
                         └── Full stopDMA sequence
```

The NHI driver is likely 1-2 years from upstream (estimate based on ATCPHY
timeline: v1 in Oct 2025, v3 merged Dec 2025, and NHI is significantly
more complex).

## What we can do NOW

1. **Stub `odl_tb5_transport_apple.c`** — implement all ops as no-ops or
   `-ENODEV`, similar to the loopback transport. This lets us compile-test
   the Apple transport path and wire it into the build system.

2. **Add Kconfig gate** — `CONFIG_ODL_TB5_TRANSPORT_APPLE` that depends on
   `ARCH_APPLE` and the future `CONFIG_APPLE_NHI` (or equivalent).

3. **Study the ATCPHY API** — understand how to request USB4/TBT mode from
   the PHY. This will be the `path_enable` implementation.

4. **Study IORDMAFamily** — determine Apple's NHI ring format from
   the macOS kernel extension and related sources. This is the hardest part
   and doesn't require any hardware access — just the kext binary.

5. **DART integration prototype** — use `dma_alloc_coherent` on the DART-
   mapped device to test that coherent DMA allocation works on Apple Silicon.
   The AGX GPU driver already does this successfully.

## Apple NHI analysis results (from macOS 26.5 kexts, May 2026)

### Binaries analyzed

All 4 kexts extracted from the arm64e kernel cache on a Mac16,10 (M4 Pro, t8132):

| Kext | Size | Analyzed? |
|------|------|-------------------|
| `com.apple.driver.AppleThunderboltNHI` (7.2.81) | 729KB | Yes |
| `com.apple.iokit.IOThunderboltFamily` (9.3.3) | 3.7MB | Yes |
| `com.apple.driver.AppleThunderboltIP` (4.0.3) | 634KB | Yes |
| `com.apple.driver.AppleThunderboltUTDM` (3.0.7) | 207KB | Yes |

### Class hierarchy

```
AppleThunderboltHALGenericACIO  (the Apple Silicon HAL — our target)
├── AppleThunderboltHALType5     (M3/M4, "Type5" = t8132 ACIO)
├── AppleThunderboltHALType7     (M1/M2, "Type7" = t8103/t6000)
AppleThunderboltNHI              (base NHI — Intel-derived API)
├── AppleThunderboltNHIType5     (M3/M4 NHI, uses HALType5)
├── AppleThunderboltNHIType7     (M1/M2 NHI, uses HALType7)
AppleThunderboltNHIGenericACIO   (Apple Silicon NHI, uses HALGenericACIO)
```

The split is: `AppleThunderboltNHI` + its subclasses provide the IOKit
framework API (allocate rings, configure, start/stop). The HAL classes
provide the hardware-specific register access. On Apple Silicon,
`AppleThunderboltNHIGenericACIO` replaces the Intel NHI with ACIO fabric
access and DART IOMMU integration.

### Descriptor format (CONFIRMED from arm64e analysis)

The DMA ring uses **16-byte descriptors** (4 x 32-bit words), confirmed by:
- `logRings` debug format: `desc[%d] 0x%08x 0x%08x 0x%08x 0x%08x`
- `writeNextDescriptor`: copies 16 bytes with `ldr q0 / str q0`
- Descriptors indexed by `index << 4` (i.e., 16 bytes per slot)

**Descriptor word layout** (from `setDescCache` analysis):

```c
struct apple_tb_descriptor {
    u32 words[4];  // 16 bytes total
};
```

`IOThunderboltTransmitCommand::setDescCache(TxBufferDescriptor *desc, u64 offset)`:
1. Loads 8 bytes from `desc[0..7]` → stores to ring at `desc_index * 16`
2. Loads 4 bytes from `desc[8..11]` → stores to ring at `desc_index * 16 + 8`

This reveals the descriptor structure:

```
Bytes 0-3  (word 0): Buffer physical address (low 32 bits)
Bytes 4-7  (word 1): Buffer physical address (high 32 bits)
Bytes 8-11 (word 2): Control/length word
Bytes 12-15 (word 3): Unused (reserved / zero)
```

So `words[0:1]` form a 64-bit DMA address, and `word[2]` is the control word.
The control word likely contains: frame length, SOF/EOF flags, interrupt mode,
and producer/consumer index update bits (exact bit fields need further
analysis of the `buildTxBufferDescriptor` / `buildRxBufferDescriptor` methods).

This is very similar to the Intel NHI frame descriptor format, which also has
a physical address + control word layout.

### Ring management (from symbol table + debug strings)

Key confirmed register names:
- `kRegisterInterruptStatusMask0` — interrupt mask for ring events
- Ring table entries: `[0]` and `[1]` are separate register writes
  (ring table [0] and ring table [1] are written at different points
  during ring start/configuration)

Ring lifecycle (from symbol names and log strings):

1. **Init**: `initWithNHI(hopID)` — each ring is identified by a HopID
2. **Configure**: `configure()` — create soft interrupt, optionally
   set dedicated interrupt, create double buffers
3. **Start**: `start()` — write ring table [0] and ring desc, enable ring
4. **Submit**: `writeNextDescriptor()` — fill 4-word descriptor, update
   producer/consumer index
5. **Complete**: `checkForCompletedDescriptors()` — scan for completed
   descriptors, call back to command owner
6. **Stop**: `stop()` — disable ring, wait for ring disable done

Producer/Consumer index management:
- TX: `writeProducerIndexInternal(index)` writes the producer index for TX
- RX: `writeConsumerIndexInternal(index)` writes the consumer index for RX
- `isProducerWriteValid(index)` / `isConsumerWriteValid(index)` — validate
  index before writing (ring full check)

### Double buffering

Apple's NHI uses a "double buffer" scheme for frames that cross page
boundaries or aren't physically contiguous. The key strings show:

```
shouldDoubleBuffer - current frame (o=0x%x,l=0x%x) crosses a page boundary
shouldDoubleBuffer - current frame not address aligned / not length aligned
shouldDoubleBuffer - current frame not physically contiguous
```

For Apple Silicon (GenericACIO variant), the alignment requirements are:
- Address must be aligned
- Length must be aligned
- Must not cross page boundary
- Must be at least one full receive frame in size

This means the Apple transport will need to bounce-buffer or double-buffer
any packet that doesn't meet these constraints (similar to the Intel NHI
path but with Apple-specific alignment).

### Interrupt management

From the symbol table and debug strings:
- `enableInterrupt(hopID, enable)` — per-HopID interrupt enable/disable
- Mask is managed in 32-bit quads: `mask_quad = %d, mask_bit = %d`
- `kRegisterInterruptStatusMask0` — the interrupt status mask register
- Interrupt throttling rate is configurable per-hop: `0x%04x`
- Dedicated interrupts: rings can have their own workloop and dedicated
  interrupt (not shared with the main NHI interrupt)

### DART IOMMU integration

`AppleThunderboltNHIDARTVMAllocator` manages DART address space:
- Per-mapper allocation (separate TX and RX mappers per HopID)
- Supports fixed DART offsets (but limited)
- Min/max DVA and page size reported by DART at init
- `vmAlloc()` allocates DART-mapped memory via `IODMACommand`

The ioreg for `dart-acio0` shows:
- 128KB register space at `0x40024000`
- 16KB config space at `0x3800E0000`
- SID list: 0x00-0x0B, 0x10-0x1B (28 SIDs)
- Bypass flags for some SIDs

### ACIO device tree properties (from ioreg on Mac16,10)

The `acio0` device at `0x40100000` (1MB register space) exposes:
- **IODeviceMemory**:
  - `0x40100000` — 1MB (main NHI register space)
  - `0x400DB000` — 192KB
  - `0x400AC000` — 16KB (x3, likely per-ring spaces)
  - `0x400E4000` — 16KB
- **Tunable blobs** (firmware register patches applied at boot):
  - `hbw_fabric_tunables` — high-bandwidth fabric tuning
  - `hi_up_tx_desc_fabric_tunables` — TX descriptor fabric tuning
  - `hi_up_rx_desc_fabric_tunables` — RX descriptor fabric tuning
  - `hi_up_tx_data_fabric_tunables` — TX data fabric tuning
  - `hi_dn_merge_fabric_tunables` — downstream merge tuning
  - `fw_int_ctl_management_tunables` — interrupt control management
  - `pcie_adapter_regs_tunables` — PCIe adapter tuning
- **portmap**: maps physical ports to ACIO fabric endpoints
- **thunderbolt-drom**: device ROM data (vendor/device strings)
- **sat-dtf-enabled-ring-mask**: `0xFFFFFFFFFAFFFFFF` (which rings are DTF-enabled)
- **Power gates**: gates 0x62, 0x63, 0x71
- **24 interrupts** (12 TX + 12 RX for 12 HopIDs)

### IOThunderboltFamily: XDomain and path management

From the IOThunderboltFamily symbol table:
- `IOThunderboltXDomainServiceClientManager` — manages XDomain clients
- `IOThunderboltConfigXDomainCommand` — XDomain config space commands
- `IOThunderboltConfigXDomainPathRequestCommand` — path setup requests
- `IOThunderboltPath` — path object with source/destination initial credits
- `IOThunderboltControlPath` — control plane path
- `IOThunderboltControlPathListener` — listens for incoming connections
- `IOThunderboltDeficitCommandQueue` — deficit round-robin scheduler
- `IOThunderboltTimerCommandQueue` — timer-based command scheduling

Path capability exchange (from debug strings):
```
processPathCapabilitiesRequest - local Service Ready = %u, local XDomain Max Hop ID = %u, local Max Credits = %u
processPathCapabilitiesRequest - remote Service Ready = %u, remote XDomain Max Hop ID = %u, remote Max Credits = %u
```

This maps directly to OdinLink's `peer_send_login` / `peer_send_logout` ops.

### AppleThunderboltIP: the IP-over-Thunderbolt driver

This is the macOS equivalent of OdinLink — it tunnels network packets over
Thunderbolt using the same XDomain mechanism. Key insights:

- `AppleThunderboltIPTransmitter` — TX side
- `AppleThunderboltIPConnection` — bidirectional connection
- `AppleThunderboltIPControlCommand` — builds login/logout packets
- Uses `EFI_GUID` for service identification (like OdinLink's protocol ID)
- Supports aggregated packets (multiple small packets in one frame)
- Login packet contains: GUID, version, ring size, E2E support flag
- `getTxRingEntryCount()` / `getRxRingEntryCount()` — ring sizes
- `getTxE2EHopID()` — E2E (end-to-end) HopID for TX
- `setPDFBitmasks()` — Packet Descriptor Filter bitmasks

### What this means for OdinLink's Apple transport

The analysis confirms the design is sound. Key takeaways:

1. **Ring format is 16-byte descriptors** (4 x u32) — very similar to Intel
   NHI's frame descriptors. The `odl_tb5_ring_dma.c` descriptor handling
   should map cleanly.

2. **Producer/consumer index model** — same as Intel NHI. TX rings use
   producer index, RX rings use consumer index. The ring-full check
   (`isProducerWriteValid` / `isConsumerWriteValid`) is identical in
   concept to Intel's `tb_ring_full()`.

3. **Per-HopID DMA mappers via DART** — each HopID gets its own IOMMU
   address space. This means `dma_device()` needs to return the DART-
   mapped device for the specific HopID, not a single shared device.

4. **Tunable blobs must be applied** — the firmware tunable infrastructure
   (`apple_tunable_parse/apply`) will be needed at boot to configure
   fabric parameters. The tunable data is in the device tree.

5. **Double buffering for non-aligned frames** — Apple's alignment
   requirements are stricter than Intel's. The loopback transport handles
   this trivially (memcpy), but real hardware will need bounce buffering.

6. **XDomain path exchange is compatible** — Apple uses the same
   MaxHopID/MaxCredits/ServiceReady exchange that OdinLink implements.
   The `protocol=1` mode should work.

### Register map from arm64e analysis (May 2026)

The analysis is at `tools/ghidra_projects/decompiled_functions.asm` (4908 lines)
and `tools/ghidra_projects/decompiled_descriptors.asm` (3636 lines). Below is the
reconstructed register map from key functions.

#### `initRegisterLayout` — the register layout ID

`AppleThunderboltGenericACIO::initRegisterLayout()` returns `0xE00002C7`.
`AppleThunderboltHALType5::initRegisterLayout()` stores constants at object offsets
0x134 and 0x144, and loads a 128-bit value from `__TEXT` at file offset 0x7a0.
This value is the base register layout table — 4 x u32 words that define
the register offset base for the ACIO NHI.

#### `writeProducerIndexInternal` (TX ring)

The TX ring's `writeProducerIndex` function does the following:

1. Calls `isProducerWriteValid` (vtable offset 0x998) — checks if the ring has space
2. If valid: calls `registerWrite32ForRange(base + 8, index & 0xFFFF)` (vtable 0x8b8)
   - This writes the producer index to register `(ring_base + 8)`
   - The `index & 0xFFFF` masks the index to 16 bits
3. If NOT valid: calls `registerWrite32(base + 8, index << 16)` (vtable 0x8a8)
   - This writes a different format: index shifted left by 16

The kdebug trace code uses:
- `0x53440B4` for the TX producer index event
- `0x5344050` for the RX consumer index event

Both use `0x534` as the subsystem ID (AppleThunderboltNHI kdebug class).

#### `writeConsumerIndexInternal` (RX ring)

Mirrors the TX logic but:
- Calls `isConsumerWriteValid` (vtable 0x998, same offset — the HAL distinguishes internally)
- Valid path: `registerWrite32ForRange(base + 8, index & 0xFFFF)` (vtable 0x8b8)
- Invalid path: `registerWrite32(base + 8, index << 16)` (vtable 0x8a8)

The RX consumer index event is `0x5344050`.

#### `startDMA` (ReceiveRing) — the critical function

`AppleThunderboltNHIReceiveRing::startDMA()` performs these register operations:

1. **Get DMA buffer**: calls `getPhysicalAddress()` (vtable 0x140) on the shared buffer
   → returns a 64-bit physical address (high 32 bits extracted with `lsr x21, x20, #0x20`)

2. **Write buffer address**:
   - `registerWrite32(ring_num, addr64)` (vtable 0x8a8) — writes the full 64-bit address
   - `registerWrite32(ring_num + 4, addr_high32)` (vtable 0x8a8) — writes the high 32 bits

   So register offsets `ring_base + 0` and `ring_base + 4` hold the DMA buffer address.

3. **Write ring size + HopID**:
   - Loads `ring_size` from object offset 0x38 and `hop_id` from offset 0x40
   - `bfi w2, w8, #0x10, #0xc` — inserts hopID into bits [27:16] of the size word
   - `registerWrite32(ring_num + 0xC, packed_value)` (vtable 0x8a8)

   Register offset `ring_base + 0xC` = ring size + HopID packed word.

4. **Enable the ring**:
   - Calls `isProducerWriteValid()` (vtable 0x998) — checks if NHI is ready
   - If ready: `registerWrite32ForRange(ring_base + 8, credit_count & 0xFFFF)` (vtable 0x8b8)
   - If not: `registerWrite32(ring_base + 8, credit_count_raw)` (vtable 0x8a8)

   Register offset `ring_base + 0x8` = credit/enable register.

5. **Build the control word and write it**:
   The control word is assembled from multiple boolean checks:
   - `supportsTwoPageBuffers()` (vtable 0x1c8) → if true, sets bits [22:12] from `maxFrameSize` (vtable 0x1b8), plus `0x10000000`
   - `supportsCoalescing()` (vtable 0x1d8) → if true, ORs in `0x40000000`
   - `supportsIntOnDesc()` (vtable 0x1e8) → if true, ORs in `0x20000000`
   - Always ORs in `0x80000000` (the "enable" bit)

   Final control word = `0x80000000 | flags...`
   Written via `registerWrite32(hop_offset, control_word)` (vtable 0x8a8)

   `hop_offset` comes from object offset 0x34 — a separate register base for
   per-HopID control registers (different from the ring descriptor base).

6. **Kdebug trace**: `0x5344034` for startDMA event

7. **Enable interrupts**: calls `setInterruptEnable(1)` on the ring's workloop
   (vtable 0x208 on the event source, offset 0x150 on the interrupt controller)
   Then calls `ringEnable()` (vtable 0x310) to tell the HAL the ring is active.

8. **Check for double-buffer support**: tests bit 3 of a register, stores
   result at object offset 0xe0 (`usesDoubleBuffering`).

#### Per-ring register layout (reconstructed)

Each ring has TWO register bases:
- **Ring descriptor base** (from object offset 0x30): holds DMA buffer info
- **Hop control base** (from object offset 0x34): holds enable/flags

```
Ring descriptor registers (base = ring_desc_base):
  +0x00  DMA buffer address (low 32 bits)
  +0x04  DMA buffer address (high 32 bits)
  +0x08  Producer/Consumer index / credit count
  +0x0C  Ring size + HopID (bits [27:16] = HopID, [11:0] = size)

Hop control registers (base = hop_ctrl_base):
  +0x00  Control word:
         bit 31    = Enable (0x80000000)
         bit 30    = Coalescing support (0x40000000)
         bit 29    = Interrupt on descriptor (0x20000000)
         bits 27:16 = HopID (duplicated from ring descriptor)
         bits 22:12 = Max frame size (if two-page buffers)
         bit 28    = Two-page buffer support (0x10000000)
```

The ring base address is computed as: `base_register + (ring_number * ring_stride)`.
The `ring_number` is stored at object offset 0x30 and is passed as the first argument
to `registerWrite32`/`registerWrite32ForRange`.

#### `stopDMA` (ReceiveRing)

1. Checks if ring is enabled (vtable 0x170)
2. If enabled: disables interrupt (`setInterruptEnable(0)`), then:
   - Reads current control word: `registerRead32(hop_ctrl_base)` (vtable 0x8a0)
   - Clears the enable bit: `val & 0x7FFFFFFF`
   - Writes back: `registerWrite32(hop_ctrl_base, val & 0x7FFFFFFF)` (vtable 0x8a8)
3. Calls `ringDisable()` (vtable 0x318)
4. If ring wasn't enabled: returns `0xE00002CD` (kIOReturnNotReady equivalent)

#### Descriptor write (`writeNextDescriptor` — TransmitRing)

The TX descriptor write function shows:

1. Gets the next command from the queue
2. Checks ring capacity: compares `next_index + 1` against `ring_size` (object offset 0x38),
   wraps if needed; compares `next_index` against `consumer_index` (object offset 0x48)
3. Calls `buildTxBufferDescriptor()` on the command object (vtable 0x328) — this fills
   the 4-word descriptor in the command's buffer
4. **Copies 16 bytes** to the ring: `ldr q0, [cmd]; str q0, [ring_desc + (index * 16)]`
   — confirms descriptors are 16 bytes, indexed by `index << 4`
5. For the double-buffer case (index wraps around): same 16-byte copy but to the
   second page of the ring buffer
6. Marks the descriptor slot as "in use" in a separate status array at object offset 0x60
   (1 byte per slot, set to 1 when a descriptor is submitted)
7. Calls `writeProducerIndexForCommand()` (vtable 0x2b0) to notify hardware

#### Object layout (AppleThunderboltNHIReceiveRing / TransmitRing)

```
Offset  Field                         Notes
------  -----                         -----
0x00    vtable pointer                PAC-signed
0x10    IOKit object                  superclass ref
0x18    NHI parent object             back-pointer to AppleThunderboltNHI
0x20    HAL object                    register access via HAL vtable
0x28    (unknown)                     possibly ring config
0x2C    ring_id (u32)                 ring identifier / debug tag
0x30    ring_desc_base (u32)          register base for ring descriptor writes
0x34    hop_ctrl_base (u32)           register base for hop control writes
0x38    ring_size (u32)                number of descriptors in ring
0x40    hop_id (u32)                  HopID for this ring
0x48    consumer_index (u32)          RX: consumer, TX: producer index
0x50    last_written_index (u32)      last index written to hardware
0x58    command_ptrs (void**)          array of command object pointers (8 bytes each)
0x60    in_use_flags (u8*)            1 byte per descriptor slot
0x70    shared_buffer (IOBuffer*)      DMA buffer object
0x78    phys_addr (u64)               physical address of DMA buffer
0x80    command_queue (IOCommand*)     pending command queue
0xA0    queue_offset (u64)            offset into the command queue
0xE0    uses_double_buffer (u8)       whether ring uses double buffering
```

#### HAL vtable offsets (AppleThunderboltHALGenericACIO)

```
Offset  Method                        Signature
------  ------                        ---------
0x140   getPhysicalAddress            () → uint64_t
0x148   getBufferSize                () → uint32_t
0x8A0   registerRead32               (u32 offset) → u32
0x8A8   registerWrite32             (u32 offset, u32 value)
0x8B8   registerWrite32ForRange      (u32 offset, u32 value, ...)
0x998   isIndexWriteValid            (u32 index) → bool
0x9A8   registerRead32ForRange       (u32 offset) → u32
```

### Remaining unknowns

- **Descriptor word bit fields** — the 4 x u32 words are built by
  `buildTxBufferDescriptor()` / `buildRxBufferDescriptor()` in the IOThunderboltFamily
  kext (not the NHI kext). Need further analysis of that kext similarly.
  The `logRings` function (at 0x9d0a0a0) prints `desc[%d] 0x%08x 0x%08x 0x%08x 0x%08x`
  which confirms the 4-word format but not the fields.

- **Ring stride** — the spacing between consecutive ring register sets.
  Likely 0x10 (16 bytes per ring, matching the 4 register offsets 0x00-0x0C)
  but could be larger if there are padding/reserved registers.

- **ATCPHY USB4 pipehandler** — the `ATCPHY_PIPEHANDLER_STATE_USB4` state
  is still unimplemented in the upstream driver. The juicecultus WIP patch
  may fill this gap.

- **Doorbell mechanism** — `writeProducerIndexForCommand` calls through the
  HAL vtable to write the producer index, which doubles as the doorbell.
  There doesn't appear to be a separate doorbell register.

### Analysis files

```
tools/ghidra_projects/decompiled_functions.asm     — 4908 lines, main functions
tools/ghidra_projects/decompiled_descriptors.asm   — 3636 lines, descriptor functions
tools/ghidra_projects/arm64e_kexts/                 — raw arm64e Mach-O binaries
tools/ghidra_projects/AppleNHI.gpr+.rep             — Ghidra project (analyzed)
tools/ghidra_projects/AppleIOThunderbolt.gpr+.rep   — Ghidra project (analyzed)
```
