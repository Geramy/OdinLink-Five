# OdinLink Thunderbolt 5, Thunderbolt 4 & USB4 / USB4v2

High-performance peer-to-peer DMA ring driver and toolchain for Thunderbolt 5, enabling GPU-to-GPU communication (via RCCL/NCCL), distributed file access, and performance testing between TB5-connected machines.

# Tested Systems
| System | USB4 | USB4v2 / TB5 | Status | Tester |
|-----------|----------------|-------------|-------------|----------|
| Minisforum MS-S1 | Working | Partially-Working | Requires power on with cable connected BIOS 1.06 | @Geramy |

Please submit a ticket proving your system works and i'll add you to the list.

## System Test on USB4v2 / Builtin - Thunderbolt 5
### WIP
- Firstly I would like to see if I can force TB5 to use more channels for TX on one host and RX on the other.
- Secondly I will add high performance mode which uses CPU polling at a higher rate to reduce latency
       The overall goal is to get latency to the level of infiniband RDMA / RoCE 2 - 5us
       
![OdinLink USB4 Performance Results](https://raw.githubusercontent.com/Geramy/OdinLink-Five/main/assets/Screenshot_2026-02-27_10-07-33.png)


## System Test on USB4 / Builtin - Thunderbolt 4

![OdinLink USB4 Performance Results](https://raw.githubusercontent.com/Geramy/OdinLink-Five/main/assets/Screenshot_2026-02-20_13-50-36.png)

## System Overview

OdinLink turns a Thunderbolt 5 cable into a high-speed interconnect between two Linux machines. The kernel driver manages NHI DMA rings over the TB5 PCIe tunnel, providing:

- **80 Gbps raw throughput** (Thunderbolt 5 bandwidth)
- **Sub-microsecond latency** for control messages
- **Zero-copy GPU transfers** via DMA-buf (RCCL plugin)
- **Configurable ring size** up to 64 MB per batch (256 MB total with double-buffered TX+RX)
- **Character device interface** (`/dev/odl_tb5_N`) with mmap'd double buffers

## Components

| Component | Binary / Module | Description |
|-----------|----------------|-------------|
| **Kernel Driver** | `odl_tb5.ko` | Thunderbolt service driver: NHI ring allocation, DMA buffer management, XDomain login/logout protocol |
| **Userspace Library** | `libodl_tb5.so` | C API for device open/close, double-buffer mmap, send/recv, DMA-buf, peer discovery |
| **RCCL Plugin** | `librccl_net_odl_tb5.so` | RCCL Net v7 network plugin for AMD GPU collective operations over TB5. Exposes shared-memory stats at `/run/odl_tb5/rccl_stats`. |
| **CLI Tool** | `odl_tb5_cli` | Client/server test tool: bandwidth, latency, jitter, latency-under-load, MIMO tests |
| **System Daemon** | `odl_tb5_daemon` | Background D-Bus service: device monitoring, test execution, RCCL stats, file operations |
| **Tray Application** | `odl_tb5_tray` | GTK3 system tray app: peer status, test runner, RCCL stats display, file management |
| **Test Suite** | `odl_tb5_test` | Unit and integration tests for the library and plugin |

## Tested Configuration

- **OS**: Ubuntu 24.04 LTS
- **Kernel**: 6.18.7 (Thunderbolt 5 support required)
- **Compiler**: GCC 14+ (must match kernel build compiler)
- **Build System**: CMake 3.10+
- **Hardware**: Thunderbolt 5 ports with bridge cable

## Quick Start

### Prerequisites

```bash
sudo apt update
sudo apt install build-essential cmake linux-headers-$(uname -r) pkg-config

# GCC version must match your kernel (check with: cat /proc/version)
sudo apt install gcc-14   # for kernel 6.18+
```

### Build (Core Only)

The core components (driver, library, RCCL plugin, CLI, tests) build with no extra dependencies:

```bash
git clone <repository-url> OdinLink-Five
cd OdinLink-Five
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with Daemon and Tray

The daemon and tray application require additional libraries. CMake auto-detects them and disables components if dependencies are missing.

**Daemon dependencies** (D-Bus service, device monitoring, test execution):
```bash
sudo apt install libglib2.0-dev
```

**Tray application dependencies** (system tray icon + GTK3 UI):
```bash
sudo apt install libgtk-3-dev libayatana-appindicator3-dev
```

**Optional - FUSE distributed file access** (transparent remote file reads over DMA):
```bash
sudo apt install libfuse3-dev
```

**Optional - SHA-256 for file operations** (used by the file transfer protocol):
```bash
sudo apt install libssl-dev
```

Then rebuild:
```bash
cd build
cmake .. && make -j$(nproc)
```

CMake will report which components are enabled:
```
-- BUILD_DAEMON: ON
-- BUILD_TRAY:   ON
```

### Load the Kernel Module

```bash
# Load with default ring size (4096 entries = 16 MB per batch)
sudo insmod driver/odl_tb5.ko

# Or load with custom ring size (power of 2, 64-16384)
sudo insmod driver/odl_tb5.ko ring_size=16384  # 64 MB per batch

# Verify
lsmod | grep odl_tb5
ls /dev/odl_tb5_*

# Install udev rule for persistent permissions
sudo cp driver/71-odl-tb5.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Run Performance Tests

Both machines must have the driver loaded and be connected via TB5 cable.

```bash
# Machine A (server):
./build/cli/odl_tb5_cli --server --device 0

# Machine B (client):
./build/cli/odl_tb5_cli --client --device 0 --test bandwidth
./build/cli/odl_tb5_cli --client --device 0 --test latency
./build/cli/odl_tb5_cli --client --device 0 --test jitter
./build/cli/odl_tb5_cli --client --device 0 --test latency-load
./build/cli/odl_tb5_cli --client --device 0 --test mimo
```

### Start the Daemon and Tray

```bash
# Start daemon (foreground for debugging):
./build/daemon/odl_tb5_daemon -f

# Or install the systemd user service:
systemctl --user enable --now odl-tb5-daemon

# Start tray application:
./build/tray/odl_tb5_tray
```

## Architecture

```
  Machine A                          Machine B
 +-----------+                     +-----------+
 | Tray App  |  D-Bus              | Tray App  |
 |  (GTK3)   |<------>+            |  (GTK3)   |
 +-----------+        |            +-----------+
                      v                   ^
               +------------+      +------------+
               |   Daemon   |      |   Daemon   |
               | (GLib/GIO) |      | (GLib/GIO) |
               +------+-----+      +-----+------+
                      |                   |
               +------v-----+      +-----v------+
               | libodl_tb5 |      | libodl_tb5 |
               +------+-----+      +-----+------+
                      |                   |
               +------v-----+      +-----v------+
               | odl_tb5.ko |      | odl_tb5.ko |
               +------+-----+      +-----+------+
                      |                   |
                      +---< TB5 Cable >---+
                         80 Gbps DMA
```

### Data Paths

- **Internal double-buffer path**: mmap'd 16-64 MB buffers for CLI tests, file transfers, control messages
- **External DMA-buf path**: zero-copy GPU memory transfers for RCCL collective operations

### Kernel Driver Architecture

`odl_tb5.ko` is built from four source files:

| File | Purpose |
|------|---------|
| `odl_tb5_service.c` | Thunderbolt service probe/remove, module init/exit, ring size module parameter |
| `odl_tb5_ring_dma.c` | NHI ring allocation, dynamic frame arrays, DMA buffer management |
| `odl_tb5_chardev.c` | Character device `/dev/odl_tb5_N`, ioctl dispatch, mmap handler |
| `odl_tb5_proto.c` | XDomain login/logout handshake over Thunderbolt properties protocol |

### Module Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ring_size` | 4096 | 64-16384 | NHI ring entries per direction (power of 2). Each entry = 4 KB. Default = 16 MB per batch, 64 MB total. |

## RCCL / GPU Usage

```bash
export RCCL_NET_PLUGIN=ODL_TB5
export RCCL_PLUGIN_DIR=/path/to/build/rccl

# Your RCCL/ROCm application will use TB5 automatically
```

The RCCL plugin exports shared-memory statistics at `/run/odl_tb5/rccl_stats`. The daemon reads these and exposes them via D-Bus; the tray app displays TX/RX bytes, operation counts, and uptime in a dedicated RCCL Stats window.

## Project Structure

```
OdinLink-Five/
+-- CMakeLists.txt                 Root build config + CPack packaging
+-- README.md
+-- driver/                        Kernel module (odl_tb5.ko)
|   +-- odl_tb5_service.c          Service driver registration
|   +-- odl_tb5_ring_dma.c         NHI ring + DMA buffer management
|   +-- odl_tb5_chardev.c          Character device interface
|   +-- odl_tb5_proto.c            XDomain login/logout protocol
|   +-- odl_tb5_core.h             Internal kernel header
|   +-- uapi/odl_tb5_uapi.h        Userspace API (ioctl defs, structs)
|   +-- 71-odl-tb5.rules           udev rules (uaccess, NHI runtime PM)
|   +-- Kbuild, Makefile
+-- lib/                           Userspace library (libodl_tb5.so)
|   +-- include/odl_tb5/
|   |   +-- odl_tb5.h              Public API
|   |   +-- odl_tb5_types.h        Shared type definitions
|   |   +-- odl_tb5_ioctl.h        Ioctl definitions (userspace mirror)
|   |   +-- odl_tb5_rccl_stats.h   RCCL shared-memory stats struct
|   +-- src/
|       +-- odl_tb5_dev.c           Device open/close, mmap
|       +-- odl_tb5_xfer.c          Send/recv (internal + DMA-buf)
|       +-- odl_tb5_peer.c          Peer discovery
|       +-- odl_tb5_completion.c    Poll/wait completions
+-- rccl/                          RCCL Net v7 plugin
|   +-- src/odl_tb5_plugin.c        Plugin with shared-memory stats
+-- cli/                           CLI test tool
|   +-- src/
|       +-- odl_tb5_cli_main.c      CLI entry point
|       +-- odl_tb5_cli.h           Protocol + test definitions
|       +-- odl_tb5_cli_proto.c     In-band control protocol
|       +-- odl_tb5_cli_server.c    Server mode
|       +-- odl_tb5_cli_client.c    Client mode
|       +-- odl_tb5_cli_bandwidth.c Bandwidth test
|       +-- odl_tb5_cli_latency.c   Latency test
|       +-- odl_tb5_cli_jitter.c    Jitter test
|       +-- odl_tb5_cli_latency_load.c  Latency-under-load test
|       +-- odl_tb5_cli_mimo.c      MIMO (multi-stream) test
|       +-- odl_tb5_cli_stats.c     Statistics + histograms
+-- daemon/                        System daemon (odl_tb5_daemon)
|   +-- src/
|   |   +-- odl_tb5_daemon_main.c   GMainLoop, signal handling
|   |   +-- odl_tb5_daemon_dbus.c/h D-Bus service (com.odinlink.Tb5Daemon)
|   |   +-- odl_tb5_daemon_monitor.c/h  Device scan (polls /dev/odl_tb5_N)
|   |   +-- odl_tb5_daemon_test.c/h     Test executor (GThreadPool)
|   |   +-- odl_tb5_daemon_rccl_stats.c/h  RCCL stats reader
|   |   +-- odl_tb5_daemon_sync.c/h     File operations engine
|   |   +-- odl_tb5_daemon_sync_proto.c/h  File transfer wire protocol
|   |   +-- odl_tb5_daemon_config.c/h   Config (~/.config/odl_tb5/)
|   +-- dbus/com.odinlink.Tb5Daemon.xml  D-Bus interface definition
|   +-- data/
|       +-- odl-tb5-daemon.service   Systemd user unit
|       +-- com.odinlink.Tb5Daemon.service  D-Bus activation
+-- tray/                          System tray application (odl_tb5_tray)
|   +-- src/
|   |   +-- odl_tb5_tray_main.c     GTK3 init, AppIndicator setup
|   |   +-- odl_tb5_tray.h          Internal header
|   |   +-- odl_tb5_tray_dbus.c     D-Bus proxy client
|   |   +-- odl_tb5_tray_menu.c     Tray menu + callbacks
|   |   +-- odl_tb5_tray_peers.c    Peer detail popup
|   |   +-- odl_tb5_tray_tests.c    Test runner dialog
|   |   +-- odl_tb5_tray_rccl.c     RCCL stats window
|   |   +-- odl_tb5_tray_sync.c     File management UI
|   +-- icons/                       SVG tray icons
|   +-- data/odl-tb5-tray.desktop    Autostart .desktop file
+-- tests/                         Unit + integration tests
|   +-- odl_tb5_test_main.c
|   +-- odl_tb5_test_device.c
|   +-- odl_tb5_test_lib_api.c
|   +-- odl_tb5_test_plugin.c
+-- packaging/                     .deb packaging (CPack + DKMS)
|   +-- dkms.conf, dkms-postinst.sh, dkms-prerm.sh
|   +-- daemon-postinst.sh
|   +-- build-meta-debs.sh.in
+-- third_party/rccl/net_v7.h     RCCL Net v7 header
```

## Build Dependencies Summary

| Component | Ubuntu Package | Required For |
|-----------|---------------|--------------|
| `build-essential` | `build-essential` | All (compiler + make) |
| `cmake` | `cmake` | All (build system) |
| `linux-headers` | `linux-headers-$(uname -r)` | Kernel module |
| `gcc-14+` | `gcc-14` | Kernel module (must match kernel) |
| `pkg-config` | `pkg-config` | Daemon + Tray dependency detection |
| `glib-2.0` | `libglib2.0-dev` | Daemon |
| `gio-2.0` | `libglib2.0-dev` | Daemon (D-Bus) |
| `gtk+-3.0` | `libgtk-3-dev` | Tray application |
| `ayatana-appindicator3` | `libayatana-appindicator3-dev` | Tray application (system tray icon) |
| `fuse3` | `libfuse3-dev` | Daemon (optional: FUSE distributed file access) |
| `openssl` | `libssl-dev` | Daemon (optional: SHA-256 for file operations) |

### Install All Dependencies

```bash
# Core (always needed):
sudo apt install build-essential cmake linux-headers-$(uname -r) gcc-14 pkg-config

# Daemon:
sudo apt install libglib2.0-dev

# Tray:
sudo apt install libgtk-3-dev libayatana-appindicator3-dev

# Optional (FUSE + file operations):
sudo apt install libfuse3-dev libssl-dev
```

## .deb Packages

Build installable packages:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cpack                    # Individual component .debs
make meta-packages       # User-friendly bundles
```

| Package | Contents |
|---------|----------|
| `odl-tb5-minimal` | dkms + library + RCCL plugin (GPU cluster node) |
| `odl-tb5-server` | dkms + library + CLI + daemon + RCCL plugin (headless server) |
| `odl-tb5-desktop` | dkms + library + CLI + daemon + tray (desktop workstation) |
| `odl-tb5-full` | Everything |

## Troubleshooting

1. **Kernel module build fails with unknown GCC flags**: Your GCC is too old. Install the version matching your kernel (`cat /proc/version`).
2. **Module won't load**: Check `dmesg | grep odl_tb5` for errors. Ensure TB5 hardware is present (`lspci | grep Thunderbolt`).
3. **No `/dev/odl_tb5_*` devices**: The device appears only when a TB5 peer connects. Check `dmesg` for XDomain events.
4. **Permission denied**: Install the udev rule or run `sudo chmod 660 /dev/odl_tb5_*`.
5. **Daemon won't start**: Check `journalctl --user -u odl-tb5-daemon` for D-Bus errors.
6. **Tray icon not visible**: Install `gnome-shell-extension-appindicator` on GNOME/Wayland desktops.

### Debug

```bash
# Kernel driver debug
echo 'module odl_tb5 +p' | sudo tee /sys/kernel/debug/dynamic_debug/control
dmesg -w | grep odl_tb5

# Daemon foreground with verbose output
./build/daemon/odl_tb5_daemon -f

# RCCL debug
export RCCL_DEBUG=INFO
export RCCL_NET_PLUGIN=ODL_TB5
```

## License

MIT
