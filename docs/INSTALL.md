# Installation Guide

## Full Build

### Prerequisites

```bash
sudo apt update
sudo apt install build-essential cmake linux-headers-$(uname -r) pkg-config

# GCC version must match your kernel (check with: cat /proc/version)
sudo apt install gcc-14   # for kernel 6.18+
```

### Build (Core Only)

Core components (driver, library, RCCL plugin, CLI, tests):

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build with Verbs Provider

```bash
sudo apt install libibverbs-dev rdma-core
cmake .. -DBUILD_VERBS=ON
make -j$(nproc) odl_tb5_verbs odl_tb5_verbs_provider
```

### Build with Daemon and Tray

Daemon dependencies:
```bash
sudo apt install libglib2.0-dev
```

Tray application dependencies:
```bash
sudo apt install libgtk-3-dev libayatana-appindicator3-dev
```

Optional:
```bash
sudo apt install libfuse3-dev   # FUSE distributed file access
sudo apt install libssl-dev     # SHA-256 for file operations
```

Then rebuild:
```bash
cmake .. && make -j$(nproc)
```

CMake reports which components are enabled:
```
-- BUILD_DAEMON: ON
-- BUILD_TRAY:   ON
```

## Userspace-Only Build (containers)

When the module runs on the host and only userspace is needed inside a
container (no kernel headers, no GUI):

```bash
# Fedora
dnf install -y rdma-core-devel libibverbs-utils pkgconf-pkg-config make gcc cmake

cmake .. -DBUILD_VERBS=ON -DBUILD_KERNEL_MODULE=OFF -DBUILD_DAEMON=OFF -DBUILD_TRAY=OFF
cmake --build . --target odl_tb5 odl_tb5_verbs rccl_net_odl_tb5 -j$(nproc)
```

If `/lib/modules/$(uname -r)/build` is missing, CMake defaults
`BUILD_KERNEL_MODULE=OFF` instead of hard-failing.

## Load the Kernel Module

```bash
# Load with default ring size (4096 entries = 16 MB per batch)
sudo insmod driver/odl_tb5.ko

# Custom ring size (power of 2, 64-16384). Name is odl_ring_size, not ring_size.
sudo insmod driver/odl_tb5.ko odl_ring_size=1024   # recommended with iommu=pt

# Loopback mode (no cable needed)
sudo insmod driver/odl_tb5.ko loopback=1

# Apple-compatible protocol mode
sudo insmod driver/odl_tb5.ko protocol=1

# Verify — /dev appears only after READY
lsmod | grep odl_tb5
dmesg | grep 'odl_tb5: entering READY'
ls /dev/odl_tb5_*

# Install udev rule for persistent permissions
sudo cp driver/71-odl-tb5.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

## Install Verbs Provider Plugin

```bash
# x86_64:
sudo mkdir -p /usr/lib/x86_64-linux-gnu/libibverbs
sudo cp build/verbs/libodl_tb5-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/

# aarch64:
# sudo mkdir -p /usr/lib/aarch64-linux-gnu/libibverbs
# sudo cp build/verbs/libodl_tb5-rdmav34.so /usr/lib/aarch64-linux-gnu/libibverbs/

# On OdinLink-only hosts the directory plugin may not load. Use:
LD_PRELOAD=build/verbs/libodl_tb5_verbs.so ibv_devinfo
```

## Run Performance Tests

Both machines must have the driver loaded and be connected via TB5 cable.
Wait for `odl_tb5: entering READY state` in dmesg on both sides first
(after a module reload there is a short DMA-ping window).

```bash
# Machine A (server) — positional mode, not --server:
./build/cli/odl_tb5_cli server -d 0

# Machine B (client):
./build/cli/odl_tb5_cli client -d 0 -t bandwidth -b 64K,1M,4M
./build/cli/odl_tb5_cli client -d 0 -t latency
./build/cli/odl_tb5_cli client -d 0 -t jitter
./build/cli/odl_tb5_cli client -d 0 -t latency-load
./build/cli/odl_tb5_cli client -d 0 -t mimo
```

## Start Daemon and Tray

```bash
# Start daemon (foreground for debugging):
./build/daemon/odl_tb5_daemon -f

# Or install the systemd user service:
systemctl --user enable --now odl-tb5-daemon

# Start tray application:
./build/tray/odl_tb5_tray
```

## Build Dependencies

| Component | Ubuntu Package | Required For |
|-----------|---------------|--------------|
| `build-essential` | `build-essential` | All (compiler + make) |
| `cmake` | `cmake` | All (build system) |
| `linux-headers` | `linux-headers-$(uname -r)` | Kernel module |
| `gcc-14+` | `gcc-14` | Kernel module (must match kernel) |
| `pkg-config` | `pkg-config` | Daemon + Tray dependency detection |
| `libibverbs-dev` | `libibverbs-dev` | Verbs provider |
| `rdma-core` | `rdma-core` | Verbs provider plugin |
| `glib-2.0` | `libglib2.0-dev` | Daemon |
| `gio-2.0` | `libglib2.0-dev` | Daemon (D-Bus) |
| `gtk+-3.0` | `libgtk-3-dev` | Tray application |
| `ayatana-appindicator3` | `libayatana-appindicator3-dev` | Tray application |
| `fuse3` | `libfuse3-dev` | Daemon (optional) |
| `openssl` | `libssl-dev` | Daemon (optional) |

### Install All

```bash
sudo apt install build-essential cmake linux-headers-$(uname -r) gcc-14 pkg-config \
    libibverbs-dev rdma-core libglib2.0-dev libgtk-3-dev libayatana-appindicator3-dev \
    libfuse3-dev libssl-dev
```
