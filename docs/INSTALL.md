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

## Load the Kernel Module

```bash
# Load with default ring size (4096 entries = 16 MB per batch)
sudo insmod driver/odl_tb5.ko

# Or load with custom ring size (power of 2, 64-16384)
sudo insmod driver/odl_tb5.ko ring_size=16384

# Loopback mode (no cable needed)
sudo insmod driver/odl_tb5.ko loopback=1

# Apple-compatible protocol mode
sudo insmod driver/odl_tb5.ko protocol=1

# Verify
lsmod | grep odl_tb5
ls /dev/odl_tb5_*

# Install udev rule for persistent permissions
sudo cp driver/71-odl-tb5.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

## Install Verbs Provider Plugin

```bash
sudo mkdir -p /usr/lib/aarch64-linux-gnu/libibverbs
sudo cp build/verbs/libodl_tb5-rdmav34.so /usr/lib/aarch64-linux-gnu/libibverbs/
ibv_devinfo
```

## Run Performance Tests

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
