# TB5 RCCL Plugin

A high-performance RCCL (ROCm Communication Collectives Library) plugin for Thunderbolt 5 that enables GPU-to-GPU communication across Thunderbolt connections with direct DMA buffer support.

## System Overview

**What this is for:** This project enables AMD GPU clusters to communicate at high speeds using Thunderbolt 5 connections instead of traditional networking. It's designed for multi-GPU workstations, servers, or laptop clusters that need low-latency, high-bandwidth interconnects.

**Kernel Module Purpose:** The `tb5_dma_ring.ko` kernel module provides a character device interface (`/dev/tb5_ol_ring0`) that manages Thunderbolt ring buffers with DMA memory validation. It ensures safe, direct memory access between GPU memory and Thunderbolt transport layers.

## Tested System Configuration

- **Operating System**: Ubuntu 24.04 LTS
- **Linux Kernel**: 6.14.0-1018-oem (OEM kernel with Thunderbolt support)
- **Compiler**: GCC 13.3.0
- **Build System**: CMake 3.28.3
- **Hardware**: Thunderbolt 5 ports with bridge cable support

## Architecture Overview

This project implements a complete Thunderbolt 5 RCCL plugin stack:

- **RCCL Plugin** (`librccl_net_tb5.so`) - Implements RCCL net v7 interface
- **Userspace Manager** (`libtb5_ring.so`) - Provides ring buffer API with ioctl interface
- **Kernel Driver** (`tb5_dma_ring.ko`) - Validates DMA buffers and interfaces with Thunderbolt hardware
- **Test Program** (`tb5_test`) - Comprehensive testing suite

### Data Flow
```
RCCL Application → RCCL Plugin → Userspace Ring → Kernel Driver → Thunderbolt Hardware
                      ↓              ↓              ↓
                DMA Buffer      ioctl calls    DMA validation
```

## Quick Start

### Prerequisites
```bash
sudo apt update
sudo apt install build-essential cmake linux-headers-$(uname -r)
```

Testing kernel module...
✓ Kernel device opened successfully
Testing userspace ring API...
✓ Userspace ring opened successfully
✓ Completion check works (completed: 0)
✓ Userspace ring closed successfully
Testing NCCL plugin interface...
✓ NCCL plugin initialized
✓ Found 3 NCCL devices
✓ Device properties: Thunderbolt 5 DMA Ring, speed: 80 Gbps, ptrSupport: 1
Testing Thunderbolt communication interface...
✓ Thunderbolt ring interface opened successfully
✓ Completion check interface works (completed: 0)
✓ Thunderbolt communication interface test completed successfully
  (Note: Full DMA buffer testing requires GPU driver integration)
✓ All tests passed! TB5 NCCL plugin is working correctly.
```
### Build & Test
```bash
# Clone and build
git clone <repository-url>
cd <repository-name>
mkdir build && cd build
cmake .. && make

# Load kernel module
cd ../kernel
sudo insmod tb5_dma_ring.ko
sudo chmod 666 /dev/tb5_ol_ring0

# Run tests
cd ../build
./tb5_test
```

**Expected output:**
```
TB5 RCCL Plugin Test Program
Testing kernel module...
✓ Kernel device opened successfully
Testing userspace ring API...
✓ Userspace ring opened successfully
✓ Completion check works (completed: 0)
✓ Userspace ring closed successfully
Testing RCCL plugin interface...
✓ RCCL plugin initialized
✓ Found 3 RCCL devices
✓ Device properties: Thunderbolt 5 DMA Ring, speed: 80 Gbps, ptrSupport: 1
Testing Thunderbolt communication interface...
✓ Thunderbolt ring interface opened successfully
✓ Completion check interface works (completed: 0)
✓ Thunderbolt communication interface test completed successfully
  (Note: Full DMA buffer testing requires GPU driver integration)
✓ All tests passed! TB5 RCCL plugin is working correctly.
```
============================
Testing kernel module...
✓ Kernel device opened successfully
Testing userspace ring API...
✓ Userspace ring opened successfully
✓ Completion check works (completed: 0)
✓ Userspace ring closed successfully
Testing NCCL plugin interface...
✓ NCCL plugin initialized
✓ Found 3 NCCL devices
✓ Device properties: Thunderbolt 5 DMA Ring, speed: 80 Gbps, ptrSupport: 1
Testing Thunderbolt communication interface...
✓ Thunderbolt ring interface opened successfully
✓ Completion check interface works (completed: 0)
✓ Thunderbolt communication interface test completed successfully
  (Note: Full DMA buffer testing requires GPU driver integration)
============================
✓ All tests passed! TB5 NCCL plugin is working correctly.
```

## Detailed Build Instructions

### Build Steps
```bash
# 1. System setup
sudo apt install build-essential cmake linux-headers-$(uname -r)

# 2. Build configuration
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. Build all components
make -j$(nproc)

# 4. Verify build
ls -la *.so tb5_test ../kernel/tb5_dma_ring.ko
```

### Kernel Module Loading
```bash
# Build kernel module
cd kernel && make

# Load module
sudo insmod tb5_dma_ring.ko

# Verify
lsmod | grep tb5
ls -la /dev/tb5_ol_ring0

# Set permissions
sudo chmod 666 /dev/tb5_ol_ring0
```

## Usage in RCCL Applications

### Environment Setup
```bash
export RCCL_PLUGIN_DIR=/path/to/project/build
export RCCL_NET_PLUGIN=tb5
```

### Example Code
```c
#include <rccl.h>

// Initialize RCCL with Thunderbolt plugin
rcclCommInitAll(&comm, nDev, devs);

// Plugin handles Thunderbolt communication automatically
rcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
```

## Project Structure
```
<project-root>/
├── CMakeLists.txt              # Main build configuration
├── test_main.c                 # Test program
├── README.md                   # This documentation
├── .gitignore                  # Git ignore rules
├── kernel/                     # Kernel module
│   ├── CMakeLists.txt
│   ├── Makefile
│   └── tb5_dma_ring.c         # Kernel driver
├── plugin/                     # RCCL plugin
│   ├── CMakeLists.txt
│   └── src/tb5_plugin.c       # Plugin implementation
├── userspace/                  # Userspace library
│   ├── CMakeLists.txt
│   ├── include/tb5/tb5_ring.h # API header
│   └── src/tb5_ring.c         # Ring library
└── third_party/rccl/net_v7.h   # RCCL headers
```

## Troubleshooting

### Common Issues
1. **Module won't load**: Check kernel version with `uname -r`
2. **Device not found**: Verify Thunderbolt hardware with `lspci | grep Thunderbolt`
3. **Permission denied**: Run `sudo chmod 666 /dev/tb5_ol_ring0`
4. **Test failures**: Check kernel logs with `dmesg | grep TB5`

### Debug Mode
```bash
# Enable kernel debug
echo 'module tb5_dma_ring +p' | sudo tee /sys/kernel/debug/dynamic_debug/control

# Enable RCCL debug
export RCCL_DEBUG=INFO
```

## Development

### Code Style
- Follow Linux kernel coding standards for C code
- Add comprehensive tests for new features
- Update documentation for API changes

### Kernel Compatibility
Tested on Linux 6.14.0-1018-oem with GCC 13.3.0 and CMake 3.28.3

## License
MIT
