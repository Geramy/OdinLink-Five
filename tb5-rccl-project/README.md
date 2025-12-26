# TB5 NCCL Plugin

A high-performance NCCL (NVIDIA Collective Communications Library) plugin for Thunderbolt 5, enabling GPU-to-GPU communication across Thunderbolt connections with DMA buffer support.

## Architecture Overview

This project implements a complete Thunderbolt 5 NCCL plugin stack:

- **NCCL Plugin** (`librccl_net_tb5.so`) - Implements NCCL net v7 interface
- **Userspace Manager** (`libtb5_ring.so`) - Provides ring buffer API with ioctl interface
- **Kernel Driver** (`tb5_dma_ring.ko`) - Kernel module with DMA buffer validation
- **Test Program** (`tb5_test`) - Comprehensive testing suite

### Data Flow
```
NCCL Application → NCCL Plugin → Userspace Ring → Kernel Driver → Thunderbolt Hardware
                      ↓              ↓              ↓
                DMA Buffer      ioctl calls    DMA validation
```

## Prerequisites

- **Linux Kernel**: 6.14.0-1018-oem (or compatible)
- **Thunderbolt 5 Hardware**: System with Thunderbolt 5 ports
- **Build Tools**:
  - CMake 3.10+
  - GCC compiler
  - Linux kernel headers
- **Permissions**: Root access for kernel module loading

## Quick Start

### 1. Clone and Build

```bash
git clone <repository-url>
cd tb5-rccl-project

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Optional: Build in parallel
make -j$(nproc)
```

### 2. Load Kernel Module

```bash
# Build and load kernel module
cd ../kernel
make
sudo insmod tb5_dma_ring.ko

# Verify module is loaded
lsmod | grep tb5
ls -la /dev/tb5_ol_ring0
```

### 3. Set Device Permissions

```bash
# Allow userspace access to the device
sudo chmod 666 /dev/tb5_ol_ring0
```

### 4. Run Tests

```bash
cd ../build

# Run comprehensive test suite
./tb5_test
```

Expected output:
```
TB5 NCCL Plugin Test Program
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

### Full Build Process

```bash
# 1. Prepare build environment
cd tb5-rccl-project
mkdir build && cd build

# 2. Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. Build userspace components
make

# 4. Build kernel module
cd ../kernel
make

# 5. Install kernel module
sudo make install  # Optional: copies .ko to /lib/modules/
```

### Build Options

- **Debug Build**: `cmake .. -DCMAKE_BUILD_TYPE=Debug`
- **Verbose Output**: `make VERBOSE=1`
- **Parallel Build**: `make -j$(nproc)`

### Generated Files

After successful build:
```
tb5-rccl-project/
├── build/
│   ├── tb5_test              # Test executable
│   ├── librccl_net_tb5.so    # NCCL plugin library
│   └── libtb5_ring.so        # Userspace ring library
├── kernel/
│   └── tb5_dma_ring.ko       # Kernel module
└── /dev/tb5_ol_ring0         # Device node (after insmod)
```

## Usage in NCCL Applications

### Environment Variables

Set the plugin path for NCCL:

```bash
export NCCL_PLUGIN_DIR=/path/to/tb5-rccl-project/build
export NCCL_NET_PLUGIN=tb5
```

### Thunderbolt Connection

1. **Connect Thunderbolt 5 bridge cable** between two systems
2. **Verify connection**: Check `dmesg` for Thunderbolt device enumeration
3. **Load module**: `sudo insmod tb5_dma_ring.ko`

### Example NCCL Application

```c
#include <nccl.h>

// Initialize NCCL with Thunderbolt plugin
ncclCommInitAll(&comm, nDev, devs);

// Plugin automatically handles:
// - Thunderbolt device discovery
// - DMA buffer allocation
// - Ring buffer communication
// - Error handling and recovery

ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
```

## Testing and Validation

### Test Suite Components

The `tb5_test` program validates:

1. **Kernel Module Interface**
   - Device node creation
   - Basic ioctl functionality

2. **Userspace Ring API**
   - Library loading
   - Ring operations
   - Completion handling

3. **NCCL Plugin Interface**
   - Plugin initialization
   - Device enumeration
   - Property reporting

4. **Thunderbolt Communication**
   - Interface validation
   - DMA buffer framework readiness

### Running Individual Tests

```bash
cd build

# Test kernel module only
./tb5_test  # Runs all tests

# Manual verification
lsmod | grep tb5           # Check module loaded
ls -la /dev/tb5_*          # Check device nodes
dmesg | grep TB5           # Check kernel messages
```

### Performance Testing

For performance validation with real NCCL applications:

```bash
# Set environment
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand
export NCCL_PLUGIN_DIR=/path/to/build

# Run NCCL test
./nccl-tests/build/all_reduce_perf -b 1M -e 1G -f 2 -g 1
```

## Troubleshooting

### Common Issues

#### 1. Kernel Module Won't Load
```bash
# Check kernel version compatibility
uname -r
sudo modinfo tb5_dma_ring.ko

# Check for Thunderbolt support
lspci | grep Thunderbolt
ls /sys/bus/thunderbolt/devices/
```

#### 2. Device Node Not Created
```bash
# Check kernel messages
dmesg | tail -20

# Try manual device creation
sudo mknod /dev/tb5_ol_ring0 c 234 0
sudo chmod 666 /dev/tb5_ol_ring0
```

#### 3. Permission Denied
```bash
# Add user to appropriate groups
sudo usermod -a -G thunderbolt $USER

# Or set device permissions
sudo chmod 666 /dev/tb5_ol_ring0
```

#### 4. Test Failures
```bash
# Run with verbose output
cd build && ./tb5_test

# Check kernel logs
sudo dmesg | grep -i tb5

# Verify Thunderbolt connection
boltctl list
```

### Debug Mode

Enable debug logging:

```bash
# Kernel debug
echo 'module tb5_dma_ring +p' | sudo tee /sys/kernel/debug/dynamic_debug/control

# NCCL debug
export NCCL_DEBUG=INFO
```

## Architecture Details

### NCCL Plugin (`librccl_net_tb5.so`)
- Implements NCCL net v7 API
- Handles connection setup and teardown
- Manages DMA buffer allocation from NCCL
- Routes data through userspace ring interface

### Userspace Ring Library (`libtb5_ring.so`)
- Provides C API for ring operations
- Manages ioctl communication with kernel
- Handles completion notifications
- Validates buffer parameters

### Kernel Driver (`tb5_dma_ring.ko`)
- Character device driver for Thunderbolt ring
- DMA buffer validation and mapping
- Thunderbolt transport layer interface
- Error handling and recovery

### DMA Buffer Handling
- Validates buffer file descriptors
- Checks size and offset bounds
- Maps/unmaps scatter-gather lists
- Provides Thunderbolt transport integration points

## Development

### Code Structure
```
tb5-rccl-project/
├── CMakeLists.txt          # Main build configuration
├── test_main.c            # Test program
├── .gitignore             # Git ignore rules
├── plugin/                # NCCL plugin
│   ├── CMakeLists.txt
│   └── src/
│       └── tb5_plugin.c
├── userspace/             # Userspace ring library
│   ├── CMakeLists.txt
│   ├── include/tb5/
│   │   └── tb5_ring.h
│   └── src/
│       └── tb5_ring.c
├── kernel/                # Kernel module
│   ├── CMakeLists.txt
│   ├── Makefile
│   └── tb5_dma_ring.c
└── third_party/           # External dependencies
    └── nccl/
        └── net_v7.h
```

### Contributing

1. Follow Linux kernel coding style for C code
2. Add tests for new functionality
3. Update documentation
4. Ensure kernel module builds on target kernel versions

### Kernel Compatibility

Tested on:
- Linux 6.14.0-1018-oem
- GCC 13.3.0
- CMake 3.28.3

## License

This project is licensed under the GPL v2.0 License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review kernel logs with `dmesg | grep TB5`
3. Verify Thunderbolt hardware with `boltctl`
4. Ensure proper permissions and kernel module loading
