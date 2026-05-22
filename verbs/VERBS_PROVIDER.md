# OdinLink Verbs Provider — Technical Manual

## Overview

The OdinLink Verbs Provider (`libodl_tb5_verbs.so`) is a **libibverbs-compatible RDMA plugin** that exposes OdinLink-Five Thunderbolt 5 DMA transports through the standard Verbs API (`ibv_*`). It makes any verbs-aware application (NCCL, MPI, PyTorch DDP) use Thunderbolt 5 DMA without code changes.

```
┌─────────────────────────────────────────────────────┐
│                  Application                         │
│  (NCCL, MPI, PyTorch, ibv_* API)                    │
├─────────────────────────────────────────────────────┤
│              libibverbs (libibverbs.so.1)            │
├─────────────────────────────────────────────────────┤
│          libodl_tb5_verbs.so (symbol interposition) │
├─────────────────────────────────────────────────────┤
│       libodl_tb5.so (OdinLink-Five API)             │
├─────────────────────────────────────────────────────┤
│          Kernel: odl_tb5.ko (char device)            │
├─────────────────────────────────────────────────────┤
│          Thunderbolt 5 NHI DMA Engine                 │
└─────────────────────────────────────────────────────┘
```

## Operating Modes

### 1. Standalone Symbol Interposition (default)

The library provides its own `ibv_open_device()` that intercepts calls at link time. For OdinLink-Five devices, it creates a context backed by the real `odl_tb5` char device. For all other devices, it forwards to the real libibverbs.

**Usage:**
```bash
# Link directly
gcc -o my_app my_app.c -lodl_tb5_verbs -libverbs

# Or LD_PRELOAD for existing binaries
LD_PRELOAD=libodl_tb5_verbs.so mpirun --hostfile hosts ./my_mpi_app
```

### 2. rdma-core Provider Plugin

Built as `libodl_tb5-rdmav34.so` and installed into the libibverbs provider directory. Discovered automatically by `ibv_devinfo` and `ibv_open_device`.

**Install:**
```bash
sudo cp build/verbs/libodl_tb5-rdmav34.so /usr/lib/aarch64-linux-gnu/libibverbs/
ibv_devinfo  # Should show odl_tb5 devices
```

### 3. Hardware Simulation / Mock

A user-space mock library simulates two TB5 peers connected via shared memory. No kernel module or Thunderbolt cable required.

**Usage:**
```bash
# Create a fake /dev/odl_tb5_0 for the verbs provider to discover
mkfifo /dev/odl_tb5_0

# Run with mock intercepting hardware calls
LD_PRELOAD=libodl_tb5_mock.so \
LD_LIBRARY_PATH=build/verbs:build/lib \
./my_verbs_app
```

## API Mapping

| Verbs API | OdinLink-Five Mapping | Zero-Copy? |
|-----------|----------------------|------------|
| `ibv_open_device` | `odl_tb5_open` | N/A |
| `ibv_close_device` | `odl_tb5_close` | N/A |
| `ibv_alloc_pd` | lightweight struct alloc | N/A |
| `ibv_reg_mr` | host memory pinning | ❌ (memcpy) |
| `ibv_reg_dmabuf_mr` | DMA-buf fd passthrough | ✅ (GPU zero-copy) |
| `ibv_create_cq` | completion ring + eventfd | N/A |
| `ibv_create_qp` | `odl_tb5_stream_open` | N/A |
| `ibv_post_send` | enqueue → worker → `stream_send` | ✅ (async) |
| `ibv_post_recv` | `stream_recv` (blocking) | ❌ |
| `ibv_poll_cq` | dequeue from eventfd ring | N/A |
| `ibv_modify_qp` | stream state tracking | N/A |

## Async Completion Model

```
ibv_post_send(qp, wr, NULL)
    │
    ▼
enqueue wr → per-QP SQ ring     ← returns immediately (non-blocking)
    │
    ▼
worker thread detects new wr
    │
    ├── dmabuf MR? → odl_tb5_stream_send_dmabuf()  ← zero-copy GPU
    └── host MR?   → odl_tb5_stream_send()         ← kernel copies data
    │
    ▼
post struct ibv_wc → CQ ring
    │
    ├── eventfd_write()     ← wakes ibv_get_cq_event()
    └── ibv_poll_cq()       ← drains from CQ ring
```

## Device Discovery

Devices are discovered by scanning `/dev/odl_tb5_N` entries. Each entry becomes an `ibv_device` that can be opened with `ibv_open_device`.

```c
#include <odl_tb5/odl_tb5_verbs_wrapper.h>

int ndev = odl_num_tb5_devices();
struct ibv_device *dev = odl_find_tb5_device(0);
struct ibv_context *ctx = ibv_open_device(dev);
```

## Debugging

Set `ODL_VERBS_DEBUG` environment variable:

| Level | Output |
|-------|--------|
| 0 | Off (default) |
| 1 | Errors only |
| 2 | + Warnings |
| 3 | + Info (device opens, etc.) |
| 4 | + Verbose (send/recv ops) |
| 5 | + Trace (all function entry/exit) |

```bash
ODL_VERBS_DEBUG=5 LD_PRELOAD=libodl_tb5_verbs.so ./my_app
```

## Zero-Copy GPU Memory

GPUDirect RDMA is achieved through `ibv_reg_dmabuf_mr()`:

```c
// 1. Export GPU memory as a dmabuf file descriptor
int dmabuf_fd = export_cuda_dmabuf(gpu_ptr, size);

// 2. Register with verbs — same API as Apple's ibv_reg_dmabuf_mr
struct ibv_mr *mr = ibv_reg_dmabuf_mr(pd, 0, size, 0, dmabuf_fd,
                                       IBV_ACCESS_LOCAL_WRITE);

// 3. Post send — dmabuf fd passed to kernel driver
struct ibv_send_wr wr = {
    .sg_list = &(struct ibv_sge){.lkey = mr->lkey, .length = size},
    .num_sge = 1,
    .opcode = IBV_WR_SEND,
};
ibv_post_send(qp, &wr, NULL);
```

## NCCL Integration

NCCL's built-in Verbs transport discovers the OdinLink-Five provider automatically when the provider plugin is installed:

```bash
# Install provider
sudo cp build/verbs/libodl_tb5-rdmav34.so /usr/lib/aarch64-linux-gnu/libibverbs/

# NCCL uses it automatically via verbs
NCCL_DEBUG=INFO torchrun --nproc_per_node=1 --nnodes=2 \
    --node_rank=0 --master_addr=192.168.1.1 --master_port=12345 \
    train.py
```

## Linux ↔ macOS Compatibility

The verbs provider on Linux creates the same `ibv_*` API surface that Apple's `libthunderboltrdma.dylib` provides on macOS. If the NHI DMA ring protocol is wire-compatible, the same application code runs on both platforms without changes.

| Platform | Provider | Kernel Driver |
|----------|----------|---------------|
| Linux | `libodl_tb5-rdmav34.so` | `odl_tb5.ko` |
| macOS | `libthunderboltrdma.dylib` | `AppleThunderboltRDMA.kext` |

## Build

```bash
mkdir build && cd build
cmake .. -DBUILD_VERBS=ON
make -j$(nproc) odl_tb5_verbs
```

## Test

```bash
# With mock (no hardware required)
mkfifo /dev/odl_tb5_0
LD_PRELOAD=libodl_tb5_mock.so \
LD_LIBRARY_PATH=build/verbs:build/lib \
build/verbs/tests/test_verbs_mock_loopback

# With real hardware
sudo insmod driver/odl_tb5.ko
build/verbs/tests/test_verbs_basic
```

## File Layout

```
verbs/
├── CMakeLists.txt                    # Build configuration
├── VERBS_PROVIDER.md                 # This file
├── src/
│   ├── odl_tb5_verbs.h               # Internal header
│   ├── odl_tb5_verbs_debug.h         # Debug logging + assertions
│   ├── odl_tb5_verbs_main.c          # Device scan + ibv_open_device interposition
│   ├── odl_tb5_verbs_device.c        # Context lifecycle + queries
│   ├── odl_tb5_verbs_pd.c            # Protection domains
│   ├── odl_tb5_verbs_mr.c            # Memory regions (+ dmabuf)
│   ├── odl_tb5_verbs_cq.c            # Completion queues + eventfd
│   ├── odl_tb5_verbs_qp.c            # Queue pairs + async workqueue
│   ├── odl_tb5_verbs_ops.c           # ibv_context ops dispatch table
│   └── odl_tb5_verbs_wrapper.h       # Public wrapper API header
├── tests/
│   ├── test_verbs_basic.c            # Hardware smoke test
│   ├── test_verbs_mock_loopback.c    # Mock loopback test
│   └── odl_tb5_verbs_mock.c          # Mock library (LD_PRELOAD)
```
