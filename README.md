# OdinLink-Five

**Thunderbolt 5 RDMA for Linux — kernel driver, libibverbs provider, NCCL/RCCL plugins**

OdinLink turns a Thunderbolt cable into a high-speed RDMA interconnect between machines. It provides the full `ibv_*` verbs API so any verbs-aware application (NCCL, MPI, PyTorch DDP) can use Thunderbolt DMA without code changes.

```
80 Gbps  ·  sub-µs latency  ·  zero-copy GPU  ·  standard ibv_verbs API
```

---

## Progress

| Layer | Component | Status |
|-------|-----------|--------|
| 🟢 | Kernel module (`odl_tb5.ko`) | NHI ring DMA, XDomain handshake, loopback mode |
| 🟢 | Userspace library (`libodl_tb5.so`) | C API, stream I/O, mmap, DMA-buf |
| 🟢 | Verbs provider (`libodl_tb5_verbs.so`) | `ibv_open_device`, `ibv_reg_dmabuf_mr`, QP/CQ lifecycle |
| 🟢 | rdma-core plugin (`libodl_tb5-rdmav34.so`) | Auto-discovered by `ibv_devinfo` |
| 🟢 | Async I/O | `poll()` + `O_NONBLOCK` ioctls end-to-end |
| 🟢 | No-cable testing | `loopback=1` module param + mock library |
| 🟡 | NCCL plugin | Custom API — needs verbs provider integration |
| 🔴 | NCCL verbs transport | Built-in NCCL verbs transport auto-discovers ODL |
| 🔴 | Async DMA-buf | Non-blocking GPU memory registration |

## Quick Start

### Build & Run

```bash
sudo apt install build-essential cmake linux-headers-$(uname -r) libibverbs-dev rdma-core pkg-config gcc-14
git clone https://github.com/johndpope/OdinLink-Five.git
cd OdinLink-Five && mkdir build && cd build
cmake .. -DBUILD_VERBS=ON && make -j$(nproc) odl_tb5_verbs odl_tb5_verbs_provider

# Test without cable:
sudo insmod driver/odl_tb5.ko loopback=1
ibv_devinfo                     # Should show odl_tb5 device
build/verbs/tests/test_verbs_basic
```

### Point-to-Point (two machines)

```bash
# Machine A:
sudo insmod driver/odl_tb5.ko
build/cli/odl_tb5_cli --server --device 0

# Machine B:
sudo insmod driver/odl_tb5.ko
build/cli/odl_tb5_cli --client --device 0 --test bandwidth
```

Full install guide → [`docs/INSTALL.md`](docs/INSTALL.md)

## Architecture

```
┌────────────────────────────────────────────────────┐
│  Application (NCCL, MPI, PyTorch, ibv_* API)       │
├────────────────────────────────────────────────────┤
│             libibverbs (libibverbs.so.1)            │
├────────────────────────────────────────────────────┤
│  libodl_tb5-rdmav34.so  (verbs provider plugin)    │
├────────────────────────────────────────────────────┤
│  libodl_tb5.so  (OdinLink C API)                   │
├────────────────────────────────────────────────────┤
│  odl_tb5.ko  (kernel module — NHI DMA)             │
├────────────────────────────────────────────────────┤
│  Thunderbolt 5 NHI DMA Engine                       │
└────────────────────────────────────────────────────┘
```

## Components

| Component | Binary | Description |
|-----------|--------|-------------|
| Kernel driver | `odl_tb5.ko` | NHI ring DMA, XDomain handshake, char device |
| Library | `libodl_tb5.so` | C API wrapping ioctls, streams, mmap |
| Verbs provider | `libodl_tb5_verbs.so` | Standalone `ibv_*` via symbol interposition |
| Verbs plugin | `libodl_tb5-rdmav34.so` | rdma-core provider plugin (`ibv_devinfo`) |
| NCCL plugin | `libnccl-net-ODL_TB5.so` | NVIDIA GPU collectives |
| RCCL plugin | `librccl_net_odl_tb5.so` | AMD GPU collectives |
| CLI tool | `odl_tb5_cli` | Bandwidth, latency, jitter, MIMO tests |
| Loopback module | `loopback=1` param | Fake peer for no-cable testing |
| Mock library | `libodl_tb5_mock.so` | LD_PRELOAD simulation (no kernel needed) |

### GPU and daemon/tray → [`docs/GPU.md`](docs/GPU.md), [`docs/INSTALL.md`](docs/INSTALL.md)

## Verbs API Coverage

| Operation | Status | Notes |
|-----------|--------|-------|
| `ibv_open_device` | ✅ | Symbol interposition + rdma-core plugin |
| `ibv_query_device` | ✅ | Attributes from peer info |
| `ibv_query_port` | ✅ | Port state from peer connection |
| `ibv_alloc_pd` / `ibv_dealloc_pd` | ✅ | Protection domains |
| `ibv_reg_mr` / `ibv_dereg_mr` | ✅ | Host memory registration |
| `ibv_reg_dmabuf_mr` | ✅ | Zero-copy GPU memory (Linux DMA-buf) |
| `ibv_create_cq` / `ibv_destroy_cq` | ✅ | Eventfd-based completion queues |
| `ibv_poll_cq` / `ibv_req_notify_cq` | ✅ | Poll + eventfd notification |
| `ibv_create_qp` / `ibv_destroy_qp` | ✅ | RC QP → stream mapping |
| `ibv_modify_qp` | ✅ | RESET → INIT → RTR → RTS |
| `ibv_post_send` | ✅ | Async via workqueue + poll() |
| `ibv_post_recv` | ✅ | Non-blocking via poll() |
| `ibv_query_qp` | ✅ | State + capabilities |

## Async I/O Model

```
ibv_post_send(qp, wr, NULL)
    │
    ▼  (non-blocking, returns immediately)
Enqueue WR → per-QP submission queue
    │
    ▼  (worker thread)
poll(fd, POLLOUT)  ← kernel signals TX readiness
    │
    ▼
ioctl(STREAM_SEND) ← O_NONBLOCK, never blocks
    │
    ├── -EAGAIN → re-queue WR, poll again
    └── success → post struct ibv_wc → CQ → eventfd
```

## Testing Without Hardware

```bash
# Option 1: kernel loopback (real module, fake peer)
sudo insmod driver/odl_tb5.ko loopback=1
build/verbs/tests/test_verbs_basic

# Option 2: user-space mock (no kernel module at all)
mkfifo /dev/odl_tb5_0
LD_PRELOAD=verbs/tests/libodl_tb5_mock.so \
  LD_LIBRARY_PATH=build/verbs:build/lib \
  build/verbs/tests/test_verbs_mock_loopback
```

## Debug

```bash
export ODL_VERBS_DEBUG=5     # Trace all verbs calls
sudo dmesg -w | grep odl_tb5 # Kernel driver logs
```

Troubleshooting → [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)

## Cross-Platform: macOS

Apple ships `libthunderboltrdma.dylib` + `libibverbs` on macOS 26.5, but
the kernel extension is a stub — `IORDMAFamily` is not shipped. Mac
Thunderbolt RDMA is not currently functional. OdinLink is the only
working implementation. See [`COMPAT.md`](COMPAT.md).

## Repository

| Resource | Link |
|----------|------|
| Install guide | [`docs/INSTALL.md`](docs/INSTALL.md) |
| GPU / NCCL / RCCL | [`docs/GPU.md`](docs/GPU.md) |
| Troubleshooting | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| Packaging / .deb | [`docs/PACKAGING.md`](docs/PACKAGING.md) |
| Agent instructions | [`AGENTS.md`](AGENTS.md) |
| Verbs provider manual | [`verbs/VERBS_PROVIDER.md`](verbs/VERBS_PROVIDER.md) |
| Cross-platform compat | [`COMPAT.md`](COMPAT.md) |

## License

- **Kernel driver** (`odl_tb5.ko`): GPL v2
- **All userspace**: MIT
