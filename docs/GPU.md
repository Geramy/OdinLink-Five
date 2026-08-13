# GPU Usage

## RCCL (AMD ROCm) — **supported path: net plugin**

RCCL does **not** discover OdinLink through `NCCL_NET_PLUGIN=IB` / verbs.
It `dlopen`s `libibverbs.so.1` and resolves symbols from that handle, so
`LD_PRELOAD=libodl_tb5_verbs.so` is bypassed. If the net plugin is missing,
RCCL silently falls back to TCP sockets (~4× slower) with no error.

### Build and install the plugin

```bash
cmake --build . --target rccl_net_odl_tb5
# Build dir also has librccl-net.so → librccl_net_odl_tb5.so (RCCL's probe name)
```

```bash
# Option A — plugin directory (RCCL probes librccl-net.so there)
export LD_LIBRARY_PATH=/path/to/build/rccl:/path/to/build/lib:$LD_LIBRARY_PATH
# Optional name pin used by some RCCL builds:
export RCCL_NET_PLUGIN=ODL_TB5
export RCCL_PLUGIN_DIR=/path/to/build/rccl

# Option B — system install (also installs as librccl-net.so)
sudo cmake --install build --component rccl
```

### Confirm you are on the fast path

With `NCCL_DEBUG=INFO` / `RCCL_DEBUG=INFO` you want:

```
NCCL INFO NET/Plugin: Loaded net plugin ODL_TB5 (v7)
NCCL INFO Using network ODL_TB5
```

If you see `Using network Socket` instead, the plugin was not found —
collectives still complete, just slowly. Fix `LD_LIBRARY_PATH` /
`RCCL_PLUGIN_DIR` so `librccl-net.so` is visible.

The RCCL plugin exports shared-memory statistics at `/run/odl_tb5/rccl_stats`.
The daemon reads these and exposes them via D-Bus; the tray app displays
TX/RX bytes, operation counts, and uptime in a dedicated RCCL Stats window.

### `iommu=pt` note (virtualisation hosts)

RCCL may print:

```
NCCL WARN Missing "iommu=pt" from kernel command line ...
```

On single-GPU Strix Halo / Proxmox nodes that warning is often wrong for
OdinLink: `iommu=pt` puts the Thunderbolt NHI in an **identity** IOMMU
domain where the default `odl_ring_size=4096` (16 MB contiguous) fails to
allocate. Prefer translated IOMMU, or load with `odl_ring_size=1024`.

## NCCL (NVIDIA CUDA / PyTorch)

### Option 1: Custom net plugin (recommended for multi-node TB)

```bash
export NCCL_NET_PLUGIN=ODL_TB5
export NCCL_PLUGIN_DIR=/path/to/build/nccl
export NCCL_DEBUG=INFO   # confirm "Using network ODL_TB5"
```

### Option 2: Built-in Verbs / IB transport

NCCL's built-in `IB` transport discovers RDMA devices via
`ibv_get_device_list`. This only works if the OdinLink device is visible
to libibverbs:

- **`LD_PRELOAD=libodl_tb5_verbs.so`** works for tools like `ibv_devinfo`
- The **rdma-core directory plugin** (`libodl_tb5-rdmav34.so`) only loads
  when sysfs already has an unclaimed RDMA device — on OdinLink-only
  hosts it is inert
- Many production stacks `dlopen` libibverbs and ignore `LD_PRELOAD`
  (same footgun as RCCL)

```bash
# x86_64 install path for the directory plugin (if you have other RDMA NICs)
sudo cp build/verbs/libodl_tb5-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/

export NCCL_NET_PLUGIN=IB
export NCCL_IB_HCA=odl_tb5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
```

Prefer Option 1 (net plugin) unless you have verified IB discovery.

### Prerequisites
- NVIDIA GPU with `nvidia-drm` modeset enabled (`nvidia-drm.modeset=1`)
- CUDA 11.7+ for `cuMemGetHandleForAddressRange` (DMA-buf FD export)
- NCCL 2.12+ (supports net plugin v4/v5)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NCCL_NET_PLUGIN=ODL_TB5` | Enables the OdinLink TB5 NCCL plugin |
| `NCCL_PLUGIN_DIR=/path/` | Directory containing `libnccl-net-ODL_TB5.so` |
| `NCCL_DEBUG=INFO` | Enables NCCL debug logging |
| `NCCL_NET_DISABLE=0` | Ensures network transport is not disabled |

### Compression — host LZ4 (ODLC), not nvCOMP

NCCL over Thunderbolt stays uncompressed. nvCOMP is **not wired**:

- NCCL `isend` would still DMA a fixed max size, so GDeflate would not
  shrink the cable transfer.
- A Mac cannot decode nvCOMP GDeflate / batched LZ4.
- The 5090 has no Blackwell decompress engine (that is B200/GB200 only).

What *is* wired is portable **ODLC lz4_block** on the TB-bridge / Mac
path (`bridge/odl_compress.py`, `odl_compress_host`). Tensors ≥ 256 KiB
are compressed unless `ODL_COMPRESS=0`. Ratio is whatever this payload
measures — `odl_tb5_cli client -t compress` prints zeros, the bandwidth
`0xAA` fill, and random.

```bash
cmake --build . --target odl_compress_host_test
./compress/odl_compress_host_test
python3 compress/tests/test_odl_compress.py
odl_tb5_cli client -t compress
```

### How It Works

1. `regMr` registers CUDA memory with the plugin, exporting it as a Linux
   DMA-buf FD via `cuMemGetHandleForAddressRange`
2. `isend`/`irecv` pass the DMA-buf FD to the kernel driver, which programs
   the TB5 NHI DMA engine to transfer directly between GPU and the Thunderbolt link
3. The transfer is zero-copy — GPU memory is read/written directly by the
   Thunderbolt DMA engine, with no CPU involvement

### Limitations

- `isend`/`irecv` are currently synchronous (block until DMA completes).
  True async support requires kernel-side non-blocking DMA-buf submission.
- Multi-buffer `irecv` (v4 API) handles `n=1` in practice. Allocate one
  buffer per NCCL channel for optimal performance.
- CUDA memory registration requires `cuMemGetHandleForAddressRange` (CUDA 11.7+).
  If unavailable, the plugin falls back to staging through host memory.

### PyTorch Distributed Training over TB5

```python
import torch
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',
    init_method='tcp://192.168.1.1:12345',
    world_size=2,
    rank=0  # or 1 on the second machine
)

model = torch.nn.Linear(1000, 1000).cuda()
model = torch.nn.parallel.DistributedDataParallel(model)
```

NCCL automatically uses the OdinLink TB5 plugin via `NCCL_NET_PLUGIN`.

Both RCCL and NCCL plugins export shared-memory statistics at
`/run/odl_tb5/{rccl,nccl}_stats`.
