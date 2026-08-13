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

### Optional GPU compression (nvCOMP) — `odin-compress`

Bandwidth on TB is the bottleneck vs VRAM. When **both peers** enable it, large
CUDA messages are compressed with **nvCOMP** (GDeflate/LZ4/Snappy) before RDMA
and decompressed on receive (Blackwell uses the hardware DE when available).

**Build is optional.** If nvCOMP is not installed, CMake still succeeds and
links a **stub**: `odl_compress_enabled()` is always 0 and behaviour is
unchanged. No user without nvCOMP is broken.

```bash
# AUTO (default): enable backend only if headers+lib found
cmake .. -DODL_ENABLE_NVCOMP=AUTO

# Force off even if nvCOMP is on the machine
cmake .. -DODL_ENABLE_NVCOMP=OFF

# Force on — warns and stubs if missing (does not fail configure)
cmake .. -DODL_ENABLE_NVCOMP=ON -DNVCOMP_ROOT=/path/to/nvcomp
```

**Optional install (Ubuntu):**

```bash
wget https://developer.download.nvidia.com/compute/nvcomp/5.3.0/local_installers/nvcomp-local-repo-ubuntu2604-5.3.0_5.3.0-1_amd64.deb
sudo dpkg -i nvcomp-local-repo-ubuntu2604-5.3.0_5.3.0-1_amd64.deb
sudo cp /var/nvcomp-local-repo-ubuntu2604-5.3.0/nvcomp-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install nvcomp
```

Or pip wheel + hint:

```bash
pip install nvidia-nvcomp-cu12
cmake .. -DNVCOMP_ROOT=$HOME/miniconda3/lib/python3.13/site-packages/nvidia/libnvcomp
```

**Runtime (both sides):**

| Variable | Default | Meaning |
|----------|---------|---------|
| `ODL_COMPRESS` | NCCL: off · bridge: on | `1` / `true` enables; `0` disables. Bridge defaults on. |
| `ODL_COMPRESS_ALGO` | `gdeflate` | `gdeflate` \| `lz4` \| `snappy` \| `lz4_block` |
| `ODL_COMPRESS_THRESHOLD` | `262144` | Min message size (bytes) |
| `ODL_COMPRESS_LEVEL` | `1` | Reserved for future |

`gdeflate` / `lz4` / `snappy` are **nvCOMP native**. A Mac cannot decode
them. They are Linux↔Linux NCCL only.

**What NVIDIA actually publishes** (not a number we invented):

| Codec | Official claim | In this tree |
|-------|----------------|--------------|
| Cascaded | up to **80×** on analytical numerical data, up to 500 GB/s | **Not used** |
| LZ4 / Snappy | up to **100 GB/s** GPU throughput; no official ratio | nvCOMP algo 2/3, Linux GPU only |
| GDeflate | GPU format; **no published ratio** | NCCL default when `ODL_COMPRESS=1` |
| Blackwell DE | up to **600 GB/s decompress** | only if you have Blackwell |

Ratio is payload-dependent. The CLI prints **measured** in/wire for zeros, the same `0xAA` fill as the bandwidth test, and random (`odl_tb5_cli -t compress`). That run needs no cable and works on a Mac.

`lz4_block` is the portable payload (64 KiB standard LZ4 raw blocks + a
chunk table). The TB-bridge and the Mac always use this. See
[`compress/include/odl_tb5/odl_compress.h`](../compress/include/odl_tb5/odl_compress.h).

NCCL `isend` still DMAs `max_wire` bytes so both ranks agree on size —
that path does **not** shrink the Thunderbolt transfer. The bridge does:
`data_len` on the TCP header is the compressed size.

Host / Mac smoke test (no CUDA):

```bash
cmake --build . --target odl_compress_host_test
./compress/odl_compress_host_test
python3 compress/tests/test_odl_compress.py
```

```bash
export ODL_COMPRESS=1
export ODL_COMPRESS_ALGO=gdeflate
export ODL_COMPRESS_THRESHOLD=262144
# then normal NCCL_NET_PLUGIN=ODL_TB5 launch
```

**Smoke test** (only built when nvCOMP was found):

```bash
cmake --build . --target odl_compress_bench
ODL_COMPRESS=1 ./compress/odl_compress_bench 4194304 20
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
