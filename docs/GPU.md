# GPU Usage

## RCCL (AMD ROCm)

```bash
export RCCL_NET_PLUGIN=ODL_TB5
export RCCL_PLUGIN_DIR=/path/to/build/rccl

# Your RCCL/ROCm application will use TB5 automatically
```

The RCCL plugin exports shared-memory statistics at `/run/odl_tb5/rccl_stats`.
The daemon reads these and exposes them via D-Bus; the tray app displays
TX/RX bytes, operation counts, and uptime in a dedicated RCCL Stats window.

## NCCL (NVIDIA CUDA / PyTorch)

The NCCL network plugin enables zero-copy GPU-to-GPU transfers over
Thunderbolt 5 for NVIDIA GPUs. It uses the Linux DMA-buf infrastructure
to transfer CUDA memory directly through the TB5 NHI DMA engine, bypassing
the CPU.

### Prerequisites
- NVIDIA GPU with `nvidia-drm` modeset enabled (`nvidia-drm.modeset=1`)
- CUDA 11.7+ for `cuMemGetHandleForAddressRange` (DMA-buf FD export)
- NCCL 2.12+ (supports net plugin v4/v5)

### Usage

```bash
export NCCL_NET_PLUGIN=ODL_TB5
export NCCL_PLUGIN_DIR=/path/to/build/nccl

# Or specify the full path:
export NCCL_NET_PLUGIN=/path/to/build/nccl/libnccl-net-ODL_TB5.so

# Run your NCCL application (e.g., PyTorch distributed training)
torchrun --nproc_per_node=1 --nnodes=2 \
    --node_rank=0 --master_addr=192.168.1.1 --master_port=12345 \
    your_training_script.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NCCL_NET_PLUGIN=ODL_TB5` | Enables the OdinLink TB5 NCCL plugin |
| `NCCL_PLUGIN_DIR=/path/` | Directory containing `libnccl-net-ODL_TB5.so` |
| `NCCL_DEBUG=INFO` | Enables NCCL debug logging |
| `NCCL_NET_DISABLE=0` | Ensures network transport is not disabled |

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
