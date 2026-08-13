# Ubuntu + PyTorch + Mac memory

Yes — as **offload RAM**, not as a CUDA device.

The Mac cannot join `torch.distributed` as an NCCL rank. Apple unified
memory has no PCI BAR for GPUDirect. Official Mac Thunderbolt RDMA is
Mac↔Mac and currently a stub.

What works today is: train on Ubuntu + CUDA, **park tensors on the Mac**
over the Thunderbolt cable.

```
Ubuntu (CUDA VRAM)  --TB5 IP copies-->  Mac unified memory
     PyTorch                bridge/           tb_bridge_server
```

~25–40 Gb/s, ~50–200 µs. Fine for spilled activations, KV cache,
optimizer shards. Not zero-copy into Metal.

## Cable and network

1. TB5-rated cable (TB3/TB4 often will not negotiate TB5).
2. On the Mac: System Settings → Thunderbolt Bridge (or `ifconfig` for
   a `bridge` / Thunderbolt IP).
3. On Linux: `ip link` should show `thunderbolt0` / `en05` once
   ThunderboltIP comes up. If you only see `ThunderboltIP login timed
   out`, this path is down — fix that before blaming PyTorch.

Hardware cheat-sheet: [`HARDWARE.md`](HARDWARE.md) ·
`scripts/tb-hw-check.sh`

## Run it

**Mac**

```bash
python3 bridge/tb_bridge_server.py --bind 0.0.0.0 --max-gb 96 -v
```

**Ubuntu trainer**

```bash
# optional: pip install torch  (already have it if you train)
python3 bridge/benchmark.py --host <mac-tb-ip> --cuda

# from the repo root
python3 examples/pytorch_mac_offload.py --url tb5://<mac-tb-ip>
```

```python
from odinlink import RemoteStore
import torch

mac = RemoteStore("tb5://<mac-tb-ip>")
x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
mac.put("kv.layer.42", x)                 # GPU → Mac RAM
y = mac.get("kv.layer.42", device="cuda") # Mac RAM → GPU
```

The copy is GPU → host → Thunderbolt IP → Mac, and back the same way.

## What is *not* ready

| Want | Status |
|------|--------|
| `NCCL_NET_PLUGIN=ODL_TB5` between two Linux GPUs | Works — see [`GPU.md`](GPU.md) |
| Mac as an NCCL/RCCL rank | No |
| PyTorch DMA straight into Mac pages | Kext + `bind_any` / `skip_login` are the start; login/RX not proven |
| Treat Mac RAM as `device="cuda:1"` | No. Use explicit `put` / `get` keys |

## Next (DMA, faster)

Linux `insmod odl_tb5.ko skip_login=1` plus the Mac kext (`mac/README.md`)
is the DMA experiment. Do not point PyTorch at that until dmesg shows
`entering READY state` and the Mac client sees `rx_done` climb. Until
then, use the bridge.
