# TB-Bridge — userspace tensor offload over Thunderbolt IP

A pure-Python client/server that lets a Linux CUDA training host offload
tensors to a Mac (or any peer) across a Thunderbolt 5 cable, using the
ordinary IP-over-Thunderbolt stack Apple already ships.

## What this is — and isn't

**It is**: a working userspace bridge usable today. ~25–40 Gbps over TB5
IP, ~50–200 µs round-trip latency. Suitable as an offload tier for spilled
tensors when the training GPU's VRAM isn't big enough.

**It isn't**: GPU-direct RDMA. Data goes through host RAM on both ends.
For zero-copy GPU-to-GPU over Thunderbolt you need OdinLink's kernel
driver path (Linux↔Linux working today, Mac interop work-in-progress;
see [`../docs/MAC_PROTOCOL_CAPTURE.md`](../docs/MAC_PROTOCOL_CAPTURE.md)
and [`../docs/REMOTE_TENSORS.md`](../docs/REMOTE_TENSORS.md)).

## Quick start

### On the Mac (or remote host)

```bash
# Find your TB-net IP — Apple assigns one when the cable is connected:
ifconfig | grep -A 3 bridge   # look for an "inet" line under a TB-named iface
# Or check System Settings → Network → "Thunderbolt Bridge"

python3 bridge/tb_bridge_server.py --bind 0.0.0.0 --max-gb 96 -v
```

### On the Linux training host

```bash
# Quick benchmark
python3 bridge/benchmark.py --host <mac-tb-ip> --cuda

# Programmatic use
python3 -c '
import torch
from bridge.tb_bridge_client import TBBridgeClient

cli = TBBridgeClient("10.0.0.2")          # mac TB-net IP
x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
cli.put("layer.42.attn", x)               # offload to mac
y = cli.get("layer.42.attn", device="cuda")  # fetch back
assert torch.equal(x, y)
print("round-trip OK")
'
```

## Wire protocol

Length-prefixed binary over TCP. One connection is reused for many
requests until the client hangs up. All integers big-endian.

```
Request:
  u8   op           PUT=1, GET=2, DEL=3, LIST=4, STAT=5
  u32  key_len
  u8[] key          (≤256 B utf-8)
  PUT only:
    u32  meta_len
    u8[] meta_json  (dtype, shape, kind: numpy|torch)
    u64  data_len
    u8[] data       (raw tensor bytes, or ODLC lz4_block if
                    meta.odlc is true / data starts with ODLC magic)

Response:
  u8   status       0=OK, 1=NOT_FOUND, 2=BAD_OP, 3=OOM, 4=PROTOCOL
  GET only (status=OK):
    u32  meta_len
    u8[] meta_json
    u64  data_len
    u8[] data
  LIST only (status=OK):
    u32  n
    n × ( u32 keylen + u8[] key )
  STAT only (status=OK):
    u64  total_bytes
    u32  num_keys
```

## bfloat16 caveat

NumPy has no native bf16. The client carries bf16 tensors as `uint16` on
the wire and rehydrates them with `tensor.view(torch.bfloat16)` on the
receive side. Lossless.

## Performance ceiling

Theoretical TB5 IP layer: 80 Gbps raw → ~10 GB/s payload after framing.
Real-world over Apple's bridge: 25–40 Gbps (~3–5 GB/s). Latency floor:
the TCP stack + scheduler costs ~50 µs each direction. So this is best
for tensors ≥ 16 MB; smaller payloads spend most of their time in
syscall overhead.

For comparison: a single PCIe Gen4 x16 link is ~64 GB/s, so the bridge
is roughly 5–15× slower than local VRAM movement. The win is *capacity*,
not bandwidth: it lets you treat the Mac's 96–192 GB unified memory as
an extension of the training host's spill pool.

## What's missing

- **Encryption**: zero. Run it on a private link only.

Connection reuse, ODLC LZ4, and the Mac-side MLX wrap are in:
`TBBridgeClient` keeps one TCP socket; tensors ≥ 256 KiB are ODLC
lz4_block (`ODL_COMPRESS=0` to disable); `bridge/mlx_helpers.py` turns a
stored blob into `mlx.array` (`copy=False` when mlx allows it) or a
NumPy view. `TensorStore.view(key)` decompresses then wraps.

Local check (no cable): `python3 tests/test_odinlink_remote.py`
and `python3 compress/tests/test_odl_compress.py`
