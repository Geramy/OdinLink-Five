# Remote Tensors — design sketch for cross-platform VRAM sharing

> **Status**: design doc / architecture sketch. The userspace bridge in
> [`bridge/`](../bridge/) is the working interim solution. The
> RDMA-backed implementation depends on
> [Mac protocol capture work](MAC_PROTOCOL_CAPTURE.md) landing first.

## What problem this solves

You're training on a Linux + NVIDIA GPU with 32 GB of VRAM. You have a
Mac with 96–192 GB of unified memory sitting on the desk, connected via
Thunderbolt 5. You want to *treat the Mac's unified memory as an
extension of the training GPU's VRAM* — push spilled activations,
attention K/V cache, optimizer state shards across the cable, and pull
them back when needed.

Three things stand between you and that:

1. **No NCCL on Mac.** NVIDIA's collective library is CUDA-only. AMD's
   RCCL is ROCm-only. Apple has no equivalent. You can't drop the Mac
   into a `torch.distributed` process group as a peer.
2. **No PCI BAR for Apple GPU.** Apple Silicon GPUs use unified memory.
   They don't expose addressable memory regions for outside peers to
   DMA into. The "GPUDirect RDMA" trick that lets one NVIDIA GPU read
   another's VRAM over InfiniBand has no Apple counterpart.
3. **Different verbs implementations.** Apple ships its own RDMA stack
   (`libthunderboltrdma.dylib` + `AppleThunderboltRDMA.kext`). OdinLink
   ships its own. The XDomain handshake formats don't match yet —
   that's what the [protocol capture work](MAC_PROTOCOL_CAPTURE.md)
   fixes.

So the right abstraction is **not** "make the Mac a CUDA peer." It's:
*give the user one Python API that hides whether the remote tensor lives
on a CUDA device or in Apple unified memory, and let it use the best
available transport.*

## The API we want

```python
import torch
from odinlink.remote import RemoteStore

# Connect to a remote host's memory (TB5 cable to Mac, or another Linux
# box over RDMA — same API)
mac = RemoteStore("tb5://mac-tb-ip:29800", capacity="96GB")

# Push a CUDA tensor across the wire. The implementation picks the
# best transport: GPUDirect-RDMA if peer is CUDA, host-copy+TCP if peer
# is Apple, etc.
x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda:0")
handle = mac.put("kv_cache.layer.42", x)

# Later — pull it back. If we never modified it on the remote side, the
# implementation may stream lazily.
y = mac.get("kv_cache.layer.42", device="cuda:0")

# Or: ask the remote side to compute on it
result = mac.run("softmax", "kv_cache.layer.42", dim=-1)

# Bulk transfers: prefetch + overlap with compute
with mac.prefetch_window(["layer.39", "layer.40", "layer.41"]):
    train_step(...)
```

## Transport layers (selected at runtime)

| Layer | Used when | Bandwidth | Latency | Status |
|---|---|---|---|---|
| `cuda-cuda-rdma` (NCCL/GPUDirect) | Both peers are CUDA, IB/RoCE/TB5-RDMA link | 100+ Gbps | ~1 µs | shipped in OdinLink |
| `tb5-rdma-host-copy` | Linux-CUDA ↔ Mac-Metal, OdinLink driver, host-RAM staging | 50–60 Gbps | ~5 µs | **blocked on Mac protocol** |
| `tb5-ip` (the [bridge/](../bridge/)) | Anything-to-anything over Thunderbolt-IP | 25–40 Gbps | ~100 µs | shipped — Python-only |
| `tcp-ethernet` | Fallback over normal LAN | 1–10 Gbps | ~200 µs | trivial |

The transport selection is keyed off **peer device class** and
**connectivity**:

```
peer = mac           ──► [tb5-rdma-host-copy] if Apple RDMA + cable up
                    ──► [tb5-ip]              if Thunderbolt-IP up
                    ──► [tcp-ethernet]        always
peer = linux-cuda    ──► [cuda-cuda-rdma]     if OdinLink + cable up
                    ──► [tcp-ethernet]        otherwise
```

## Memory layout assumptions

Each `RemoteStore` instance owns a remote address space. Keys are
strings (≤ 256 B UTF-8). Values are tensors with serialised metadata
(dtype, shape, layout, device-class, optional KV-cache geometry hints).

The remote side decides where the bytes actually live:
- On a CUDA peer: by default in pinned host RAM, but the peer can elect
  to keep hot keys in GPU VRAM via an LRU policy
- On an Apple peer: in unified memory, which is *simultaneously* CPU-
  and GPU-addressable on Apple Silicon. The remote daemon can answer
  "compute X on key K" using Metal without copying

The client gets a `Handle` (lightweight), not the bytes. Bytes only
move on `get(...)` or `run(...)`. This is essential for offload
patterns where 95% of operations are "stash this, fetch it back later"
and only 5% need actual computation.

## What we ship vs. what's deferred

| Component | Now | Later |
|---|---|---|
| Userspace bridge (TB5-IP transport) | ✅ working ([`bridge/`](../bridge/)) — one TCP connection, [`mlx_helpers.py`](../bridge/mlx_helpers.py) wrap | compression / encryption |
| Mac↔Linux RDMA verbs handshake | partial (`protocol=1` lenient response) | `protocol=2` (Apple-format outbound) blocked on [capture](MAC_PROTOCOL_CAPTURE.md) |
| Linux↔Linux CUDA-CUDA over TB5 | ✅ working (NCCL plugin) | — |
| Unified `RemoteStore` Python API | ✅ shipped on `tb5-ip` ([`odinlink/remote.py`](../odinlink/remote.py)) | DMA transport once Mac RX shows `rx_done` |
| Tensor-aware compute on remote (`mac.run("softmax", ...)`) | future | requires Metal/CUDA-side compute daemons |
| Prefetch window / async streaming | ✅ IP path (`prefetch_window`) | RDMA streaming |

## Why this beats "buy a bigger GPU"

The 96–192 GB on a Mac Studio is real memory you can use *today* for
offload. At TB5 speeds, the round-trip cost of fetching a 256 MB tensor
back from the Mac is ~50–100 ms. For training patterns where you spill
activations between transformer blocks and bring them back during
backward, that's tolerable if you overlap with compute. It would not
substitute for local VRAM during the forward path, but it expands the
*reachable working set* dramatically without buying a new GPU.

The alternative — renting an 80 GB H100 — is faster per step but costs
$2–4/hr indefinitely. A one-time Mac purchase gets paid back in <1000
hours of training time. Whether that's a good trade depends on how
often you train.

## Risks / open questions

1. **Apple GPU is not directly addressable from outside.** Even with
   perfect Mac↔Linux RDMA, "Mac VRAM" means Apple unified memory
   reachable from Metal. The Linux GPU never reads it directly; the
   bytes always staging through Mac host RAM. This is a hard constraint
   of Apple's architecture, not a software issue.
2. **TB5 cable quality matters.** Apple's TN3205 requires "active TB5
   cables for sustained RDMA traffic." Cheap passive cables will
   negotiate down to TB3 speeds and the RDMA layer will refuse.
3. **macOS RDMA is new (26.2+).** Apple has not yet committed to API
   stability. A point release could change the kext's behavior.
4. **No model-parallel for Mac**: even with this layer working, the
   Mac can't be a *training* peer — it can only be an offload target.
   The student model lives entirely on the Linux GPU.

## References

- [`bridge/README.md`](../bridge/README.md) — userspace transport docs
- [`MAC_PROTOCOL_CAPTURE.md`](MAC_PROTOCOL_CAPTURE.md) — unblocking the RDMA path
- [`../COMPAT.md`](../COMPAT.md) — Mac↔Linux compatibility matrix and protocol notes
- Apple TN3205, "Low-latency communication with RDMA over Thunderbolt"
- d3LLM training write-up (sister project) — motivating use case for
  bigger effective VRAM during 27B-class fine-tuning
