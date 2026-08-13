# Linux → Mac memory over Thunderbolt 5

DMA a Linux box into Mac unified memory. Apple’s own RDMA stack is
Mac↔Mac only and is a stub on current macOS. This path **owns both
ends**: OdinLink on Linux, this kext on the Mac.

```
Linux PC (Intel NHI)                         MacBook (Apple ACIO + DART)
odl_tb5.ko  ── XDomain login + 4 KB frames ──►  OdinLinkRDMA.kext
odl_tensor_send                                mmap’d buffer → your app
```

Two-sided send/recv, not one-sided RDMA WRITE. Linux posts TX frames.
The kext posts RX descriptors of the same 4 KB size. Hardware copies
into a DART-mapped buffer. Userspace maps that buffer — no extra copy.

## Status

| Piece | Status |
|-------|--------|
| Linux driver + stream sender | Builds. Works today Linux↔Linux. Linux→Mac waits on Mac RX. |
| Mac kext: DART buffer + mmap | Implemented |
| Mac kext: watch XDomain for OdinLink (`0x4F4C`) / Apple (`0xFA57`) | Implemented — logs when Linux advertises |
| Mac kext: ACIO RX ring | Implemented, **off by default**. Register map is reverse-engineered and unverified. Arm with `odl_rdma_client -a`. |
| Apple official ThunderboltRDMA | Not used. Stub / Mac-only. |
| SIP | Must be off to load this unsigned kext |

Until the Mac RX ring is armed **and** Linux reaches `entering READY state`,
this is a control-plane + buffer. Data will not land.

## Mac (receiver)

SIP off (Recovery → `csrutil disable`). Xcode CLT installed.

```bash
cd mac/kext
make
sudo make load          # kextutil /tmp/OdinLinkRDMA.kext

cd ..
clang -o odl_rdma_client odl_rdma_client.c \
    -framework IOKit -framework CoreFoundation
./odl_rdma_client                 # watch completions
./odl_rdma_client -a              # also arm the RX ring
./odl_rdma_client -p 100          # poll 100 times
```

`log stream --predicate 'eventMessage CONTAINS "OdinLinkRDMA"'` shows
XDomain attach and whether NHI MMIO mapped.

Unload: `sudo make -C kext unload`

## Linux (sender)

```bash
cmake --build build --target odl_tensor_send
sudo insmod driver/odl_tb5.ko          # default protocol 0x4F4C
# wait for: odl_tb5: probed device … entering READY state
./build/mac/odl_tensor_send --fps 30
```

If `/dev/odl_tb5_*` never appears, the Mac is not advertising OdinLink
on XDomain. Check the kext is loaded and the TB5 cable is up
(`thunderbolt 0-1: new host found` is not enough).

`protocol=1` on Linux also advertises Apple’s `0xFA57`. The kext watches
both. Prefer default `0x4F4C` unless you are capturing Apple’s login.

The Mac kext still does **not** publish an XDomain directory or answer
login. Linux  `bind_any=1` (default) attaches to the Thunderbolt host
anyway; after three login timeouts it skips the handshake and brings
the data path up on hop 1 (`skip_login=1` does that immediately):

```bash
sudo insmod driver/odl_tb5.ko skip_login=1
```

## What you should see

1. Cable in. Linux: host found. Mac: Thunderbolt device appears.
2. Linux `insmod`. Mac kext logs `Protocol ID = 20236`.
3. Linux: `probed device` then `entering READY state`.
4. Mac: `./odl_rdma_client -a` — `RX ring ARMED`.
5. Linux `odl_tensor_send`. Mac client `rx_done` climbs.

If step 2 never happens, XDomain matching failed (wrong personality /
Thunderbolt stack did not publish the service). If step 4 panics, the
ACIO map is wrong — reboot, do **not** arm, file the panic log.

## Compressed tensors (ODLC)

The TB-bridge and `libodl_compress` wrap large tensors as **ODLC + LZ4
raw blocks** (algo 4) so more data fits on the Thunderbolt IP link.
Mac decode: `bridge/odl_compress.py` (uses `libcompression`) or
`odl_decompress_host()` in `compress/src/odl_compress_lz4.c`.

nvCOMP GDeflate / batched LZ4 (algo 1–3) is Linux CUDA only. If a blob's
header `algo` is not 4, reject it — do not feed it to MLX.

## Files

| File | Role |
|------|------|
| `include/odinlink_mac_proto.h` | Shared 4 KB slot + stream id + HELLO |
| `kext/OdinLinkRDMA.*` | Buffer, user client, lifecycle |
| `kext/OdinLinkNHI.cpp` | MMIO map, RX ring, XDomain watch |
| `kext/apple_tb5_nhi_mac.h` | ACIO offsets (same as Linux `apple_tb5_nhi_regs.h`) |
| `odl_rdma_client.c` | mmap + poll + optional `--arm` |
| `linux_test/odl_tensor_send.c` | Linux stream sender (no verbs) |
