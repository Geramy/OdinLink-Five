# Capturing Apple's ThunderboltRDMA Wire Protocol

## Why this exists

OdinLink-Five's `protocol=1` mode advertises Apple's protocol ID
(`0xFA57` / 64087) and accepts lenient login responses, but the **outbound
login message** is still serialised in OdinLink's own UUID + opcode +
field layout. That means a Mac receiving our login probably rejects it,
because Apple's `AppleThunderboltRDMA.kext` is matching against its own
expected message format.

To finish Mac↔Linux interop (`protocol=2` mode: actually send Apple-
compatible login bytes) we need to know:

| Item | Today's status |
|---|---|
| Apple's RDMA UUID | Unknown |
| Login message TYPE opcode | Unknown |
| Login payload byte layout (transmit_path / proto_version offsets) | Unknown |
| Login response field layout | Partial (lenient parser already in place) |
| Hop ID negotiation order | Unknown |
| NHI ring descriptor format compatibility | Unknown but probably standard NHI |

Every one of these is *recoverable* by running a Mac↔Mac RDMA session
once and capturing the relevant kernel + IOKit state. This doc shows how.

## What you need

- **Two Macs**, both with **Thunderbolt 5** ports (M4 Mac Mini, M4 Pro/Max
  MacBook Pro, M3 Ultra Mac Studio, etc. — TB3/TB4 won't work, TN3205
  is explicit about this).
- **macOS 26.2 or later** on both.
- **A TB5-rated cable** (the cheap TB3 passive cables don't carry TB5
  speeds and won't negotiate RDMA).
- RDMA enabled on **both** Macs:
  ```
  # Boot into macOS Recovery (Apple Silicon: hold power on boot)
  Terminal → Utilities → Terminal
  rdma_ctl enable
  reboot
  ```

## Run the capture

On *one* of the two Macs (preferably the one that will initiate the
login — that's whoever connects the cable last):

```bash
git clone https://github.com/Geramy/OdinLink-Five.git
cd OdinLink-Five
chmod +x scripts/mac_rdma_capture.sh
./scripts/mac_rdma_capture.sh
```

The script will:

1. Verify macOS version and TB5 hardware
2. Snapshot `ioreg` Thunderbolt + XDomain + RDMA trees (before)
3. Start a 40-second `log stream` capture of the Thunderbolt + RDMA
   subsystems
4. Wait 20 seconds while you (re-)plug the TB5 cable — the kext will
   discover the peer and run its XDomain login during this window
5. Snapshot `ioreg` trees again (after)
6. Probe `ibv_devinfo` if available
7. Copy `AppleThunderboltRDMA.kext/Contents/Info.plist` for protocol IDs
8. Bundle everything into `~/mac_rdma_capture_<timestamp>.tar.gz`

If anything fails in step 5 (peer not visible after connect), the
script will tell you and the analyser will say so too.

## Analyze the capture

Transfer the tarball to a Linux box (the OdinLink dev machine works fine):

```bash
scp mymac:~/mac_rdma_capture_*.tar.gz .
python3 scripts/mac_rdma_analyze.py mac_rdma_capture_<ts>.tar.gz
```

The analyser will report:

- **Apple's RDMA Protocol ID** and **Protocol Version** (from the kext
  plist — confirms the well-known `0xFA57` and surfaces the version
  number, which OdinLink currently hardcodes to 1)
- **XDomain property dirs** the Mac is advertising (so we know the
  property-key names to match against — Apple may use `"rdma"` plus
  some other key we haven't seen)
- **UUID candidates** from the log stream (the Apple RDMA UUID will
  appear many times; the most-frequent one is the strongest lead)
- **Opcode candidates** (any `type=N` / `opcode=N` style strings)
- **Login lines** — actual `pr_info`-style messages around the login
  handshake, which is where the message format is described in
  human-readable form even when the raw bytes aren't logged

It also prints a **next-steps** checklist: what's missing, what's
ambiguous, what to capture next.

## From capture → driver code

Once you have:

- Apple's RDMA UUID (as a 16-byte UUID literal)
- Login message TYPE opcode (a small integer)
- Login payload byte layout (which u32 is `transmit_path`, which is
  `proto_version`, what reserved fields look like)

…implementing `protocol=2` in `driver/odl_tb5_proto.c` is mechanical:

1. Add a second UUID constant at the top of `odl_tb5_proto.c`:
   ```c
   const uuid_t odl_tb5_proto_uuid_apple =
       UUID_INIT(0xXXXXXXXX, 0xXXXX, 0xXXXX,
                 0xXX, 0xXX, 0xXX, 0xXX, 0xXX, 0xXX, 0xXX, 0xXX);
   ```
2. Add a second login struct (`struct odl_tb5_login_msg_apple`) with the
   recovered field layout.
3. In `odl_tb5_proto_send_login()`, branch on `odl_protocol_mode == 2`
   to fill in the Apple struct, use the Apple UUID, and pass the Apple
   opcode.
4. Register the Apple UUID as a second protocol handler in
   `odl_tb5_proto_register()` so incoming Apple-format login requests
   are routed to the same handler.

Total surgery: ~150 LoC if the capture nails down all four unknowns.

## What "success" looks like

After implementing `protocol=2` and re-running on Linux:

```bash
sudo insmod driver/odl_tb5.ko protocol=2
# On the Mac, plug in the TB5 cable. Then:
ibv_devinfo
# should show the Linux peer as a connected RDMA device
ibv_rc_pingpong -d <device>
# should complete a ping-pong without error
```

That's the verbs-level handshake. Sharing GPU VRAM zero-copy still
requires the dmabuf bridge on the Mac side and is a separate problem
documented in [`REMOTE_TENSORS.md`](REMOTE_TENSORS.md).

## Why not just sniff PCIe?

Thunderbolt control packets ride on the PCIe upstream link, which would
normally need a PCIe analyzer (Teledyne LeCroy, Keysight — $$$$).
However, Apple's kext **logs the high-level handshake** to the `log
stream` subsystem at the debug level, which is more than enough to
reconstruct the wire format because Apple's engineers debug the same
way. PCIe-level capture would only be necessary if Apple suppresses
the relevant log lines — and so far there's no evidence they do.

If `log stream` turns out to be insufficient, the fallback is a
DriverKit-based passive shim that interposes between the kext and
`libthunderboltrdma.dylib` — that's a much bigger project and shouldn't
be needed for this task.

## Gotchas

- **SIP**: System Integrity Protection blocks some `ioreg` views and
  may rate-limit `log stream`. If the capture is sparse, disable SIP
  on a sacrificial Mac (`csrutil disable` from Recovery) and re-run.
- **First-connect-only**: Apple's kext caches discovery state. If you
  ran the script once and it captured nothing useful, fully unload the
  kext (`sudo kmutil unload -p .../AppleThunderboltRDMA.kext`) or
  reboot before re-running, otherwise the second capture will be
  noise.
- **Sleep**: don't let the Mac sleep during capture — the TB controller
  may renegotiate on wake and you'll get a confusing mix of events.
