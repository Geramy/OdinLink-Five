# Troubleshooting

## Start here: automatic diagnosis

```bash
sudo build/cli/odl_tb5_cli diag -v
```

Checks every layer — controller → cable → link training → peer discovery →
driver bind → login → READY — and says in plain English which one is broken
and what to do about it. Run it on **both** machines. The kernel side of the
picture is also available raw at `/sys/kernel/debug/odl_tb5/status`.

Two hard-won field notes it encodes:

- A **host router reset** (done by default on every `thunderbolt` module
  load/boot) can wedge the peer's connector state mid-link: ports then report
  "unplugged" with the cable seated until a physical replug. Prevent it with
  `options thunderbolt host_reset=false` in `/etc/modprobe.d/thunderbolt.conf`.
- "Cable detected but training never completes" at every speed (Gen 2/3/4),
  on more than one port pair, means the physical path — reseat or replace the
  cable before suspecting the driver.

## Kernel Module

| Problem | Solution |
|---------|----------|
| Build fails with unknown GCC flags | GCC is too old. Install version matching kernel (`cat /proc/version`) |
| Module won't load | Check `dmesg \| grep odl_tb5`. Ensure TB5 hardware is present (`lspci \| grep Thunderbolt`) |
| No `/dev/odl_tb5_*` devices | Device appears only when a TB5 peer connects and the link reaches **READY**. After plug or reload wait for `odl_tb5: entering READY state` (DMA-ping can take ~tens of seconds). Use `loopback=1` for no-cable testing |
| Permission denied | Install udev rule or `sudo chmod 660 /dev/odl_tb5_*` |
| Probe fails with `failed to alloc TX DMA buf` / `-12` | Identity IOMMU (`iommu=pt`). Load with `odl_ring_size=1024`, or use a translated IOMMU domain. Recent drivers auto-downgrade ring size |
| `insmod ... ring_size=1024` ignored | Parameter is named **`odl_ring_size`**, not `ring_size` |
| Link looks READY but every app handshake times out | Protocol version mismatch between nodes. dmesg should now show `peer protocol version N != local M — refusing`. Update both sides to the same revision |
| CLI server: `Receive error: Protocol error` after 1–2 tests | Multi-block-size bandwidth needs one TEST_REQ per size (fixed in recent CLI). Update both client and server binaries |

## Daemon & Tray

| Problem | Solution |
|---------|----------|
| Daemon won't start | Check `journalctl --user -u odl-tb5-daemon` for D-Bus errors |
| Tray icon not visible | Install `gnome-shell-extension-appindicator` on GNOME/Wayland |

## Verbs Provider

| Problem | Solution |
|---------|----------|
| `ibv_devinfo` shows no devices | On OdinLink-only hosts the directory plugin never loads (rdma-core only dlopens providers for sysfs RDMA devices). Use `LD_PRELOAD=.../libodl_tb5_verbs.so ibv_devinfo` |
| `ibv_open_device` fails | Check `ODL_VERBS_DEBUG=5` for details. Ensure module loaded and peer READY |
| `ibv_post_send` returns `-EAGAIN` | Normal for non-blocking mode. Worker thread retries after poll |

## RCCL / NCCL

| Problem | Solution |
|---------|----------|
| Collectives work but are ~4× too slow | Silent Socket fallback. Check logs for `Using network Socket` vs `Using network ODL_TB5`. Put `librccl-net.so` on `LD_LIBRARY_PATH` / `RCCL_PLUGIN_DIR` |
| `NET/IB : No device found` with RCCL | Expected without the net plugin — RCCL bypasses `LD_PRELOAD` verbs. Use `rccl_net_odl_tb5` / `librccl-net.so` |
| Built artifact name vs probe name | Build produces `librccl_net_odl_tb5.so`; RCCL probes `librccl-net.so`. CMake creates a symlink in the build tree and installs both names |

## GRUB

If TB5 ports aren't working reliably, add to kernel command line:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_port_pm=off"
```

## Debug

```bash
# Kernel driver debug
echo 'module odl_tb5 +p' | sudo tee /sys/kernel/debug/dynamic_debug/control
dmesg -w | grep odl_tb5

# Verbs provider trace
export ODL_VERBS_DEBUG=5

# Daemon foreground with verbose output
./build/daemon/odl_tb5_daemon -f

# RCCL debug
export RCCL_DEBUG=INFO

# NCCL debug
export NCCL_DEBUG=INFO
```
