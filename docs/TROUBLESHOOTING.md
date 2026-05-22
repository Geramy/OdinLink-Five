# Troubleshooting

## Kernel Module

| Problem | Solution |
|---------|----------|
| Build fails with unknown GCC flags | GCC is too old. Install version matching kernel (`cat /proc/version`) |
| Module won't load | Check `dmesg | grep odl_tb5`. Ensure TB5 hardware is present (`lspci | grep Thunderbolt`) |
| No `/dev/odl_tb5_*` devices | Device appears only when a TB5 peer connects. Check `dmesg` for XDomain events. Use `loopback=1` for no-cable testing |
| Permission denied | Install udev rule or `sudo chmod 660 /dev/odl_tb5_*` |

## Daemon & Tray

| Problem | Solution |
|---------|----------|
| Daemon won't start | Check `journalctl --user -u odl-tb5-daemon` for D-Bus errors |
| Tray icon not visible | Install `gnome-shell-extension-appindicator` on GNOME/Wayland |

## Verbs Provider

| Problem | Solution |
|---------|----------|
| `ibv_devinfo` doesn't show ODL device | Provider plugin not installed. Check `/usr/lib/*/libibverbs/` for `libodl_tb5-rdmav34.so` |
| `ibv_open_device` fails | Check `ODL_VERBS_DEBUG=5` for details. Ensure module loaded |
| `ibv_post_send` returns `-EAGAIN` | Normal for non-blocking mode. Worker thread retries after poll |

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
