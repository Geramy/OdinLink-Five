# AGENTS.md — OdinLink-Five

## Build

```bash
mkdir build && cd build && cmake .. && make -j$(nproc)
```

- **GCC version must match kernel build compiler** (`cat /proc/version`). CMake probes `gcc-15`, `gcc-14`, then `gcc`.
- Kernel module (`driver/odl_tb5.ko`) is built via CMake custom target — not `add_subdirectory`. It runs `make` inside `driver/`.
- Daemon and tray are **auto-disabled** if dependency `pkg-config` checks fail. Run `cmake ..` to see which components are ON.
- `gdbus-codegen` (from `libglib2.0-dev-bin`) is required at build time — generates D-Bus C bindings from XML.
- .deb packages: `cpack` (individual), `make meta-packages` (bundled: minimal/server/desktop/full).

## Tests

- Single binary: `build/tests/odl_tb5_test` (3 suites: device, lib_api, plugin).
- **Prerequisites**: kernel module loaded (`sudo insmod driver/odl_tb5.ko`), device readable (`sudo chmod 666 /dev/odl_tb5_0`).
- No test framework — plain C `main()` returning failure count.
- Verbs provider test: `build/verbs/tests/test_verbs_basic` (link with `-lodl_tb5_verbs -libverbs`).

## Architecture

| Component | Path | License |
|-----------|------|---------|
| Kernel driver | `driver/odl_tb5.ko` (4 .c files) | **GPL v2** |
| Userspace library | `lib/libodl_tb5.so` | MIT |
| RCCL plugin (AMD) | `rccl/librccl_net_odl_tb5.so` | MIT |
| NCCL plugin (NVIDIA) | `nccl/libnccl-net-ODL_TB5.so` | MIT |
| CLI tool | `cli/odl_tb5_cli` | MIT |
| Daemon | `daemon/odl_tb5_daemon` (GLib/D-Bus) | MIT |
| Tray app | `tray/odl_tb5_tray` (GTK3/AppIndicator) | MIT |
| Verbs provider (standalone) | `verbs/libodl_tb5_verbs.so` (symbol interposition) | MIT |
| Verbs provider (plugin) | `verbs/libodl_tb5-rdmav34.so` (rdma-core provider) | MIT |

Cross-platform compat docs at `COMPAT.md`. Apple's TB RDMA protocol ID = 64087 (0xFA57); OdinLink uses 0x4F4C. They must match for Mac↔Linux interop.

- Library source: `lib/src/odl_tb5_{dev,xfer,peer,completion,stream}.c`, header at `lib/include/odl_tb5/odl_tb5.h`.
- Kernel uapi header: `driver/uapi/odl_tb5_uapi.h` (ioctl defs shared with userspace).
- Third-party headers (vendored): `third_party/rccl/net_v7.h`, `third_party/nccl/`.
- Daemon reuses CLI source files (proto, stats, test sources) — linked directly, not via a shared library.

## Module Parameters

`ring_size=4096` (range 64–16384, power of 2). Each entry = 4 KB. Pass as `sudo insmod driver/odl_tb5.ko ring_size=1024`.

## Gotchas

- `/dev/odl_tb5_N` appears **only when a TB5 peer connects** (XDomain event). No peer → no device node. Exception: `loopback=1` module parameter creates fake devices for testing without a cable.
- Verbs provider plugin (`libodl_tb5-rdmav34.so`) auto-registers with rdma-core — install to `/usr/lib/*/libibverbs/` for `ibv_devinfo` discovery.
- Kernel module `hrtimer_setup` compat macro checks `< KERNEL_VERSION(6, 11, 0)`. Ubuntu 24.04 (6.8) uses the fallback.
- udev rule: `driver/71-odl-tb5.rules`. Install with `sudo cp ... /etc/udev/rules.d/ && sudo udevadm control --reload-rules`.
- Two API layers in the uapi header: legacy double-buffer ioctls (0x01–0x0D) and stream-based multiplexed I/O (0x20–0x27). Both coexist.
- NCCL plugin env: `NCCL_NET_PLUGIN=ODL_TB5`, `NCCL_PLUGIN_DIR=<build>/nccl`. Requires CUDA 11.7+, nvidia-drm modeset, NCCL 2.12+.
- RCCL plugin env: `RCCL_NET_PLUGIN=ODL_TB5`, `RCCL_PLUGIN_DIR=<build>/rccl`.
- Shared-memory stats exported at `/run/odl_tb5/{rccl,nccl}_stats`.
- No linter, formatter, or CI configuration in the repo. Pure C99 with CMake. No codegen beyond `gdbus-codegen`.
