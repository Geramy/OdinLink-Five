# Packaging

## .deb Packages

Build installable packages:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cpack                    # Individual component .debs
make meta-packages       # User-friendly bundles
```

### Packages

| Package | Contents |
|---------|----------|
| `odl-tb5-minimal` | dkms + library + RCCL plugin (GPU cluster node) |
| `odl-tb5-server` | dkms + library + CLI + daemon + RCCL plugin (headless server) |
| `odl-tb5-desktop` | dkms + library + CLI + daemon + tray (desktop workstation) |
| `odl-tb5-full` | Everything |

## DKMS

The kernel module can be built via DKMS for automatic recompilation on
kernel updates. The necessary files are in `packaging/dkms.conf` and the
driver source files are installed by CPack's `dkms` component.
