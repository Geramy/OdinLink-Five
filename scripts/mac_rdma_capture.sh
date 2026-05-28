#!/usr/bin/env bash
# mac_rdma_capture.sh — collect all the data needed to reverse-engineer
# Apple's ThunderboltRDMA XDomain wire protocol.
#
# Run this on a Mac with TB5 hardware connected to ANOTHER Mac (also TB5)
# with RDMA enabled. Apple's stack discovers the peer via XDomain; we
# snapshot the discovery state and stream the kernel/RDMA logs while
# the connect happens.
#
# Prerequisites:
#   - macOS 26.2+ on Apple Silicon (Apple ThunderboltRDMA shipped here)
#   - Two Macs, both with TB5 ports
#   - TB5 cable (TB3/TB4 cables won't work — TN3205 confirms HW requirement)
#   - RDMA enabled on both: boot into Recovery → Terminal → `rdma_ctl enable`
#   - Reboot back to macOS
#   - This script run on EITHER Mac (better to run on the side that initiates
#     the login — usually whoever connects last)
#
# Output: a timestamped tarball with everything OdinLink-Five needs to
# implement `protocol=2` (Apple-compatible login send path).

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: this script must run on macOS (got $(uname))"
    exit 1
fi

OUTDIR="${1:-$HOME/mac_rdma_capture_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"
echo "==> capture output: $OUTDIR"

# ── Prereqs check ──────────────────────────────────────────────────────────
echo
echo "==> [1/8] Checking prerequisites"
macos_ver=$(sw_vers -productVersion)
echo "  macOS: $macos_ver"
echo "$macos_ver" > "$OUTDIR/macos_version.txt"

arch=$(uname -m)
echo "  arch:  $arch"
[[ "$arch" != "arm64" ]] && echo "  WARNING: Apple Silicon required for RDMA"

# Check kext
if kextstat 2>/dev/null | grep -qi ThunderboltRDMA; then
    echo "  ThunderboltRDMA.kext: loaded ✓"
    kextstat 2>/dev/null | grep -i Thunderbolt > "$OUTDIR/kextstat_thunderbolt.txt"
else
    echo "  ThunderboltRDMA.kext: NOT loaded — kext loads automatically when a TB5 RDMA peer is detected."
fi

# ── 2. Hardware inventory ─────────────────────────────────────────────────
echo
echo "==> [2/8] Snapshotting Thunderbolt hardware"
system_profiler SPThunderboltDataType > "$OUTDIR/thunderbolt_hw.txt" 2>&1
echo "  -> thunderbolt_hw.txt ($(wc -l < "$OUTDIR/thunderbolt_hw.txt") lines)"

# Check we have actual TB5
if grep -qi "thunderbolt 5\|usb4.*v2\|80 *gb" "$OUTDIR/thunderbolt_hw.txt"; then
    echo "  TB5 link: detected ✓"
else
    echo "  TB5 link: NOT detected — capture will fail without TB5"
    echo "  (RDMA only works on Thunderbolt 5 hardware per TN3205)"
fi

# ── 3. IOKit Thunderbolt XDomain tree (BEFORE peer connect) ───────────────
echo
echo "==> [3/8] IOKit XDomain tree (BEFORE peer connect)"
ioreg -lw0 -c IOThunderboltSwitch > "$OUTDIR/ioreg_thunderbolt_before.txt" 2>&1
ioreg -r -c IOThunderboltXDomainService > "$OUTDIR/ioreg_xdomain_before.txt" 2>&1
ioreg -r -c AppleThunderboltRDMA > "$OUTDIR/ioreg_rdma_before.txt" 2>&1
echo "  -> ioreg_*_before.txt"

# ── 4. Start log capture in background ────────────────────────────────────
echo
echo "==> [4/8] Starting log capture (40 seconds of Thunderbolt subsystem)"
# Predicate covers anything mentioning Thunderbolt or RDMA. Use --debug
# to get kernel-level messages too.
log stream --debug \
    --predicate 'subsystem CONTAINS "thunderbolt" OR subsystem CONTAINS "rdma" OR senderImagePath CONTAINS "Thunderbolt" OR senderImagePath CONTAINS "RDMA"' \
    > "$OUTDIR/log_stream.txt" 2>&1 &
LOG_PID=$!
echo "  log stream PID: $LOG_PID"

cleanup() {
    if kill -0 "$LOG_PID" 2>/dev/null; then
        kill "$LOG_PID" 2>/dev/null || true
        wait "$LOG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Brief wait for log stream to spin up
sleep 2

# ── 5. Trigger RDMA connect ───────────────────────────────────────────────
echo
echo "==> [5/8] Triggering RDMA connection"
echo "  Please now PHYSICALLY connect the TB5 cable to the peer Mac"
echo "  (or unplug & replug if already connected)"
echo "  Waiting 20 seconds for handshake..."
echo

sleep 20

# ── 6. IOKit snapshot (AFTER peer connect) ────────────────────────────────
echo "==> [6/8] IOKit XDomain tree (AFTER peer connect)"
ioreg -lw0 -c IOThunderboltSwitch > "$OUTDIR/ioreg_thunderbolt_after.txt" 2>&1
ioreg -r -c IOThunderboltXDomainService > "$OUTDIR/ioreg_xdomain_after.txt" 2>&1
ioreg -r -c AppleThunderboltRDMA > "$OUTDIR/ioreg_rdma_after.txt" 2>&1
echo "  -> ioreg_*_after.txt"

# Diff the before/after to highlight what changed during connect
diff -u "$OUTDIR/ioreg_xdomain_before.txt" "$OUTDIR/ioreg_xdomain_after.txt" \
    > "$OUTDIR/ioreg_xdomain_diff.txt" 2>&1 || true
diff -u "$OUTDIR/ioreg_rdma_before.txt" "$OUTDIR/ioreg_rdma_after.txt" \
    > "$OUTDIR/ioreg_rdma_diff.txt" 2>&1 || true

# ── 7. Probe with ibv_devinfo + ibv_rc_pingpong if available ──────────────
echo
echo "==> [7/8] Probe RDMA verbs API"
if command -v ibv_devinfo >/dev/null 2>&1; then
    ibv_devinfo > "$OUTDIR/ibv_devinfo.txt" 2>&1 || true
    ibv_devinfo -v >> "$OUTDIR/ibv_devinfo.txt" 2>&1 || true
    echo "  -> ibv_devinfo.txt"
else
    echo "  ibv_devinfo not installed. Apple's libthunderboltrdma.dylib"
    echo "  ships the verbs binaries — they're under"
    echo "  /System/Library/PrivateFrameworks/ThunderboltRDMA.framework/Versions/A/Helpers/"
    if [[ -d /System/Library/PrivateFrameworks/ThunderboltRDMA.framework ]]; then
        ls -la /System/Library/PrivateFrameworks/ThunderboltRDMA.framework/ \
            > "$OUTDIR/ls_apple_rdma_framework.txt" 2>&1 || true
    fi
fi

# Drop the kext Info.plist if accessible (advertises protocol IDs we need)
APPLE_KEXT="/System/Library/Extensions/AppleThunderboltRDMA.kext"
APPLE_KEXT_DT="/System/Library/DriverExtensions/AppleThunderboltRDMA.dext"
for kext in "$APPLE_KEXT" "$APPLE_KEXT_DT"; do
    if [[ -d "$kext" ]]; then
        cp -R "$kext/Contents/Info.plist" "$OUTDIR/$(basename "$kext")_Info.plist" 2>/dev/null || true
    fi
done

# ── 8. Stop log capture, package the bundle ───────────────────────────────
echo
echo "==> [8/8] Stopping log capture"
sleep 2  # let log stream catch up
kill "$LOG_PID" 2>/dev/null || true
wait "$LOG_PID" 2>/dev/null || true
trap - EXIT

# Filter log_stream for the high-signal lines so the analyst doesn't drown
grep -iE "login|xdomain|protocol|hop|ring|rdma|connect|handshake|uuid|prtcid" \
    "$OUTDIR/log_stream.txt" > "$OUTDIR/log_stream_filtered.txt" 2>/dev/null || true

# Summary
echo
echo "==> Summary"
{
    echo "macOS: $macos_ver  arch: $arch  captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "Files (size):"
    cd "$OUTDIR" && du -h *
} > "$OUTDIR/SUMMARY.txt"
cat "$OUTDIR/SUMMARY.txt"

# Bundle for transport
BUNDLE="$(dirname "$OUTDIR")/$(basename "$OUTDIR").tar.gz"
tar czf "$BUNDLE" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")"
echo
echo "==> Bundle: $BUNDLE"
echo "==> Transfer this to the OdinLink-Five Linux machine and run:"
echo "      python3 scripts/mac_rdma_analyze.py $BUNDLE"
echo
echo "Done."
