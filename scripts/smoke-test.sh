#!/usr/bin/env bash
#
# OdinLink-Five Smoke Test with Hardware
#
# Usage:
#   ./scripts/smoke-test.sh                  # run full suite
#   ./scripts/smoke-test.sh -t verbs         # verbs provider test only
#   ./scripts/smoke-test.sh -t suite         # library test suite only
#   ./scripts/smoke-test.sh -t bandwidth     # CLI bandwidth test (two machines)
#   ./scripts/smoke-test.sh -o /tmp/debug    # custom output directory
#
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$(realpath "${0%/*}/../build")}"
SCRIPT_DIR="$(realpath "${0%/*}")"
MODULE="${MODULE:-driver/odl_tb5.ko}"
DEVICE_NODE="/dev/odl_tb5_0"

# --- defaults ---
OUTPUT_DIR=""
TEST_TYPE="all"
CLI_MODE=""       # "server" or "client" for bandwidth test

while getopts "t:o:m:" opt; do
  case $opt in
    t) TEST_TYPE="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    m) CLI_MODE="$OPTARG"   ;;
    *) echo "Usage: $0 [-t suite|verbs|bandwidth] [-o output_dir] [-m server|client]"; exit 1 ;;
  esac
done

# --- output directory ---
if [ -z "$OUTPUT_DIR" ]; then
  TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
  OUTPUT_DIR="${SCRIPT_DIR}/../smoke-test-${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"
DEBUG_LOG="${OUTPUT_DIR}/debug.log"
KERNEL_LOG="${OUTPUT_DIR}/kernel.log"
TEST_LOG="${OUTPUT_DIR}/test.log"

# --- logging preamble ---
echo "============================================" | tee -a "$DEBUG_LOG"
echo " OdinLink-Five Smoke Test"                    | tee -a "$DEBUG_LOG"
echo " Started : $(date)"                           | tee -a "$DEBUG_LOG"
echo " Output  : ${OUTPUT_DIR}"                     | tee -a "$DEBUG_LOG"
echo "============================================" | tee -a "$DEBUG_LOG"

# --- enable verbose logging ---
export ODL_VERBS_DEBUG=5
echo "[setup] ODL_VERBS_DEBUG=5" | tee -a "$DEBUG_LOG"

# --- trap cleanup ---
cleanup() {
  local rc=$?
  if [ -n "${DMESG_PID:-}" ]; then
    kill "$DMESG_PID" 2>/dev/null || true
    wait "$DMESG_PID" 2>/dev/null || true
  fi
  echo "" | tee -a "$DEBUG_LOG"
  echo "============================================" | tee -a "$DEBUG_LOG"
  if [ "$rc" -eq 0 ]; then
    echo " RESULT: ALL SMOKE TESTS PASSED" | tee -a "$DEBUG_LOG"
  else
    echo " RESULT: SMOKE TEST FAILED (exit code $rc)" | tee -a "$DEBUG_LOG"
  fi
  echo " Kernel log : ${KERNEL_LOG}"                    | tee -a "$DEBUG_LOG"
  echo " Test log   : ${TEST_LOG}"                      | tee -a "$DEBUG_LOG"
  echo " Full debug : ${DEBUG_LOG}"                     | tee -a "$DEBUG_LOG"
  echo "============================================" | tee -a "$DEBUG_LOG"
  exit "$rc"
}
trap cleanup EXIT INT TERM

# --- kernel log watcher (background) ---
echo "[setup] Starting kernel log watcher → ${KERNEL_LOG}" | tee -a "$DEBUG_LOG"
sudo dmesg -w 2>/dev/null | grep --line-buffered 'odl_tb5' > "$KERNEL_LOG" &
DMESG_PID=$!
# give dmesg a moment to start
sleep 0.2

# --- 1. check kernel module ---
run_step() {
  local step="$1"
  local label="$2"
  shift 2
  echo "" | tee -a "$DEBUG_LOG"
  echo "--- [${step}] ${label} ---" | tee -a "$DEBUG_LOG"
  "$@" 2>&1 | tee -a "$TEST_LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "  ✓ ${label}" | tee -a "$DEBUG_LOG"
  elif [ "$rc" -eq 77 ]; then
    echo "  ∼ ${label} (skipped)" | tee -a "$DEBUG_LOG"
  else
    echo "  ✗ ${label} (exit code ${rc})" | tee -a "$DEBUG_LOG"
  fi
  return "$rc"
}

# --- module check ---
echo "" | tee -a "$DEBUG_LOG"
echo "--- [pre] Checking kernel module ---" | tee -a "$DEBUG_LOG"
if lsmod | grep -q '^odl_tb5'; then
  echo "  ✓ odl_tb5 module loaded" | tee -a "$DEBUG_LOG"
else
  echo "  ! Module not loaded — try: sudo insmod ${MODULE}" | tee -a "$DEBUG_LOG"
  # don't fail, some tests can still run
fi

# --- tests ---

ALL_PASSED=true

if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "suite" ]; then
  run_step "2" "Full test suite" "${BUILD_DIR}/tests/odl_tb5_test" || ALL_PASSED=false
fi

if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "verbs" ]; then
  run_step "3" "Verbs provider lifecycle" \
    "${BUILD_DIR}/verbs/tests/test_verbs_basic" || ALL_PASSED=false
fi

if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "ibv_devinfo" ]; then
  run_step "5" "ibv_devinfo" bash -c "ibv_devinfo 2>&1 | head -30" || ALL_PASSED=false
fi

if [ "$TEST_TYPE" = "bandwidth" ]; then
  if [ "$CLI_MODE" = "server" ]; then
    run_step "4" "CLI server" \
      "${BUILD_DIR}/cli/odl_tb5_cli" --server --device 0 || ALL_PASSED=false
  elif [ "$CLI_MODE" = "client" ]; then
    run_step "4" "CLI bandwidth test" \
      "${BUILD_DIR}/cli/odl_tb5_cli" --client --device 0 --test bandwidth || ALL_PASSED=false
  else
    echo "  ! Bandwidth test requires -m server or -m client" | tee -a "$DEBUG_LOG"
    ALL_PASSED=false
  fi
fi

# --- finish ---
if [ "$ALL_PASSED" = true ]; then
  exit 0
else
  exit 1
fi
