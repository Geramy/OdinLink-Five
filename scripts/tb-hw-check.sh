#!/usr/bin/env bash
# Tell the user whether this Linux box already has Thunderbolt, and
# what card (if any) to buy. No sudo required for the usual sysfs/lspci path.
set -euo pipefail

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

vendor=$(tr -d '\0' < /sys/class/dmi/id/sys_vendor 2>/dev/null || echo unknown)
product=$(tr -d '\0' < /sys/class/dmi/id/product_name 2>/dev/null || echo unknown)
board=$(tr -d '\0' < /sys/class/dmi/id/board_name 2>/dev/null || echo unknown)

bold "This machine"
note "vendor : $vendor"
note "product: $product"
note "board  : $board"

nhi=$(lspci -nn 2>/dev/null | grep -iE '0c0340|Thunderbolt|USB4 Host' || true)
tb_sys=
if [ -d /sys/bus/thunderbolt/devices ]; then
	tb_sys=$(ls /sys/bus/thunderbolt/devices 2>/dev/null | tr '\n' ' ')
fi

echo
bold "Thunderbolt / USB4 NHI"
if [ -n "$nhi" ]; then
	printf '%s\n' "$nhi" | sed 's/^/  /'
else
	note "No Intel Thunderbolt NHI in lspci (class 0c0340)."
	note "A USB-C port or a generic USB4 card is not enough for OdinLink."
fi
if [ -n "${tb_sys:-}" ]; then
	note "thunderbolt bus: $tb_sys"
else
	note "No /sys/bus/thunderbolt — kernel has not bound a TB domain."
fi

echo
bold "What to do"
case "$board $product $vendor" in
	*"MS-S1 MAX"*|*"SHWSA"*)
		note "Minisforum MS-S1 MAX — onboard TB5. Do not buy an add-in card."
		note "Use a TB5 cable. For PyTorch + Mac RAM see docs/PYTORCH.md"
		;;
	*"Z690 CARBON"*|*"MS-7D30"*)
		note "MSI MPG Z690 Carbon WiFi — no onboard NHI."
		note "Buy MSI ThunderboltM4 (TB4) if JTBT1 + JUSB are free."
		note "Do NOT buy ASUS ThunderboltEX 5 or Gigabyte THUNDERBOLTS 5."
		note "No retail MSI TB5 card exists for this board."
		;;
	*"Micro-Star"*|*"MSI "*)
		note "MSI board: only MSI ThunderboltM4 (TB4, 500/600/700 + JTBT)"
		note "or a board that already has TB. ASUS/Gigabyte cards will not work."
		note "MSI ThunderboltM5 is TB5 but not sold separately (GODLIKE bundle)."
		;;
	*"ASUS"*|*"ASUSTeK"*)
		note "ASUS: ThunderboltEX 5 only if this model is on ASUS's list"
		note "and the Thunderbolt header is present."
		;;
	*"Gigabyte"*|*"GIGABYTE"*)
		note "Gigabyte: THUNDERBOLTS 5 only if this model has their TB headers."
		;;
	*)
		note "Unknown board. Need an Intel Thunderbolt NHI (lspci 0c0340),"
		note "not a USB4-only AIC. Add-in cards are brand-locked to the motherboard."
		;;
esac

if [ -n "$nhi" ]; then
	note "This box already has an NHI — for Ubuntu+PyTorch+Mac RAM you only"
	note "need a TB cable and docs/PYTORCH.md (Thunderbolt Bridge + bridge/)."
fi

echo
note "Docs: docs/HARDWARE.md  docs/PYTORCH.md  mac/README.md"
