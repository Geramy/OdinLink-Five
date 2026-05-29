#!/usr/bin/env python3
"""Analyze a tarball produced by `scripts/mac_rdma_capture.sh`.

Extracts the bits of Apple's XDomain protocol that OdinLink-Five needs to
implement `protocol=2` (Apple-compatible login *send* path, not just the
lenient response handling already in `protocol=1`).

Specifically attempts to recover:
- Apple's RDMA service UUID (from the kext Info.plist if present)
- Protocol ID + Protocol Version (cross-checks the 0xFA57 / 64087 value)
- XDomain login message TYPE opcode (from logs)
- Login payload field offsets (transmit_path, proto_version, …)
- Hop ID exchange order

Most of these are unknowns until a real capture lands. The script emits a
report indicating which fields it was able to recover and which still need
human inspection of the raw logs.
"""
import argparse
import json
import os
import plistlib
import re
import sys
import tarfile
from pathlib import Path
from collections import Counter


def open_bundle(path: str) -> Path:
    """Extract the tarball into a temp dir and return its root."""
    if path.endswith((".tgz", ".tar.gz")):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="mac_rdma_"))
        with tarfile.open(path) as tf:
            tf.extractall(tmp)
        # Find the single subdir
        children = [c for c in tmp.iterdir() if c.is_dir()]
        return children[0] if len(children) == 1 else tmp
    if Path(path).is_dir():
        return Path(path)
    raise ValueError(f"unrecognized bundle path: {path}")


def parse_kext_plist(root: Path) -> dict:
    """Return Apple's kext Info.plist as a dict (if captured)."""
    for name in ["AppleThunderboltRDMA.kext_Info.plist",
                 "AppleThunderboltRDMA.dext_Info.plist"]:
        p = root / name
        if p.exists():
            with open(p, "rb") as f:
                return plistlib.load(f)
    return {}


def extract_protocol_ids(plist: dict) -> dict:
    """Walk the kext plist for IOKitPersonalities → property matches."""
    found = {}
    if not plist:
        return found
    personalities = plist.get("IOKitPersonalities") or {}
    for pname, props in personalities.items():
        match = props.get("IOPropertyMatch")
        if isinstance(match, dict):
            pid = match.get("Protocol ID")
            pver = match.get("Protocol Version")
            if pid is not None or pver is not None:
                found[pname] = {"Protocol ID": pid, "Protocol Version": pver,
                                "Hex Protocol ID": hex(pid) if pid else None}
        # Some kexts use ProviderClass + a flat key
        if "Protocol ID" in props:
            found.setdefault(pname, {})["Protocol ID"] = props["Protocol ID"]
    return found


def parse_xdomain_props(root: Path) -> dict:
    """Read ioreg_xdomain_after.txt and surface the property dirs."""
    p = root / "ioreg_xdomain_after.txt"
    if not p.exists():
        return {"_note": "ioreg_xdomain_after.txt missing — peer probably did not connect"}
    text = p.read_text(errors="replace")
    # Look for "prtcid" = <number> and "ptcvr" / similar
    fields = {}
    for key in ("prtcid", "ptcvr", "prtcrev", "prtcstns"):
        m = re.search(rf'"{key}"\s*=\s*(\d+)', text)
        if m:
            fields[key] = int(m.group(1))
            if key == "prtcid":
                fields[key + "_hex"] = hex(int(m.group(1)))
    # Look for protocol *string* directory name
    keys = re.findall(r'\| \+-o ([A-Za-z0-9_\-]+) <class IOThunderbolt', text)
    if keys:
        fields["_property_dirs"] = sorted(set(keys))
    return fields


def parse_log_stream(root: Path) -> dict:
    """Scan log_stream_filtered.txt for the login/XDomain signal we care about."""
    p = root / "log_stream_filtered.txt"
    if not p.exists():
        # Fall back to full log
        p = root / "log_stream.txt"
        if not p.exists():
            return {"_error": "no log_stream files found"}
    text = p.read_text(errors="replace")

    findings = {
        "n_lines": text.count("\n"),
        "login_lines": [],
        "uuid_candidates": [],
        "opcode_candidates": [],
        "hop_id_lines": [],
    }

    # Login mentions
    for line in text.splitlines():
        ll = line.lower()
        if "login" in ll and ("rdma" in ll or "xdomain" in ll or "thunderbolt" in ll):
            findings["login_lines"].append(line[:300])
        if re.search(r"hop[ -_]?id", ll):
            findings["hop_id_lines"].append(line[:300])
        # UUID candidates: 8-4-4-4-12 hex
        for m in re.finditer(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                              r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", line):
            findings["uuid_candidates"].append(m.group(0))
        # Opcode-like patterns: "type=N" or "msg type N"
        for m in re.finditer(r"(?:msg.?type|opcode|type)\s*[:=]\s*(0x[0-9a-f]+|\d+)", ll):
            findings["opcode_candidates"].append(m.group(1))

    # Dedupe + most-common
    findings["uuid_candidates"] = Counter(findings["uuid_candidates"]).most_common(10)
    findings["opcode_candidates"] = Counter(findings["opcode_candidates"]).most_common(20)
    findings["login_lines"] = findings["login_lines"][:30]
    findings["hop_id_lines"] = findings["hop_id_lines"][:30]
    return findings


def make_report(root: Path) -> dict:
    plist = parse_kext_plist(root)
    protocol_ids = extract_protocol_ids(plist)
    xdomain_props = parse_xdomain_props(root)
    log_findings = parse_log_stream(root)

    macos_ver = (root / "macos_version.txt")
    macos_ver_str = macos_ver.read_text().strip() if macos_ver.exists() else "?"

    return {
        "capture_root": str(root),
        "macos_version": macos_ver_str,
        "kext_plist_found": bool(plist),
        "protocol_ids_from_kext": protocol_ids,
        "xdomain_properties_after_connect": xdomain_props,
        "log_findings": log_findings,
        "next_steps": _next_steps(plist, xdomain_props, log_findings),
    }


def _next_steps(plist, xdomain_props, log_findings) -> list:
    out = []
    if not plist:
        out.append("Apple kext Info.plist was not captured (SIP may have blocked the copy). "
                   "Try running mac_rdma_capture.sh as root or manually copy "
                   "/System/Library/Extensions/AppleThunderboltRDMA.kext/Contents/Info.plist")
    if "_error" in log_findings:
        out.append("log_stream files missing — capture script may not have run with "
                   "sufficient privileges. Re-run with `sudo` and ensure `log` allows debug streams.")
    if not log_findings.get("login_lines"):
        out.append("No 'login' lines in the log capture. Either the peer didn't connect "
                   "during the 20-second window (re-plug the cable during the capture) "
                   "or Apple's log subsystem suppresses RDMA login messages by default. "
                   "Try widening the predicate in mac_rdma_capture.sh.")
    if not xdomain_props or xdomain_props.get("_note"):
        out.append("No XDomain peer properties recorded post-connect. Verify "
                   "`rdma_ctl enable` was run on both Macs (in Recovery) and that "
                   "`ibv_devinfo` shows the device on at least one side.")
    if log_findings.get("uuid_candidates"):
        out.append("UUID candidates found — cross-reference these against the kext "
                   "personalities to identify Apple's RDMA proto UUID. "
                   f"Top candidate: {log_findings['uuid_candidates'][0]}")
    if not out:
        out.append("Capture looks complete. Hand-inspect log_stream_filtered.txt for "
                   "the exact login message bytes; encode those into a new "
                   "odl_tb5_login_msg_apple struct in driver/odl_tb5_proto.c.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="path to .tar.gz or extracted directory")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of human report")
    args = ap.parse_args()

    root = open_bundle(args.bundle)
    report = make_report(root)

    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        print()
        return

    print(f"=== mac_rdma_analyze — {report['capture_root']} ===")
    print(f"macOS: {report['macos_version']}")
    print()
    print("[1] Apple kext personalities (Protocol IDs):")
    if report["protocol_ids_from_kext"]:
        for pname, props in report["protocol_ids_from_kext"].items():
            print(f"  {pname}:")
            for k, v in props.items():
                print(f"    {k}: {v}")
    else:
        print("  none (kext plist not captured)")
    print()
    print("[2] XDomain peer properties (post-connect):")
    if isinstance(report["xdomain_properties_after_connect"], dict):
        for k, v in report["xdomain_properties_after_connect"].items():
            print(f"  {k}: {v}")
    print()
    print("[3] Log findings:")
    lf = report["log_findings"]
    if "_error" in lf:
        print(f"  ERROR: {lf['_error']}")
    else:
        print(f"  total lines parsed: {lf.get('n_lines', 0)}")
        print(f"  login lines: {len(lf.get('login_lines', []))}")
        print(f"  hop-id lines: {len(lf.get('hop_id_lines', []))}")
        if lf.get("uuid_candidates"):
            print("  top UUID candidates:")
            for u, n in lf["uuid_candidates"][:5]:
                print(f"    {u}  ({n}×)")
        if lf.get("opcode_candidates"):
            print("  top opcode candidates:")
            for o, n in lf["opcode_candidates"][:5]:
                print(f"    {o}  ({n}×)")
        if lf.get("login_lines"):
            print("  sample login log lines:")
            for ln in lf["login_lines"][:5]:
                print(f"    {ln}")
    print()
    print("[4] Next steps:")
    for s in report["next_steps"]:
        print(f"  • {s}")
    print()


if __name__ == "__main__":
    main()
