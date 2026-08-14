#!/usr/bin/env python3
"""Park a CUDA (or CPU) tensor on a Mac over Thunderbolt IP and fetch it back.

    # Mac:  python3 bridge/tb_bridge_server.py --bind 0.0.0.0 --max-gb 96 -v
    # Linux:
    python3 examples/pytorch_mac_offload.py --url tb5://192.168.167.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from odinlink import RemoteStore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True,
                    help="tb5://<mac-thunderbolt-bridge-ip>[:29800]")
    ap.add_argument("--cpu", action="store_true",
                    help="Use CPU tensors (no CUDA)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("install PyTorch on the Ubuntu trainer first", file=sys.stderr)
        return 1

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    mac = RemoteStore(args.url)
    x = torch.randn(256, 1024, dtype=torch.float16, device=device)
    mac.put("demo.activations", x)
    y = mac.get("demo.activations", device=device)
    ok = torch.equal(x, y)
    info = mac.stat()
    print(f"device={device} equal={ok} "
          f"store={info['num_keys']} keys, {info['total_bytes']} bytes")
    mac.delete("demo.activations")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
