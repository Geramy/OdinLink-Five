"""
RemoteStore — park PyTorch tensors in Mac unified memory.

Today this is Thunderbolt-IP copies via bridge/ (25–40 Gb/s). It is not
NCCL and not device=\"cuda:1\". DMA into Mac pages is a separate kext
path and is not wired here until that path shows READY + rx_done.

    from odinlink import RemoteStore
    import torch

    mac = RemoteStore("tb5://192.168.167.2")   # Mac Thunderbolt Bridge IP
    x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
    mac.put("kv.layer.42", x)
    y = mac.get("kv.layer.42", device="cuda")
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence
from urllib.parse import urlparse

# Repo layout: odinlink/ sits next to bridge/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bridge.tb_bridge_client import TBBridgeClient  # noqa: E402


def _parse_url(url: str) -> tuple[str, int]:
    raw = url.strip()
    if "://" not in raw:
        raw = "tb5://" + raw
    parsed = urlparse(raw)
    host = parsed.hostname
    if not host:
        raise ValueError(f"RemoteStore needs a host, got {url!r}")
    port = parsed.port if parsed.port else 29800
    return host, port


class RemoteStore:
    """One Mac (or other peer) used as extra RAM for a Linux trainer."""

    def __init__(self, url: str, timeout: float = 30.0):
        host, port = _parse_url(url)
        self.url = url
        self.host = host
        self.port = port
        self._cli = TBBridgeClient(host, port, timeout=timeout)

    def put(self, key: str, tensor) -> str:
        self._cli.put(key, tensor)
        return key

    def get(self, key: str, device: Optional[str] = None):
        return self._cli.get(key, device=device)

    def delete(self, key: str) -> bool:
        return self._cli.delete(key)

    def list(self) -> Sequence[str]:
        return self._cli.list()

    def stat(self):
        return self._cli.stat()

    @contextmanager
    def prefetch_window(self, keys: Sequence[str]) -> Iterator[None]:
        """API placeholder — fetches nothing extra on the IP path."""
        del keys
        yield
