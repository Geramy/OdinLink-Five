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

    with mac.prefetch_window(["kv.layer.41", "kv.layer.42"]):
        y = mac.get("kv.layer.42", device="cuda")
"""
from __future__ import annotations

import sys
import threading
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


def _to_device(val, device: Optional[str]):
    if device is None:
        return val
    to = getattr(val, "to", None)
    if callable(to):
        return to(device)
    return val


class RemoteStore:
    """One Mac (or other peer) used as extra RAM for a Linux trainer."""

    def __init__(self, url: str, timeout: float = 30.0):
        host, port = _parse_url(url)
        self.url = url
        self.host = host
        self.port = port
        self._cli = TBBridgeClient(host, port, timeout=timeout)
        self._prefetch_lock = threading.Lock()
        self._prefetch: dict = {}
        self._prefetch_pending: dict = {}
        self._prefetch_errors: dict = {}

    def close(self) -> None:
        self._cli.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def put(self, key: str, tensor) -> str:
        self._cli.put(key, tensor)
        return key

    def get(self, key: str, device: Optional[str] = None):
        ev = None
        with self._prefetch_lock:
            if key in self._prefetch:
                val = self._prefetch.pop(key)
                return _to_device(val, device)
            ev = self._prefetch_pending.get(key)
        if ev is not None:
            ev.wait()
            with self._prefetch_lock:
                err = self._prefetch_errors.pop(key, None)
                if err is not None:
                    raise err
                if key in self._prefetch:
                    val = self._prefetch.pop(key)
                    return _to_device(val, device)
        return self._cli.get(key, device=device)

    def delete(self, key: str) -> bool:
        with self._prefetch_lock:
            self._prefetch.pop(key, None)
            self._prefetch_errors.pop(key, None)
        return self._cli.delete(key)

    def list(self) -> Sequence[str]:
        return self._cli.list()

    def stat(self):
        return self._cli.stat()

    @contextmanager
    def prefetch_window(self, keys: Sequence[str]) -> Iterator[None]:
        """Pull ``keys`` over the reused TCP link while the caller computes.

        ``get()`` inside the window waits for an in-flight key and consumes
        the cached copy (no second round-trip). Unused entries are dropped
        when the window exits.
        """
        wanted = [k for k in keys if k]
        events = {k: threading.Event() for k in wanted}
        with self._prefetch_lock:
            for k, ev in events.items():
                if k not in self._prefetch_pending:
                    self._prefetch_pending[k] = ev

        def worker() -> None:
            for k in wanted:
                try:
                    val = self._cli.get(k)
                    with self._prefetch_lock:
                        self._prefetch[k] = val
                except BaseException as exc:
                    with self._prefetch_lock:
                        self._prefetch_errors[k] = exc
                finally:
                    ev = events[k]
                    ev.set()
                    with self._prefetch_lock:
                        if self._prefetch_pending.get(k) is ev:
                            self._prefetch_pending.pop(k, None)

        if wanted:
            thread = threading.Thread(
                target=worker, name="odinlink-prefetch", daemon=True
            )
            thread.start()
        else:
            thread = None
        try:
            yield
        finally:
            if thread is not None:
                thread.join()
            with self._prefetch_lock:
                for k in wanted:
                    self._prefetch.pop(k, None)
                    self._prefetch_errors.pop(k, None)
                    ev = events.get(k)
                    if ev is not None and self._prefetch_pending.get(k) is ev:
                        self._prefetch_pending.pop(k, None)
