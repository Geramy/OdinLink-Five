#!/usr/bin/env python3
"""
TB-Bridge client — runs on the training host (Linux + CUDA).

Pushes/pulls PyTorch tensors to a tb_bridge_server. Numpy is the only hard
dependency; torch is optional (used for zero-copy tensor view if present).

Example:
    import torch
    from bridge.tb_bridge_client import TBBridgeClient

    cli = TBBridgeClient(host="10.0.0.2", port=29800)   # mac TB-net IP
    x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
    cli.put("layer.42.attn", x)                          # offload
    y = cli.get("layer.42.attn", device="cuda")          # fetch back
    cli.delete("layer.42.attn")

Designed to be the offload-tier piece of a "Linux trains, Mac holds
spilled VRAM over TB5" topology. Not RDMA — uses ordinary sockets over
TB-net IP (which Apple exposes natively over Thunderbolt). Real bandwidth
on TB5 IP is ~25-40 Gbps, latency 50-200 µs per round-trip.
"""
import json
import socket
import struct
import threading
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

OP_PUT = 1
OP_GET = 2
OP_DEL = 3
OP_LIST = 4
OP_STAT = 5

_TORCH_TO_NUMPY_DTYPE = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "uint16",  # numpy doesn't have bf16; carry bytes as u16
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
}


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining}/{n} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_exact(sock: socket.socket, buf):
    view = memoryview(buf)
    while view:
        sent = sock.send(view)
        view = view[sent:]


class TBBridgeClient:
    """One TCP connection, reused across operations.

    The server keeps the socket open until the client closes. A dead
    socket is dropped and the next call reconnects once.
    """

    def __init__(self, host: str, port: int = 29800, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _close_unlocked(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
        except OSError:
            pass
        sock.connect((self.host, self.port))
        return sock

    def _ensure_unlocked(self) -> socket.socket:
        if self._sock is None:
            self._sock = self._connect()
        return self._sock

    def _call(self, fn):
        """Run fn(sock). Reconnect once if the cached socket is dead."""
        with self._lock:
            try:
                return fn(self._ensure_unlocked())
            except (OSError, ConnectionError):
                self._close_unlocked()
                return fn(self._ensure_unlocked())

    def put(self, key: str, tensor) -> None:
        """Offload a tensor. Works with torch.Tensor or numpy.ndarray.
        For torch tensors on GPU, this triggers a host copy (no avoiding it
        without GPU-aware RDMA)."""
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            arr_bytes, meta = self._torch_to_bytes(tensor)
        elif isinstance(tensor, np.ndarray):
            arr_bytes = tensor.tobytes()
            meta = {
                "kind": "numpy",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        else:
            raise TypeError(f"unsupported tensor type: {type(tensor)}")

        try:
            from bridge.odl_compress import maybe_compress, looks_compressed
        except ImportError:
            from odl_compress import maybe_compress, looks_compressed

        packed = maybe_compress(arr_bytes)
        if looks_compressed(packed):
            meta["odlc"] = True
            meta["odlc_algo"] = "lz4_block"
            meta["raw_bytes"] = len(arr_bytes)
        arr_bytes = packed

        meta_bytes = json.dumps(meta).encode("utf-8")
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 256:
            raise ValueError("key too long (max 256 bytes UTF-8)")

        hdr = (
            bytes([OP_PUT])
            + struct.pack(">I", len(key_bytes))
            + key_bytes
            + struct.pack(">I", len(meta_bytes))
            + meta_bytes
            + struct.pack(">Q", len(arr_bytes))
        )

        def do(sock):
            _send_exact(sock, hdr)
            _send_exact(sock, arr_bytes)
            status = _recv_exact(sock, 1)[0]
            if status != 0:
                raise RuntimeError(f"server returned status {status}")

        self._call(do)

    def get(self, key: str, device: Optional[str] = None):
        """Fetch a previously-put tensor. Returns the same kind it was
        stored as (torch on torch, numpy on numpy). For torch, an optional
        device= moves the result."""
        key_bytes = key.encode("utf-8")
        hdr = bytes([OP_GET]) + struct.pack(">I", len(key_bytes)) + key_bytes

        def do(sock):
            _send_exact(sock, hdr)
            status = _recv_exact(sock, 1)[0]
            if status != 0:
                raise KeyError(f"key {key!r} not found on server (status={status})")
            meta_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
            meta = json.loads(_recv_exact(sock, meta_len))
            data_len = struct.unpack(">Q", _recv_exact(sock, 8))[0]
            data = _recv_exact(sock, data_len)
            return meta, data

        meta, data = self._call(do)

        try:
            from bridge.odl_compress import maybe_decompress
        except ImportError:
            from odl_compress import maybe_decompress

        data = maybe_decompress(data)

        if meta["kind"] == "torch":
            return self._bytes_to_torch(data, meta, device=device)
        elif meta["kind"] == "numpy":
            arr = np.frombuffer(data, dtype=meta["dtype"]).reshape(meta["shape"])
            return arr.copy()  # detach from the recv buffer
        else:
            raise ValueError(f"unknown tensor kind {meta.get('kind')!r}")

    def delete(self, key: str) -> bool:
        key_bytes = key.encode("utf-8")
        hdr = bytes([OP_DEL]) + struct.pack(">I", len(key_bytes)) + key_bytes

        def do(sock):
            _send_exact(sock, hdr)
            return _recv_exact(sock, 1)[0] == 0

        return self._call(do)

    def list(self) -> Sequence[str]:
        def do(sock):
            _send_exact(sock, bytes([OP_LIST]) + struct.pack(">I", 0))
            status = _recv_exact(sock, 1)[0]
            if status != 0:
                raise RuntimeError(f"LIST failed: {status}")
            n = struct.unpack(">I", _recv_exact(sock, 4))[0]
            keys = []
            for _ in range(n):
                kl = struct.unpack(">I", _recv_exact(sock, 4))[0]
                keys.append(_recv_exact(sock, kl).decode("utf-8"))
            return keys

        return self._call(do)

    def stat(self):
        def do(sock):
            _send_exact(sock, bytes([OP_STAT]) + struct.pack(">I", 0))
            status = _recv_exact(sock, 1)[0]
            if status != 0:
                raise RuntimeError(f"STAT failed: {status}")
            total, n = struct.unpack(">QI", _recv_exact(sock, 12))
            return {"total_bytes": total, "num_keys": n}

        return self._call(do)

    # ── torch ↔ bytes helpers ────────────────────────────────────────────
    def _torch_to_bytes(self, t):
        # Move to CPU and contiguous before serializing
        cpu = t.detach().contiguous().cpu()
        # bf16 carried as raw u16 bytes; numpy doesn't have it
        if cpu.dtype == torch.bfloat16:
            buf = cpu.view(torch.uint16).numpy().tobytes()
        else:
            buf = cpu.numpy().tobytes()
        meta = {
            "kind": "torch",
            "dtype": str(t.dtype),
            "shape": list(t.shape),
        }
        return buf, meta

    def _bytes_to_torch(self, data: bytes, meta: dict, device: Optional[str] = None):
        if not _HAS_TORCH:
            raise RuntimeError("torch not installed but stored tensor is a torch tensor")
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        if dtype_str == "torch.bfloat16":
            arr = np.frombuffer(data, dtype="uint16").reshape(shape).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            np_dtype = _TORCH_TO_NUMPY_DTYPE.get(dtype_str)
            if np_dtype is None:
                raise ValueError(f"unsupported dtype {dtype_str}")
            arr = np.frombuffer(data, dtype=np_dtype).reshape(shape).copy()
            t = torch.from_numpy(arr)
            # Restore the original torch dtype (covers cases where torch and
            # numpy share a bit-level representation but distinct dtype objects)
            torch_dtype = getattr(torch, dtype_str.removeprefix("torch."))
            if t.dtype != torch_dtype:
                t = t.to(torch_dtype)
        if device:
            t = t.to(device)
        return t


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="TB-Bridge client CLI")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=29800)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("stat")
    d = sub.add_parser("delete"); d.add_argument("key")
    args = ap.parse_args()
    cli = TBBridgeClient(args.host, args.port)
    try:
        if args.cmd == "list":
            for k in cli.list():
                print(k)
        elif args.cmd == "stat":
            s = cli.stat()
            print(f"{s['total_bytes']/1e9:.2f} GB across {s['num_keys']} keys")
        elif args.cmd == "delete":
            ok = cli.delete(args.key)
            print(f"deleted: {ok}")
            sys.exit(0 if ok else 1)
    finally:
        cli.close()
