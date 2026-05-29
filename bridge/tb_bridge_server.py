#!/usr/bin/env python3
"""
TB-Bridge server — runs on the Mac (or any host with extra RAM/VRAM).

Holds tensor blobs in unified memory and serves push/pull over a length-
prefixed TCP protocol. Intended to run on a Mac across a Thunderbolt 5
cable from the training host; the TB-net link gives 25-40 Gbps at the
IP layer without any kernel-driver work.

Why this exists: RDMA-level Mac↔Linux interop in OdinLink-Five still
requires Apple-protocol XDomain login work that hasn't been verified
against real hardware. This bridge is the userspace fallback that
works today — slower than RDMA but immediately usable.

Wire format (per request, all big-endian on the wire):
    u8   op     (PUT=1, GET=2, DEL=3, LIST=4, STAT=5)
    u32  key_len
    u8[] key      (UTF-8 string, ≤ 256 bytes)
    For PUT:
        u32  meta_len
        u8[] meta_json   (dtype, shape, device, etc.)
        u64  data_len
        u8[] data        (raw tensor bytes)
    Response:
        u8   status (0=OK, nonzero=err)
        For GET: same payload trailer as PUT
        For LIST: u32 count + count×(u32 keylen + keystr)
        For STAT: u64 total_bytes + u32 num_keys

Local storage: a flat dict {key: (meta_bytes, data_bytes)}. Memory is
the OS-managed mapped bytes — on Apple Silicon this is unified memory
visible to Metal, so Mac-side training code can mlx-import the buffer
zero-copy if desired (see bridge/mlx_helpers.py — TODO).
"""
import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Dict, Tuple

OP_PUT = 1
OP_GET = 2
OP_DEL = 3
OP_LIST = 4
OP_STAT = 5

STATUS_OK = 0
STATUS_NOT_FOUND = 1
STATUS_BAD_OP = 2
STATUS_OOM = 3
STATUS_PROTOCOL = 4


class TensorStore:
    """In-memory tensor blob store. Thread-safe."""

    def __init__(self, max_bytes: int = 64 * (1 << 30)):
        self._store: Dict[str, Tuple[bytes, bytes]] = {}
        self._bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def put(self, key: str, meta: bytes, data: bytes) -> int:
        with self._lock:
            old = self._store.get(key)
            new_bytes = self._bytes - (len(old[1]) if old else 0) + len(data)
            if new_bytes > self._max_bytes:
                return STATUS_OOM
            self._store[key] = (meta, data)
            self._bytes = new_bytes
            return STATUS_OK

    def get(self, key: str):
        with self._lock:
            return self._store.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            old = self._store.pop(key, None)
            if old is None:
                return False
            self._bytes -= len(old[1])
            return True

    def keys(self):
        with self._lock:
            return list(self._store.keys())

    def stat(self):
        with self._lock:
            return self._bytes, len(self._store)


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


def _send_exact(sock: socket.socket, buf: memoryview):
    view = memoryview(buf)
    while view:
        sent = sock.send(view)
        view = view[sent:]


def handle_client(sock: socket.socket, addr, store: TensorStore, verbose: bool):
    """One client = one request/response cycle, then close. Simple is fast."""
    try:
        op = _recv_exact(sock, 1)[0]
        key_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
        if key_len > 256:
            sock.sendall(bytes([STATUS_PROTOCOL]))
            return
        key = _recv_exact(sock, key_len).decode("utf-8")

        if op == OP_PUT:
            meta_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
            meta = _recv_exact(sock, meta_len)
            data_len = struct.unpack(">Q", _recv_exact(sock, 8))[0]
            data = _recv_exact(sock, data_len)
            status = store.put(key, meta, data)
            sock.sendall(bytes([status]))
            if verbose:
                print(f"[PUT] {key} meta={meta_len}B data={data_len/1e6:.1f}MB status={status}")

        elif op == OP_GET:
            entry = store.get(key)
            if entry is None:
                sock.sendall(bytes([STATUS_NOT_FOUND]))
                return
            meta, data = entry
            hdr = bytes([STATUS_OK]) + struct.pack(">I", len(meta)) + meta + struct.pack(">Q", len(data))
            _send_exact(sock, memoryview(hdr))
            _send_exact(sock, memoryview(data))
            if verbose:
                print(f"[GET] {key} -> {len(data)/1e6:.1f}MB")

        elif op == OP_DEL:
            ok = store.delete(key)
            sock.sendall(bytes([STATUS_OK if ok else STATUS_NOT_FOUND]))
            if verbose:
                print(f"[DEL] {key} ok={ok}")

        elif op == OP_LIST:
            keys = store.keys()
            parts = [bytes([STATUS_OK]), struct.pack(">I", len(keys))]
            for k in keys:
                kb = k.encode("utf-8")
                parts.append(struct.pack(">I", len(kb)))
                parts.append(kb)
            sock.sendall(b"".join(parts))
            if verbose:
                print(f"[LIST] -> {len(keys)} keys")

        elif op == OP_STAT:
            total, n = store.stat()
            sock.sendall(bytes([STATUS_OK]) + struct.pack(">QI", total, n))
            if verbose:
                print(f"[STAT] {total/1e9:.2f}GB across {n} keys")

        else:
            sock.sendall(bytes([STATUS_BAD_OP]))

    except (ConnectionError, OSError) as e:
        if verbose:
            print(f"[client {addr}] dropped: {e}")
    finally:
        try:
            sock.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bind", default="0.0.0.0",
                    help="Bind address. For TB-net, set to the thunderbolt0 IP "
                         "(or 0.0.0.0 to listen on all interfaces).")
    ap.add_argument("--port", type=int, default=29800)
    ap.add_argument("--max-gb", type=float, default=64.0,
                    help="Max total tensor bytes held in memory (default 64 GB).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    store = TensorStore(max_bytes=int(args.max_gb * (1 << 30)))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # Larger socket buffers help TB5 saturation
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
    except OSError:
        pass

    sock.bind((args.bind, args.port))
    sock.listen(64)
    print(f"tb_bridge_server listening on {args.bind}:{args.port}  "
          f"(max {args.max_gb} GB resident)")
    sys.stdout.flush()

    try:
        while True:
            client, addr = sock.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = threading.Thread(
                target=handle_client,
                args=(client, addr, store, args.verbose),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
