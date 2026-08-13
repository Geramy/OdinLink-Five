"""ODLC + portable LZ4-block codec (the Mac-readable format).

Matches ``compress/include/odl_tb5/odl_compress.h`` algo 4:

    [32-byte LE header][num_chunks × {raw,comp} u32][LZ4 raw blocks]

nvCOMP GDeflate / batched-LZ4 / Snappy (algo 1/2/3) are rejected here.
A Mac cannot decode those.

Backends, first one that works:
  1. ``lz4.block`` (pip / python3-lz4)
  2. Apple ``libcompression`` (COMPRESSION_LZ4_RAW) — always on macOS
  3. ``liblz4`` via ctypes
  4. ``libodl_compress`` via ctypes (``odl_compress_host``)

Env (same names as the C library):
  ODL_COMPRESS=0          disable
  ODL_COMPRESS_THRESHOLD  default 262144
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import sys
from typing import Optional

MAGIC = 0x4F444C43
VERSION = 1
ALGO_GDEFLATE = 1
ALGO_LZ4_NVCOMP = 2
ALGO_SNAPPY = 3
ALGO_LZ4_BLOCK = 4
CHUNK = 65536
HDR_FMT = "<IHHQQII"
HDR_SIZE = struct.calcsize(HDR_FMT)
CHUNK_FMT = "<II"
CHUNK_SIZE = struct.calcsize(CHUNK_FMT)

# Apple compression.h
_COMPRESSION_LZ4_RAW = 0x101


def _env_off() -> bool:
    v = os.environ.get("ODL_COMPRESS")
    if v is None or v == "":
        return False
    return v in ("0", "false", "False", "off", "OFF", "no", "NO")


def threshold() -> int:
    raw = os.environ.get("ODL_COMPRESS_THRESHOLD", "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 262144


def should_compress(n: int) -> bool:
    if _env_off():
        return False
    return n >= threshold()


def looks_compressed(buf: bytes) -> bool:
    if len(buf) < HDR_SIZE:
        return False
    magic, ver, algo, orig, comp, nchunks, _res = struct.unpack(
        HDR_FMT, buf[:HDR_SIZE]
    )
    return (
        magic == MAGIC
        and ver == VERSION
        and orig > 0
        and comp > 0
        and nchunks > 0
        and algo == ALGO_LZ4_BLOCK
    )


class _Backend:
    name = "none"

    def compress(self, src: bytes) -> bytes:
        raise NotImplementedError

    def decompress(self, src: bytes, dst_len: int) -> bytes:
        raise NotImplementedError


class _Lz4Block(_Backend):
    name = "lz4.block"

    def __init__(self, mod):
        self._mod = mod

    def compress(self, src: bytes) -> bytes:
        return self._mod.compress(src, store_size=False)

    def decompress(self, src: bytes, dst_len: int) -> bytes:
        return self._mod.decompress(src, uncompressed_size=dst_len)


class _CtypesLz4(_Backend):
    name = "liblz4"

    def __init__(self, lib):
        self._lib = lib
        lib.LZ4_compress_default.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int
        ]
        lib.LZ4_compress_default.restype = ctypes.c_int
        lib.LZ4_decompress_safe.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int
        ]
        lib.LZ4_decompress_safe.restype = ctypes.c_int
        lib.LZ4_compressBound.argtypes = [ctypes.c_int]
        lib.LZ4_compressBound.restype = ctypes.c_int

    def compress(self, src: bytes) -> bytes:
        bound = self._lib.LZ4_compressBound(len(src))
        dst = ctypes.create_string_buffer(bound)
        n = self._lib.LZ4_compress_default(src, dst, len(src), bound)
        if n <= 0:
            raise RuntimeError("LZ4_compress_default failed")
        return dst.raw[:n]

    def decompress(self, src: bytes, dst_len: int) -> bytes:
        dst = ctypes.create_string_buffer(dst_len)
        n = self._lib.LZ4_decompress_safe(src, dst, len(src), dst_len)
        if n != dst_len:
            raise RuntimeError("LZ4_decompress_safe failed")
        return dst.raw[:n]


class _AppleCompression(_Backend):
    name = "libcompression"

    def __init__(self, lib):
        self._lib = lib
        lib.compression_encode_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_int,
        ]
        lib.compression_encode_buffer.restype = ctypes.c_size_t
        lib.compression_decode_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_int,
        ]
        lib.compression_decode_buffer.restype = ctypes.c_size_t

    def compress(self, src: bytes) -> bytes:
        dst = ctypes.create_string_buffer(len(src) + (len(src) // 16) + 64)
        n = self._lib.compression_encode_buffer(
            dst, len(dst), src, len(src), None, _COMPRESSION_LZ4_RAW
        )
        if n == 0:
            raise RuntimeError("compression_encode_buffer failed")
        return dst.raw[:n]

    def decompress(self, src: bytes, dst_len: int) -> bytes:
        dst = ctypes.create_string_buffer(dst_len)
        n = self._lib.compression_decode_buffer(
            dst, dst_len, src, len(src), None, _COMPRESSION_LZ4_RAW
        )
        if n != dst_len:
            raise RuntimeError("compression_decode_buffer failed")
        return dst.raw[:n]


_backend: Optional[_Backend] = None
_backend_tried = False


def _load_odl_so():
    names = []
    env = os.environ.get("ODL_COMPRESS_LIB")
    if env:
        names.append(env)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in (
        "build/compress/libodl_compress.so",
        "build/libodl_compress.so",
        "build/compress/libodl_compress.dylib",
        "build/libodl_compress.dylib",
    ):
        names.append(os.path.join(here, rel))
    for path in names:
        if os.path.isfile(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None


class _OdlLib(_Backend):
    """Whole-blob host API from libodl_compress — used as last encode path."""

    name = "libodl_compress"

    def __init__(self, lib):
        self._lib = lib
        lib.odl_compress_host.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.odl_compress_host.restype = ctypes.c_int
        lib.odl_decompress_host.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.odl_decompress_host.restype = ctypes.c_int
        lib.odl_compress_host_max_wire_bytes.argtypes = [ctypes.c_size_t]
        lib.odl_compress_host_max_wire_bytes.restype = ctypes.c_size_t

    def wrap_compress(self, src: bytes) -> Optional[bytes]:
        cap = self._lib.odl_compress_host_max_wire_bytes(len(src))
        dst = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_size_t()
        rc = self._lib.odl_compress_host(src, len(src), dst, cap, ctypes.byref(out_n))
        if rc != 0:
            return None
        return dst.raw[: out_n.value]

    def wrap_decompress(self, wire: bytes) -> bytes:
        magic, ver, algo, orig, comp, nchunks, _res = struct.unpack(
            HDR_FMT, wire[:HDR_SIZE]
        )
        del magic, ver, algo, comp, nchunks
        dst = ctypes.create_string_buffer(orig)
        out_n = ctypes.c_size_t()
        rc = self._lib.odl_decompress_host(
            wire, len(wire), dst, orig, ctypes.byref(out_n)
        )
        if rc != 0:
            raise RuntimeError("odl_decompress_host failed")
        return dst.raw[: out_n.value]


def _pick_backend() -> Optional[_Backend]:
    global _backend, _backend_tried
    if _backend_tried:
        return _backend
    _backend_tried = True

    try:
        import lz4.block as lb  # type: ignore
        _backend = _Lz4Block(lb)
        return _backend
    except ImportError:
        pass

    if sys.platform == "darwin":
        for path in ("/usr/lib/libcompression.dylib", "libcompression.dylib"):
            try:
                _backend = _AppleCompression(ctypes.CDLL(path))
                return _backend
            except OSError:
                continue

    for cand in (
        ctypes.util.find_library("lz4"),
        "liblz4.so.1",
        "liblz4.so",
        "liblz4.dylib",
    ):
        if not cand:
            continue
        try:
            _backend = _CtypesLz4(ctypes.CDLL(cand))
            return _backend
        except OSError:
            continue

    return _backend


def backend_name() -> str:
    b = _pick_backend()
    if b:
        return b.name
    so = _load_odl_so()
    return "libodl_compress" if so else "none"


def _header(orig: int, comp: int, nchunks: int) -> bytes:
    return struct.pack(
        HDR_FMT, MAGIC, VERSION, ALGO_LZ4_BLOCK, orig, comp, nchunks, 0
    )


def compress(data: bytes) -> Optional[bytes]:
    """Return an ODLC blob smaller than ``data``, or None to keep raw."""
    if not data:
        return None
    so = _load_odl_so()
    if so is not None:
        try:
            wrapped = _OdlLib(so).wrap_compress(data)
            if wrapped and len(wrapped) < len(data) and looks_compressed(wrapped):
                return wrapped
        except (OSError, RuntimeError):
            pass

    be = _pick_backend()
    if be is None:
        return None

    n = len(data)
    nchunks = (n + CHUNK - 1) // CHUNK
    table = bytearray()
    blocks = bytearray()
    for i in range(nchunks):
        chunk = data[i * CHUNK : (i + 1) * CHUNK]
        try:
            c = be.compress(chunk)
        except (ValueError, RuntimeError):
            return None
        if not c:
            return None
        table += struct.pack(CHUNK_FMT, len(chunk), len(c))
        blocks += c
    payload = bytes(table) + bytes(blocks)
    if HDR_SIZE + len(payload) >= n:
        return None
    return _header(n, len(payload), nchunks) + payload


def decompress(wire: bytes) -> bytes:
    """Expand an ODLC lz4_block blob. Raises on nvCOMP or corrupt input."""
    if len(wire) < HDR_SIZE:
        raise ValueError("truncated ODLC header")
    magic, ver, algo, orig, comp, nchunks, _res = struct.unpack(
        HDR_FMT, wire[:HDR_SIZE]
    )
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not ODLC")
    if algo != ALGO_LZ4_BLOCK:
        raise ValueError(
            f"ODLC algo={algo} is nvCOMP-native; Mac/host cannot decode it "
            "(use lz4_block)"
        )
    if orig == 0 or nchunks == 0:
        raise ValueError("empty ODLC")
    table_n = nchunks * CHUNK_SIZE
    if HDR_SIZE + comp > len(wire) or comp < table_n:
        raise ValueError("truncated ODLC payload")

    so = _load_odl_so()
    if so is not None:
        try:
            return _OdlLib(so).wrap_decompress(wire)
        except (OSError, RuntimeError):
            pass

    be = _pick_backend()
    if be is None:
        raise RuntimeError(
            "no LZ4 backend (install python3-lz4, or build libodl_compress)"
        )

    table = wire[HDR_SIZE : HDR_SIZE + table_n]
    blocks = wire[HDR_SIZE + table_n : HDR_SIZE + comp]
    out = bytearray()
    boff = 0
    for i in range(nchunks):
        raw, csz = struct.unpack_from(CHUNK_FMT, table, i * CHUNK_SIZE)
        piece = blocks[boff : boff + csz]
        if len(piece) != csz:
            raise ValueError("truncated LZ4 block")
        out += be.decompress(piece, raw)
        boff += csz
    if len(out) != orig:
        raise ValueError("decompressed size mismatch")
    return bytes(out)


def maybe_compress(data: bytes) -> bytes:
    """Compress when it helps; otherwise return ``data`` unchanged."""
    if not should_compress(len(data)):
        return data
    out = compress(data)
    return out if out is not None else data


def maybe_decompress(data: bytes) -> bytes:
    """Expand ODLC lz4_block; pass other buffers through."""
    if looks_compressed(data):
        return decompress(data)
    return data
