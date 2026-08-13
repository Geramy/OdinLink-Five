#!/usr/bin/env python3
"""ODLC lz4_block codec tests. No Thunderbolt, no CUDA.

    python3 compress/tests/test_odl_compress.py
"""
from __future__ import annotations

import os
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from bridge.odl_compress import (  # noqa: E402
    ALGO_GDEFLATE,
    ALGO_LZ4_BLOCK,
    HDR_FMT,
    MAGIC,
    VERSION,
    backend_name,
    compress,
    decompress,
    looks_compressed,
    maybe_compress,
    maybe_decompress,
    should_compress,
)


class FormatTests(unittest.TestCase):
    def test_reject_empty(self):
        self.assertIsNone(compress(b""))
        self.assertFalse(looks_compressed(b""))
        self.assertFalse(looks_compressed(b"not-odlc"))

    def test_reject_gdeflate_header(self):
        hdr = struct.pack(HDR_FMT, MAGIC, VERSION, ALGO_GDEFLATE, 8, 8, 1, 0)
        blob = hdr + b"\x00" * 8
        with self.assertRaises(ValueError):
            decompress(blob)


@unittest.skipIf(backend_name() == "none", "no LZ4 backend on this machine")
class RoundTripTests(unittest.TestCase):
    def test_backend_named(self):
        self.assertNotEqual(backend_name(), "none")

    def test_repeated_pattern(self):
        src = bytes([i & 0x3F for i in range(200_000)])
        wire = compress(src)
        self.assertIsNotNone(wire)
        self.assertTrue(looks_compressed(wire))
        self.assertLess(len(wire), len(src))
        magic, ver, algo, orig, comp, nchunks, _ = struct.unpack(
            HDR_FMT, wire[: struct.calcsize(HDR_FMT)]
        )
        self.assertEqual(magic, MAGIC)
        self.assertEqual(ver, VERSION)
        self.assertEqual(algo, ALGO_LZ4_BLOCK)
        self.assertEqual(orig, len(src))
        self.assertGreater(nchunks, 1)
        self.assertEqual(decompress(wire), src)

    def test_maybe_helpers(self):
        src = bytes([7]) * 300_000
        packed = maybe_compress(src)
        self.assertTrue(looks_compressed(packed))
        self.assertEqual(maybe_decompress(packed), src)
        raw = b"hello"
        self.assertEqual(maybe_decompress(raw), raw)

    def test_threshold_env(self):
        old = os.environ.get("ODL_COMPRESS")
        old_t = os.environ.get("ODL_COMPRESS_THRESHOLD")
        try:
            os.environ["ODL_COMPRESS"] = "0"
            self.assertFalse(should_compress(10_000_000))
            os.environ["ODL_COMPRESS"] = "1"
            os.environ["ODL_COMPRESS_THRESHOLD"] = "100"
            self.assertTrue(should_compress(200))
        finally:
            if old is None:
                os.environ.pop("ODL_COMPRESS", None)
            else:
                os.environ["ODL_COMPRESS"] = old
            if old_t is None:
                os.environ.pop("ODL_COMPRESS_THRESHOLD", None)
            else:
                os.environ["ODL_COMPRESS_THRESHOLD"] = old_t


class CInteropTests(unittest.TestCase):
    def test_python_reads_c_blob(self):
        so = Path(ROOT) / "build" / "compress" / "libodl_compress.so"
        if not so.is_file():
            so = Path(ROOT) / "build" / "compress" / "libodl_compress.dylib"
        if not so.is_file():
            self.skipTest("libodl_compress not built")
        import bridge.odl_compress as m
        old_load = m._load_odl_so
        os.environ["ODL_COMPRESS_LIB"] = str(so)
        try:
            src = bytes([i & 0x3F for i in range(80_000)])
            wire = m.compress(src)
            self.assertIsNotNone(wire)
            m._load_odl_so = lambda: None
            self.assertEqual(m.decompress(wire), src)
        finally:
            m._load_odl_so = old_load
            os.environ.pop("ODL_COMPRESS_LIB", None)


@unittest.skipIf(backend_name() == "none", "no LZ4 backend on this machine")
class BridgeTests(unittest.TestCase):
    def test_put_stores_odlc_get_roundtrips(self):
        import socket
        import threading
        import numpy as np
        from bridge.tb_bridge_server import TensorStore, handle_client
        from bridge.tb_bridge_client import TBBridgeClient

        store = TensorStore(max_bytes=8 << 20)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        port = srv.getsockname()[1]
        stop = threading.Event()

        def loop():
            srv.settimeout(0.2)
            while not stop.is_set():
                try:
                    client, addr = srv.accept()
                except (socket.timeout, OSError):
                    continue
                handle_client(client, addr, store, False)

        threading.Thread(target=loop, daemon=True).start()
        old_t = os.environ.get("ODL_COMPRESS_THRESHOLD")
        os.environ["ODL_COMPRESS_THRESHOLD"] = "1000"
        try:
            cli = TBBridgeClient("127.0.0.1", port, timeout=5)
            x = np.zeros(40_000, dtype=np.float32)
            x[:50] = np.arange(50)
            cli.put("kv.0", x)
            meta, data = store.get("kv.0")
            self.assertTrue(looks_compressed(data))
            self.assertLess(len(data), x.nbytes)
            y = cli.get("kv.0")
            np.testing.assert_array_equal(x, y)
            viewed = store.view("kv.0")
            np.testing.assert_array_equal(np.asarray(viewed), x)
        finally:
            stop.set()
            srv.close()
            if old_t is None:
                os.environ.pop("ODL_COMPRESS_THRESHOLD", None)
            else:
                os.environ["ODL_COMPRESS_THRESHOLD"] = old_t


if __name__ == "__main__":
    print("lz4 backend:", backend_name())
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)
