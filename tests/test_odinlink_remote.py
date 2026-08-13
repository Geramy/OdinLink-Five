#!/usr/bin/env python3
"""Localhost tests for odinlink.RemoteStore and the TB-bridge.

No Thunderbolt cable. Starts tb_bridge_server on 127.0.0.1:0 and talks
to it with numpy tensors. mlx is optional — wrap_blob falls back to
NumPy when mlx is not installed.

    python3 tests/test_odinlink_remote.py
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge.mlx_helpers import HAS_MLX, as_numpy, wrap_blob, wrap_store
from bridge.tb_bridge_client import TBBridgeClient
from bridge.tb_bridge_server import TensorStore, handle_client
from odinlink import RemoteStore
from odinlink.remote import _parse_url


class _LocalServer:
    def __init__(self, max_bytes: int = 32 << 20):
        self.store = TensorStore(max_bytes=max_bytes)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                client, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=handle_client,
                args=(client, addr, self.store, False),
                daemon=True,
            ).start()

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class ParseUrlTests(unittest.TestCase):
    def test_bare_host(self):
        self.assertEqual(_parse_url("192.168.167.2"), ("192.168.167.2", 29800))

    def test_tb5_scheme(self):
        self.assertEqual(_parse_url("tb5://10.0.0.2"), ("10.0.0.2", 29800))

    def test_explicit_port(self):
        self.assertEqual(_parse_url("tb5://10.0.0.2:1234"), ("10.0.0.2", 1234))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            _parse_url("")


class BridgeClientTests(unittest.TestCase):
    def setUp(self):
        self.srv = _LocalServer()
        self.cli = TBBridgeClient("127.0.0.1", self.srv.port, timeout=5.0)

    def tearDown(self):
        self.cli.close()
        self.srv.close()

    def test_numpy_roundtrip(self):
        x = np.arange(24, dtype=np.float32).reshape(3, 8)
        self.cli.put("layer.0", x)
        y = self.cli.get("layer.0")
        np.testing.assert_array_equal(x, y)
        self.assertIn("layer.0", self.cli.list())
        info = self.cli.stat()
        self.assertEqual(info["num_keys"], 1)
        self.assertEqual(info["total_bytes"], x.nbytes)
        self.assertTrue(self.cli.delete("layer.0"))
        self.assertEqual(list(self.cli.list()), [])

    def test_missing_get(self):
        with self.assertRaises(KeyError):
            self.cli.get("nope")

    def test_reuses_one_socket(self):
        x = np.ones(8, dtype=np.float32)
        self.cli.put("a", x)
        first = self.cli._sock
        self.assertIsNotNone(first)
        self.cli.put("b", x)
        self.cli.get("a")
        self.cli.stat()
        self.cli.list()
        self.assertIs(self.cli._sock, first)

    def test_reconnects_after_drop(self):
        x = np.ones(4, dtype=np.float32)
        self.cli.put("a", x)
        dead = self.cli._sock
        self.assertIsNotNone(dead)
        dead.close()
        y = self.cli.get("a")
        np.testing.assert_array_equal(x, y)
        self.assertIsNotNone(self.cli._sock)
        self.assertIsNot(self.cli._sock, dead)


class RemoteStoreTests(unittest.TestCase):
    def setUp(self):
        self.srv = _LocalServer()
        self.mac = RemoteStore(f"tb5://127.0.0.1:{self.srv.port}", timeout=5.0)

    def tearDown(self):
        self.mac.close()
        self.srv.close()

    def test_put_get_delete(self):
        x = np.linspace(0, 1, 16, dtype=np.float32)
        self.assertEqual(self.mac.put("kv.42", x), "kv.42")
        y = self.mac.get("kv.42")
        np.testing.assert_array_equal(x, y)
        self.assertTrue(self.mac.delete("kv.42"))
        self.assertEqual(self.mac.stat()["num_keys"], 0)

    def test_prefetch_window_serves_cache(self):
        a = np.arange(8, dtype=np.float32)
        b = np.arange(8, 16, dtype=np.float32)
        self.mac.put("layer.39", a)
        self.mac.put("layer.40", b)
        with self.mac.prefetch_window(["layer.39", "layer.40"]):
            deadline = time.time() + 5
            while time.time() < deadline:
                with self.mac._prefetch_lock:
                    pending = bool(self.mac._prefetch_pending)
                if not pending:
                    break
                time.sleep(0.01)
            else:
                self.fail("prefetch did not finish")
            # Worker has both keys locally; a wire delete must not
            # break the still-cached get.
            self.assertTrue(self.mac._cli.delete("layer.40"))
            got_a = self.mac.get("layer.39")
            got_b = self.mac.get("layer.40")
        np.testing.assert_array_equal(got_a, a)
        np.testing.assert_array_equal(got_b, b)

    def test_prefetch_missing_raises(self):
        with self.mac.prefetch_window(["ghost"]):
            with self.assertRaises(KeyError):
                self.mac.get("ghost")

    def test_empty_prefetch_window(self):
        with self.mac.prefetch_window([]):
            pass


class MlxHelpersTests(unittest.TestCase):
    def test_wrap_blob_numpy_fallback(self):
        meta = {"kind": "numpy", "dtype": "float32", "shape": [2, 2]}
        data = np.array([[1, 2], [3, 4]], dtype=np.float32).tobytes()
        arr = wrap_blob(meta, data)
        self.assertEqual(tuple(arr.shape), (2, 2))
        np.testing.assert_array_equal(as_numpy(meta, data),
                                      np.array([[1, 2], [3, 4]], dtype=np.float32))

    def test_wrap_store_and_tensorstore_view(self):
        store = TensorStore(max_bytes=1 << 20)
        meta = b'{"kind":"numpy","dtype":"float32","shape":[3]}'
        data = np.array([1, 2, 3], dtype=np.float32).tobytes()
        self.assertEqual(store.put("w", meta, data), 0)
        views = wrap_store(store)
        self.assertIn("w", views)
        np.testing.assert_array_equal(np.asarray(views["w"]),
                                      np.array([1, 2, 3], dtype=np.float32))
        viewed = store.view("w")
        np.testing.assert_array_equal(np.asarray(viewed),
                                      np.array([1, 2, 3], dtype=np.float32))

    def test_mlx_flag_is_bool(self):
        self.assertIsInstance(HAS_MLX, bool)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)
