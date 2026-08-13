#!/usr/bin/env python3
"""Wrap TB-bridge stored tensors as MLX arrays (Mac-side).

The server keeps each key as (meta_json, raw_bytes) in unified memory.
On Apple Silicon, MLX can view those pages as an ``mlx.array`` when
``copy=False`` is honoured. mlx is optional: without it, callers get a
NumPy view of the same bytes (still zero-copy from the store).

    from bridge.mlx_helpers import wrap_blob, wrap_store
    arr = wrap_blob(meta, data)          # mlx.array or numpy.ndarray
    views = wrap_store(server_store)     # {key: array}
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Union

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    HAS_MLX = False

_NUMPY_DTYPE = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "uint16",
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
    "bfloat16": "uint16",
}

_MLX_DTYPE_NAME = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
    "uint16": "uint16",
}


def _parse_meta(meta: Union[bytes, bytearray, Mapping[str, Any]]) -> dict:
    if isinstance(meta, (bytes, bytearray)):
        return json.loads(meta)
    return dict(meta)


def as_numpy(meta: Union[bytes, bytearray, Mapping[str, Any]],
             data: bytes) -> np.ndarray:
    """Zero-copy NumPy view of a stored blob (bf16 is uint16 bits)."""
    m = _parse_meta(meta)
    dtype_str = str(m["dtype"])
    shape = tuple(m["shape"])
    np_dtype = _NUMPY_DTYPE.get(dtype_str, dtype_str)
    return np.frombuffer(data, dtype=np_dtype).reshape(shape)


def _mlx_array(arr: np.ndarray, mlx_dtype):
    """Prefer a shared-memory wrap; fall back to a copy on older mlx."""
    kwargs = {}
    if mlx_dtype is not None:
        kwargs["dtype"] = mlx_dtype
    try:
        return mx.array(arr, copy=False, **kwargs)
    except TypeError:
        try:
            return mx.array(arr, **kwargs)
        except TypeError:
            return mx.array(arr)
    except ValueError:
        if kwargs:
            try:
                return mx.array(arr, **kwargs)
            except TypeError:
                pass
        return mx.array(arr)


def as_mlx(meta: Union[bytes, bytearray, Mapping[str, Any]], data: bytes):
    """``mlx.array`` sharing ``data`` when mlx allows ``copy=False``.

    torch.bfloat16 is stored as raw uint16 bits. We bitcast via
    ``.view(mx.bfloat16)`` when that exists; otherwise the uint16 wrap
    is returned so the caller still sees the same bits.
    """
    if not HAS_MLX:
        raise RuntimeError("mlx is not installed")
    m = _parse_meta(meta)
    dtype_str = str(m["dtype"])
    arr = as_numpy(m, data)
    mlx_name = _MLX_DTYPE_NAME.get(dtype_str, dtype_str.replace("torch.", ""))
    mlx_dtype = getattr(mx, mlx_name, None)

    if dtype_str in ("torch.bfloat16", "bfloat16"):
        wrapped = _mlx_array(arr, getattr(mx, "uint16", None))
        view = getattr(wrapped, "view", None)
        if callable(view) and hasattr(mx, "bfloat16"):
            try:
                return view(mx.bfloat16)
            except (TypeError, ValueError):
                pass
        return wrapped

    return _mlx_array(arr, mlx_dtype)


def wrap_blob(meta: Union[bytes, bytearray, Mapping[str, Any]],
              data: bytes):
    """MLX array when mlx is present, otherwise a NumPy view."""
    if HAS_MLX:
        return as_mlx(meta, data)
    return as_numpy(meta, data)


def wrap_store(store) -> dict:
    """Wrap every entry in a ``TensorStore`` (or anything with keys/get)."""
    out = {}
    for key in store.keys():
        entry = store.get(key)
        if entry is None:
            continue
        meta, data = entry
        out[key] = wrap_blob(meta, data)
    return out
