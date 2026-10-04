"""Read-only, lazy safetensors loader with strict structural validation.

No torch, safetensors or remote-code dependency. Decoded arrays own their memory,
so evicting a mapped shard cannot invalidate a caller's tensor. FP8 ``weight_scale_inv``
is the *multiplicative dequantization scale*, despite its name. Metadata-only
construction never downloads or opens weight shards.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import mmap
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np


class CheckpointError(ValueError):
    """Malformed, unsupported or inconsistent checkpoint."""


DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E4M3FN": 1}
MAX_HEADER_BYTES = 100 * 1024 * 1024


def _object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise CheckpointError(f"duplicate JSON key: {k}")
        result[k] = v
    return result


def _json(data: str | bytes) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_object,
                          parse_constant=lambda x: (_ for _ in ()).throw(CheckpointError(f"invalid JSON constant: {x}")))
    except (ValueError, UnicodeError) as e:
        raise CheckpointError(f"invalid checkpoint JSON: {e}") from e


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


# All 256 codes, including signed zero, subnormals and the two NaN codes.
# E4M3FN has exponent bias 7; exponent 15 is finite except mantissa 7.
def _fp8_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exp = ((codes >> 3) & 15).astype(np.int32)
    mant = (codes & 7).astype(np.float32)
    values = np.where(exp == 0, mant * np.float32(2.0 ** -9),
                      (np.float32(1) + mant / np.float32(8)) * np.exp2(exp - 7)).astype(np.float32)
    values = np.copysign(values, np.where(codes & 128, -1.0, 1.0))
    values[(exp == 15) & (mant == 7)] = np.nan
    values.flags.writeable = False
    return values


FP8_E4M3FN_TABLE = _fp8_table()


def decode_e4m3fn(data) -> np.ndarray:
    """Decode E4M3 finite-numbers FP8 to a new float32 array (no scale)."""
    raw = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else np.asarray(data)
    if raw.dtype != np.uint8:
        raise CheckpointError("FP8 decoder expects uint8 codes")
    return FP8_E4M3FN_TABLE[raw]


class SafetensorsFile:
    """One lazily resident read-only mapping, validated before data is decoded."""

    def __init__(self, path: str | Path, *, max_tensor_bytes: int = 8 * 1024**3):
        if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
            raise CheckpointError("max_tensor_bytes must be a positive integer")
        self.path = Path(path)
        self.max_tensor_bytes = max_tensor_bytes
        self._map = None
        self._file = None
        self.tensors: dict[str, TensorInfo] = {}
        self.metadata: dict[str, str] = {}
        try:
            self._file = self.path.open("rb")
            size = os.fstat(self._file.fileno()).st_size
            if size < 8:
                raise CheckpointError("truncated safetensors length prefix")
            length = struct.unpack("<Q", self._file.read(8))[0]
            if length < 2 or length > MAX_HEADER_BYTES or length > size - 8:
                raise CheckpointError("invalid safetensors header length")
            header_bytes = self._file.read(length)
            if not header_bytes.startswith(b"{"):
                raise CheckpointError("safetensors header must begin with '{'")
            header = _json(header_bytes)
            if not isinstance(header, dict):
                raise CheckpointError("safetensors header must be an object")
            self.data_start = 8 + length
            data_size = size - self.data_start
            intervals = []
            for name, spec in header.items():
                if name == "__metadata__":
                    if not isinstance(spec, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in spec.items()):
                        raise CheckpointError("safetensors metadata must map strings to strings")
                    self.metadata = dict(spec)
                    continue
                if not name or not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}:
                    raise CheckpointError(f"invalid tensor entry {name!r}")
                dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
                if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
                    raise CheckpointError(f"unsupported dtype {dtype!r} for {name}")
                if (not isinstance(shape, list) or len(shape) > 32
                        or any(type(d) is not int or d < 0 or d > 2**63 - 1 for d in shape)):
                    raise CheckpointError(f"invalid shape for {name}")
                if (not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(x) is not int for x in offsets)):
                    raise CheckpointError(f"invalid offsets for {name}")
                start, end = offsets
                if start < 0 or end < start or end > data_size:
                    raise CheckpointError(f"out-of-bounds offsets for {name}")
                elements = math.prod(shape)
                if elements > 2**63 - 1 or elements * DTYPE_BYTES[dtype] != end - start:
                    raise CheckpointError(f"byte count does not match shape/dtype for {name}")
                self.tensors[name] = TensorInfo(name, dtype, tuple(shape), (start, end))
                intervals.append((start, end, name))
            cursor = 0
            for start, end, name in sorted(intervals):
                if start != cursor:
                    raise CheckpointError(f"overlap or gap before {name}")
                cursor = end
            if cursor != data_size:
                raise CheckpointError("unclaimed bytes at end of safetensors file")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

    def info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as e:
            raise CheckpointError(f"tensor {name!r} not present in {self.path.name}") from e

    def tensor(self, name: str, *, rows: slice | None = None) -> np.ndarray:
        """Decode to owned float32; row slices avoid materializing entire matrices."""
        if self._map is None:
            raise CheckpointError("safetensors file is closed")
        info = self.info(name)
        shape = info.shape
        begin, end = info.data_offsets
        if rows is not None:
            row_start, row_end = _row_bounds(rows, shape)
            width = math.prod(shape[1:])
            begin += row_start * width * DTYPE_BYTES[info.dtype]
            end = begin + (row_end - row_start) * width * DTYPE_BYTES[info.dtype]
            shape = (row_end - row_start,) + shape[1:]
        if math.prod(shape) * 4 > self.max_tensor_bytes:
            raise CheckpointError(f"decoded tensor {name} exceeds max_tensor_bytes; use row slices")
        offset = self.data_start + begin
        count = math.prod(shape)
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "F8_E4M3": "u1", "F8_E4M3FN": "u1"}[info.dtype]
        raw = np.frombuffer(self._map, dtype=dtype, count=count, offset=offset)
        if info.dtype == "BF16":
            out = (raw.astype(np.uint32) << 16).view(np.float32)
        elif info.dtype.startswith("F8_"):
            out = decode_e4m3fn(raw)
        else:
            out = raw.astype(np.float32, copy=True)
        return out.reshape(shape)

    def close(self):
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _row_bounds(rows: slice, shape: tuple[int, ...]) -> tuple[int, int]:
    if not isinstance(rows, slice) or not shape:
        raise CheckpointError("rows requires a slice and a tensor with at least one dimension")
    if rows.step not in (None, 1) or type(rows.step) is bool:
        raise CheckpointError("only contiguous row slices are supported")
    start = 0 if rows.start is None else rows.start
    stop = shape[0] if rows.stop is None else rows.stop
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= shape[0]:
        raise CheckpointError("row slice is out of bounds")
    return start, stop


def dequantize_block_fp8(codes: np.ndarray, scales: np.ndarray,
                         block_size: tuple[int, int] = (128, 128), *, row_offset: int = 0) -> np.ndarray:
    """Decode a 2D FP8 matrix or contiguous row window, applying block scales.

    scales is the full tensor's grid. Each [r,c] scale multiplies one block,
    including partial edge blocks. row_offset selects the correct scale rows.
    """
    if not isinstance(block_size, (list, tuple)) or len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise CheckpointError("block_size must contain two positive integers")
    if type(row_offset) is not int or row_offset < 0:
        raise CheckpointError("row_offset must be nonnegative")
    values = decode_e4m3fn(codes) if codes.dtype == np.uint8 else np.array(codes, dtype=np.float32, copy=True)
    scales = np.asarray(scales, dtype=np.float32)
    if values.ndim != 2 or scales.ndim != 2:
        raise CheckpointError("block FP8 requires a matrix and a 2D scale grid")
    br, bc = block_size
    nr, nc = values.shape
    if scales.shape[1] != (nc + bc - 1) // bc or scales.shape[0] < (row_offset + nr + br - 1) // br:
        raise CheckpointError("FP8 scale grid shape mismatch")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise CheckpointError("FP8 scales must be positive and finite")
    # No expanded scale matrix: memory remains one decoded tensor plus a block.
    for sr in range(row_offset // br, (row_offset + nr + br - 1) // br):
        r0 = max(0, sr * br - row_offset)
        r1 = min(nr, (sr + 1) * br - row_offset)
        for c in range(scales.shape[1]):
            values[r0:r1, c * bc:min((c + 1) * bc, nc)] *= scales[sr, c]
    return values


class SafetensorsCheckpoint:
    """Index plus LRU of read-only shard mappings; no eager weight allocation."""

    def __init__(self, root: str | Path, *, block_size: tuple[int, int] = (128, 128),
                 max_open_shards: int = 8, max_tensor_bytes: int = 8 * 1024**3):
        self.root = Path(root).resolve()
        if type(max_open_shards) is not int or max_open_shards <= 0:
            raise CheckpointError("max_open_shards must be positive")
        if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
            raise CheckpointError("max_tensor_bytes must be positive")
        if not isinstance(block_size, (list, tuple)) or len(block_size) != 2 or any(type(x) is not int or x <= 0 for x in block_size):
            raise CheckpointError("invalid block_size")
        self.block_size = tuple(block_size)
        self.max_open_shards = max_open_shards
        self.max_tensor_bytes = max_tensor_bytes
        self._shards: OrderedDict[str, SafetensorsFile] = OrderedDict()
        self._closed = False
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            index = _json(index_path.read_bytes())
            if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
                raise CheckpointError("invalid or empty weight index")
            if set(index) - {"weight_map", "metadata"}:
                raise CheckpointError("unknown index fields")
            self.weight_map = dict(index["weight_map"])
            self.metadata = index.get("metadata", {})
            if not isinstance(self.metadata, dict):
                raise CheckpointError("index metadata must be an object")
            total = self.metadata.get("total_size")
            if total is not None and (type(total) is not int or total < 0):
                raise CheckpointError("invalid index total_size")
            for name, shard in self.weight_map.items():
                if not isinstance(name, str) or not name or not isinstance(shard, str):
                    raise CheckpointError("weight_map must map tensor names to shard filenames")
                self._shard_path(shard)
        else:
            single = self.root / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no model.safetensors.index.json or model.safetensors in {self.root}")
            mapped = SafetensorsFile(single, max_tensor_bytes=max_tensor_bytes)
            self.weight_map = {name: single.name for name in mapped.tensors}
            self.metadata = {}
            self._shards[single.name] = mapped

    def _shard_path(self, name: str) -> Path:
        # Check resolved containment too, so existing symlinks cannot escape root.
        if not name or Path(name).name != name or "\\" in name or not name.endswith(".safetensors"):
            raise CheckpointError(f"unsafe shard filename {name!r}")
        path = self.root / name
        if path.resolve().parent != self.root:
            raise CheckpointError(f"shard path escapes checkpoint directory: {name!r}")
        return path

    def _shard(self, name: str) -> SafetensorsFile:
        if self._closed:
            raise CheckpointError("checkpoint is closed")
        if name not in self.weight_map:
            raise CheckpointError(f"tensor not found in checkpoint index: {name}")
        shard = self.weight_map[name]
        mapped = self._shards.pop(shard, None)
        if mapped is None:
            if len(self._shards) >= self.max_open_shards:
                _, old = self._shards.popitem(last=False)
                old.close()
            mapped = SafetensorsFile(self._shard_path(shard), max_tensor_bytes=self.max_tensor_bytes)
        self._shards[shard] = mapped
        mapped.info(name)  # Fail explicitly if the index points to the wrong shard.
        return mapped

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def info(self, name: str) -> TensorInfo:
        return self._shard(name).info(name)

    def tensor(self, name: str, *, rows: slice | None = None) -> np.ndarray:
        mapped = self._shard(name)
        info = mapped.info(name)
        out = mapped.tensor(name, rows=rows)
        if info.dtype in ("F8_E4M3", "F8_E4M3FN"):
            if not name.endswith(".weight") or len(info.shape) != 2:
                raise CheckpointError(f"FP8 requires a 2D .weight tensor: {name}")
            scale_name = name[:-len("weight")] + "weight_scale_inv"
            scale_info = self.info(scale_name)
            expected = tuple((s + b - 1) // b for s, b in zip(info.shape, self.block_size))
            if scale_info.dtype != "F32" or scale_info.shape != expected:
                raise CheckpointError(f"invalid FP8 scale tensor {scale_name}: expected F32 {expected}")
            scales = self._shard(scale_name).tensor(scale_name)
            row_start = 0 if rows is None else _row_bounds(rows, info.shape)[0]
            out = dequantize_block_fp8(out, scales, self.block_size, row_offset=row_start)
        elif name.endswith(".weight") and name[:-len("weight")] + "weight_scale_inv" in self:
            raise CheckpointError(f"scale supplied for non-FP8 weight: {name}")
        return out

    get = tensor
    load = tensor
    __getitem__ = tensor

    def validate_manifest(self, expected_shapes: dict[str, tuple[int, ...]], *, inspect_shards: bool = False) -> dict[str, int]:
        """Check required names, optionally local tensor headers, without decoding.

        Extra tensors (e.g. layer 78 MTP) are reported, never silently substituted.
        Metadata-only mode does not assert that shards exist or weights are valid.
        """
        missing = set(expected_shapes) - set(self.weight_map)
        if missing:
            raise CheckpointError(f"checkpoint missing {len(missing)} required tensors: {sorted(missing)[:5]}")
        if inspect_shards:
            for name, shape in expected_shapes.items():
                info = self.info(name)
                if info.shape != tuple(shape):
                    raise CheckpointError(f"wrong shape for {name}: {info.shape} != {shape}")
                if info.dtype.startswith("F8_"):
                    if not name.endswith(".weight") or len(shape) != 2:
                        raise CheckpointError(f"unsupported FP8 tensor {name}")
                    scale = self.info(name[:-len("weight")] + "weight_scale_inv")
                    expected = tuple((s + b - 1) // b for s, b in zip(shape, self.block_size))
                    if scale.dtype != "F32" or scale.shape != expected:
                        raise CheckpointError(f"invalid FP8 scale tensor for {name}")
        return {"required_tensors": len(expected_shapes), "index_tensors": len(self.weight_map),
                "shards": len(set(self.weight_map.values())),
                "extra_tensors": len(set(self.weight_map) - set(expected_shapes)),
                "headers_checked": int(inspect_shards)}

    def close(self):
        for mapped in self._shards.values():
            mapped.close()
        self._shards.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


Checkpoint = SafetensorsCheckpoint
