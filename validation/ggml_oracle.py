"""Bounded, independent real-GGUF codec parity against pinned native GGML.

Run from the repository root::

    .venv/bin/python -m validation.ggml_oracle \
        --reference ../glm-vx-reference --checkpoint ../models/glm-5.3-iq1s

The candidate is GGUFStore (gguf 0.17.1); the oracle calls the exported C
``dequantize_row_*`` symbols with ctypes, never gguf-py's dequantizers. F32
has no quantization codec: it is copied by native memmove. Raw oracle bytes
come from positional file reads, independent of the candidate's mmap slicing.
No tensor is decoded in full except bounded 1-D vectors. All eleven formats
are required; optional assets missing in the CLI are errors, not skips.

The receipt fingerprints the native binary and pinned codec source files.
Source hashes verify the declared source pin; they do not cryptographically
attest that an arbitrary supplied binary was built from those sources. The
archive build has no embedded Git commit, so that limitation is explicit.
This is sampled codec/layout parity, not full model or inference parity.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

PIN = "11fe02151f79c41d0d4af7da708755d73b9c0da6"
GGUF_VERSION = "0.17.1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT.parent / "glm-vx-reference"
DEFAULT_CHECKPOINT = ROOT.parent / "models/glm-5.3-iq1s"
FORMATS = ("F32", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
           "IQ1_S", "IQ2_XXS", "IQ3_XXS", "IQ4_XS")
SYMBOLS = {name: "dequantize_row_" + (name.lower()[:-1] + "K"
           if name.endswith("_K") else name.lower()) for name in FORMATS if name != "F32"}
# Codec inputs from the official immutable source archive at PIN.
SOURCE_HASHES = {
    "ggml/src/ggml-quants.c": "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab",
    "ggml/src/ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "ggml/src/ggml-quants.h": "28ae5fca1f3be636b36cd6c4fa2ca74fd42d229bfbd5352eaf66f3727bb8a6da",
    "ggml/src/ggml-impl.h": "43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3",
    "ggml/src/ggml.c": "7e8bf2b60598b50d718f46aa3c36c9293b10f4e7dfde6844f0693e6fc976d505",
    "ggml/include/ggml.h": "12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9",
}
MAX_ROW_BYTES = 4 * 1024 * 1024


class OracleError(RuntimeError):
    """Unavailable, unpinned or unsafe oracle input."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def verify_sources(reference: Path) -> dict:
    result = {}
    for name, expected in SOURCE_HASHES.items():
        path = reference / name
        if not path.is_file():
            raise OracleError(f"missing pinned reference source: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise OracleError(f"source pin mismatch for {name}: {actual} != {expected}")
        result[name] = actual
    return result


class NativeGGML:
    """Native C oracle; decode() deliberately has no gguf import or fallback."""

    def __init__(self, reference: Path, library: Path | None = None):
        self.reference = Path(reference).resolve()
        self.source_hashes = verify_sources(self.reference)
        self.library_path = Path(library or self.reference / "build/bin/libggml-base.so").resolve()
        if not self.library_path.is_file():
            raise OracleError(f"missing native GGML library: {self.library_path}")
        try:
            self.lib = ctypes.CDLL(str(self.library_path))
            self.lib.ggml_blck_size.argtypes = [ctypes.c_int]
            self.lib.ggml_blck_size.restype = ctypes.c_int64
            self.lib.ggml_type_size.argtypes = [ctypes.c_int]
            self.lib.ggml_type_size.restype = ctypes.c_size_t
            self.lib.ggml_type_name.argtypes = [ctypes.c_int]
            self.lib.ggml_type_name.restype = ctypes.c_char_p
            self.functions = {}
            for name, symbol in SYMBOLS.items():
                fn = getattr(self.lib, symbol)
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
                fn.restype = None
                self.functions[name] = fn
        except (OSError, AttributeError) as exc:
            raise OracleError(f"native GGML library cannot supply all codecs: {exc}") from exc

    def layout(self, name: str, type_id: int) -> tuple[int, int]:
        # Restrict IDs before crossing C: ggml's trait lookup can assert/abort.
        ids = {"F32": 0, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13,
               "Q6_K": 14, "Q8_0": 8, "IQ1_S": 19, "IQ2_XXS": 16,
               "IQ3_XXS": 18, "IQ4_XS": 23}
        if name not in ids or type_id != ids[name]:
            raise OracleError(f"unsupported or mismatched GGML type: {name}/{type_id}")
        native_name = self.lib.ggml_type_name(type_id).decode().upper()
        if native_name != name:
            raise OracleError(f"native GGML enum mismatch: {name} != {native_name}")
        return int(self.lib.ggml_blck_size(type_id)), int(self.lib.ggml_type_size(type_id))

    def decode(self, raw: bytes, name: str, type_id: int, count: int) -> np.ndarray:
        block, block_bytes = self.layout(name, type_id)
        if count <= 0 or count % block or count * 4 > MAX_ROW_BYTES:
            raise OracleError("native decode needs a bounded positive whole-block count")
        if len(raw) != count // block * block_bytes:
            raise OracleError("encoded byte count does not match native block layout")
        source = np.frombuffer(raw, dtype=np.uint8).copy()
        out = np.empty(count, dtype=np.float32)
        if name == "F32":
            ctypes.memmove(out.ctypes.data, source.ctypes.data, len(raw))
        else:
            self.functions[name](source.ctypes.data, out.ctypes.data, count)
        return out

    def provenance(self) -> dict:
        result = {"repository": "https://github.com/ggml-org/llama.cpp", "commit": PIN,
                  "codec_source_sha256": self.source_hashes,
                  "library": fingerprint(self.library_path),
                  "engine": "ctypes calls exported GGML C dequantize_row_*; native memmove for F32",
                  "symbols": SYMBOLS,
                  "binary_source_attestation": "source hashes verified; binary fingerprint recorded; no embedded commit attestation"}
        # Keep build information auditable without copying a library into git.
        for name in ("build/CMakeCache.txt", "build/compile_commands.json", "build/common/build-info.cpp"):
            path = self.reference / name
            if path.is_file():
                result[name] = fingerprint(path)
        return result


def representative_indices(count: int) -> list[int]:
    if count < 1:
        raise OracleError("cannot sample an empty dimension")
    return sorted({0, count // 2, count - 1})


def compare_arrays(candidate: np.ndarray, oracle: np.ndarray) -> dict:
    """Strict equality with finite-value checking; no loose error tolerance."""
    candidate, oracle = np.asarray(candidate), np.asarray(oracle)
    if candidate.shape != oracle.shape:
        raise OracleError(f"comparison shape mismatch: {candidate.shape} != {oracle.shape}")
    finite = bool(np.isfinite(candidate).all() and np.isfinite(oracle).all())
    unequal = int(np.count_nonzero(candidate != oracle))
    delta = candidate.astype(np.float64) - oracle.astype(np.float64)
    return {"passed": finite and unequal == 0, "finite": finite,
            "elements": int(candidate.size), "unequal_elements": unequal,
            "max_abs_error": float(np.max(np.abs(delta))) if finite and delta.size else None,
            "candidate_sha256": hashlib.sha256(candidate.astype("<f4").tobytes()).hexdigest(),
            "native_sha256": hashlib.sha256(oracle.astype("<f4").tobytes()).hexdigest()}


def read_exact(fd: int, count: int, offset: int) -> bytes:
    raw = os.pread(fd, count, offset)
    if len(raw) != count:
        raise OracleError(f"short positional read at byte {offset}: {len(raw)} != {count}")
    return raw


def run_parity(reference: Path, checkpoint: Path, library: Path | None = None) -> dict:
    if sys.byteorder != "little":
        raise OracleError("this oracle requires a little-endian host")
    if version("gguf") != GGUF_VERSION:
        raise OracleError(f"gguf version must be {GGUF_VERSION}, found {version('gguf')}")
    if not Path(checkpoint).exists():
        raise OracleError(f"missing real GGUF checkpoint: {checkpoint}")
    native = NativeGGML(reference, library)
    import gguf
    from glm_vx.gguf_reader import GGUFStore
    store = GGUFStore(checkpoint, decode_rows=1, max_decode_bytes=MAX_ROW_BYTES)
    try:
        descriptors = {}
        for path, reader in zip(store.paths, store.readers):
            for tensor in reader.tensors:
                descriptors[tensor.name] = path
        receipt = {"schema": "glm-vx.ggml-codec-parity.v1", "status": "running",
                   "scope": "sampled real trained rows, blocks and expert offsets; not end-to-end model parity",
                   "tolerance": {"atol": 0, "rtol": 0, "require_finite": True},
                   "gguf_version": version("gguf"), "numpy_version": np.__version__,
                   "python_version": platform.python_version(), "platform": platform.platform(),
                   "oracle": native.provenance(), "required_formats": list(FORMATS),
                   "sampling": "first/middle/last tensor per format; first/middle/last row and outer slice; first/middle/last block plus a two-block boundary window",
                   "max_decoded_row_bytes": MAX_ROW_BYTES,
                   "shards": [{"name": path.name, "bytes": path.stat().st_size} for path in store.paths],
                   "samples": []}
        # Manifest provenance is cheap to hash; do not reread 216 GB of shard data.
        model_dir = Path(checkpoint) if Path(checkpoint).is_dir() else Path(checkpoint).parent
        manifest = model_dir / "manifest.json"
        if manifest.is_file():
            receipt["checkpoint_manifest"] = fingerprint(manifest)
            data = json.loads(manifest.read_text())
            receipt["checkpoint_origin"] = {key: data.get(key) for key in ("repo", "revision")}
        receipt["shard_hash_policy"] = "hash sampled raw ranges and manifest only; full shard hashes are not reverified by this run"
        import gguf.quants
        receipt["candidate_decoder_source"] = fingerprint(Path(gguf.quants.__file__))
        receipt["candidate_store_source"] = fingerprint(ROOT / "glm_vx/gguf_reader.py")
        receipt["oracle_tool_source"] = fingerprint(Path(__file__))
        by_type = {kind: [] for kind in FORMATS}
        for tensor in store.tensors.values():
            name = tensor.tensor_type.name
            if name not in by_type:
                raise OracleError(f"checkpoint contains unexpected format: {name}")
            by_type[name].append(tensor)
        missing = [kind for kind, tensors in by_type.items() if not tensors]
        if missing:
            raise OracleError("checkpoint missing required formats: " + ", ".join(missing))
        receipt["tensor_counts"] = {kind: len(tensors) for kind, tensors in by_type.items()}
        for kind, tensors in by_type.items():
            for index in representative_indices(len(tensors)):
                tensor = tensors[index]
                shape = store.shape(tensor.name)
                if not 1 <= len(shape) <= 3:
                    raise OracleError(f"unsupported sample rank for {tensor.name}: {shape}")
                width = shape[-1]
                block, block_bytes = native.layout(kind, int(tensor.tensor_type))
                if (block, block_bytes) != gguf.GGML_QUANT_SIZES[tensor.tensor_type]:
                    raise OracleError(f"gguf/native block layout disagrees: {kind}")
                if width % block or width * 4 > MAX_ROW_BYTES:
                    raise OracleError(f"row is not safely bounded and block aligned: {tensor.name}")
                row_bytes = width // block * block_bytes
                rows = shape[-2] if len(shape) >= 2 else 1
                outer_count = shape[0] if len(shape) == 3 else 1
                path = descriptors[tensor.name]
                with path.open("rb") as stream:
                    for outer in representative_indices(outer_count):
                        for row in representative_indices(rows):
                            offset = int(tensor.data_offset) + (outer * rows + row) * row_bytes
                            raw = read_exact(stream.fileno(), row_bytes, offset)
                            expected = native.decode(raw, kind, int(tensor.tensor_type), width)
                            if len(shape) == 1:
                                actual = store.read(tensor.name).reshape(-1)
                            else:
                                actual = store.read(tensor.name, rows=slice(row, row + 1),
                                                    expert=outer if len(shape) == 3 else None).reshape(-1)
                            sample = {"format": kind, "native_symbol": SYMBOLS.get(kind, "memmove (unquantized F32)"),
                                      "shard": path.name, "tensor": tensor.name, "shape": list(shape),
                                      "tensor_byte_offset": int(tensor.data_offset), "outer_index": outer,
                                      "expert": outer if "_exps." in tensor.name else None,
                                      "row": row, "row_byte_offset": offset, "row_bytes": row_bytes,
                                      "block_elements": block, "block_bytes": block_bytes,
                                      "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                      "row_comparison": compare_arrays(actual, expected), "blocks": []}
                            nblocks = width // block
                            windows = [(b, 1) for b in representative_indices(nblocks)]
                            if nblocks > 1:
                                windows.append((max(0, nblocks // 2 - 1), 2))
                            for b, count in windows:
                                # Separate positional reads also verify nonzero block offsets.
                                start, size = b * block_bytes, count * block_bytes
                                fragment = read_exact(stream.fileno(), size, offset + start)
                                cblock = native.decode(fragment, kind, int(tensor.tensor_type), count * block)
                                pyblock = gguf.quants.dequantize(np.frombuffer(fragment, np.uint8), tensor.tensor_type).reshape(-1)
                                item = {"block_index": b, "block_count": count,
                                        "byte_offset": offset + start, "bytes": size,
                                        "raw_sha256": hashlib.sha256(fragment).hexdigest(),
                                        "codec_comparison": compare_arrays(pyblock, cblock),
                                        "row_slice_comparison": compare_arrays(actual[b * block:(b + count) * block], cblock)}
                                sample["blocks"].append(item)
                            receipt["samples"].append(sample)
        receipt["summary"] = summarize(receipt["samples"])
        receipt["status"] = "passed" if receipt["summary"]["failed_comparisons"] == 0 else "failed"
        return receipt
    finally:
        store.close()


def summarize(samples: list[dict]) -> dict:
    checks = [s["row_comparison"] for s in samples]
    checks.extend(check for s in samples for b in s["blocks"]
                  for check in (b["codec_comparison"], b["row_slice_comparison"]))
    return {"formats": sorted({s["format"] for s in samples}), "row_samples": len(samples),
            "block_samples": sum(len(s["blocks"]) for s in samples),
            "expert_row_samples": sum(s["expert"] is not None for s in samples),
            "decoded_row_elements": sum(s["row_comparison"]["elements"] for s in samples),
            "raw_row_bytes_read": sum(s["row_bytes"] for s in samples),
            "comparisons": len(checks), "failed_comparisons": sum(not c["passed"] for c in checks),
            "max_abs_error": max((c["max_abs_error"] for c in checks if c["max_abs_error"] is not None), default=None)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=Path(os.environ.get("GLM_VX_GGML_REFERENCE", DEFAULT_REFERENCE)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("GLM_VX_GGUF_CHECKPOINT", DEFAULT_CHECKPOINT)))
    parser.add_argument("--library", type=Path, help="override libggml-base.so path (fingerprinted in receipt)")
    parser.add_argument("--output", type=Path, default=ROOT / "build/parity/ggml-codec-parity.json")
    args = parser.parse_args(argv)
    try:
        receipt = run_parity(args.reference, args.checkpoint, args.library)
        code = 0 if receipt["status"] == "passed" else 1
    except Exception as exc:
        receipt = {"schema": "glm-vx.ggml-codec-parity.v1", "status": "error",
                   "required_formats": list(FORMATS), "commit": PIN,
                   "error": f"{type(exc).__name__}: {exc}"}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output),
                      "summary": receipt.get("summary"), "error": receipt.get("error")}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
