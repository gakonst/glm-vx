"""Independent native codec checks; optional reference/checkpoint tests skip only if absent.

Set GLM_VX_GGML_REFERENCE and GLM_VX_GGUF_CHECKPOINT to run against external
assets. The standalone CLI always fails explicitly when assets are missing.
"""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from validation.ggml_oracle import (
    DEFAULT_CHECKPOINT, DEFAULT_REFERENCE, FORMATS, GGUF_VERSION, MAX_ROW_BYTES,
    NativeGGML, OracleError, PIN, SOURCE_HASHES, SYMBOLS, compare_arrays, main,
    read_exact, representative_indices, run_parity, verify_sources,
)


@pytest.fixture(scope="module")
def native():
    reference = Path(os.environ.get("GLM_VX_GGML_REFERENCE", DEFAULT_REFERENCE))
    if not reference.exists() or not (reference / "build/bin/libggml-base.so").exists():
        pytest.skip("optional pinned native GGML reference/build absent")
    # Existing but corrupt or wrong-pin references must fail, never silently skip.
    return NativeGGML(reference)


@pytest.fixture(scope="module")
def real_receipt(native):
    pytest.importorskip("gguf")
    checkpoint = Path(os.environ.get("GLM_VX_GGUF_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.exists():
        pytest.skip("optional real GLM GGUF checkpoint absent")
    return run_parity(native.reference, checkpoint)


def q8_payload():
    values = np.arange(-16, 16, dtype=np.int8)
    return struct.pack("<e", 0.5) + values.tobytes(), values.astype(np.float32) * 0.5


def test_oracle_calls_c_even_when_python_dequantizer_is_disabled(native, monkeypatch):
    gguf = pytest.importorskip("gguf")

    def forbidden(*args, **kwargs):
        pytest.fail("native oracle must not call Python's decoder")

    monkeypatch.setattr(gguf.quants, "dequantize", forbidden)
    raw, expected = q8_payload()
    actual = native.decode(raw, "Q8_0", 8, 32)
    np.testing.assert_array_equal(actual, expected)
    assert native.functions["Q8_0"].__name__ == "dequantize_row_q8_0"


def test_c_oracle_detects_a_broken_python_codec(native, monkeypatch):
    gguf = pytest.importorskip("gguf")
    raw, expected = q8_payload()
    original = gguf.quants.dequantize

    def broken(data, kind):
        return original(data, kind) + np.float32(0.125)

    monkeypatch.setattr(gguf.quants, "dequantize", broken)
    candidate = gguf.quants.dequantize(np.frombuffer(raw, np.uint8), gguf.GGMLQuantizationType.Q8_0)
    oracle = native.decode(raw, "Q8_0", 8, 32)
    np.testing.assert_array_equal(oracle, expected)
    metrics = compare_arrays(candidate, oracle)
    assert not metrics["passed"]
    assert metrics["unequal_elements"] == 32
    assert metrics["max_abs_error"] == 0.125
    assert metrics["candidate_sha256"] != metrics["native_sha256"]


def test_native_f32_control_preserves_raw_bits(native):
    expected = np.array([0, -0.0, -1.25, 100.5], dtype=np.float32)
    actual = native.decode(expected.tobytes(), "F32", 0, 4)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("count,raw_size", [(0, 0), (31, 34), (32, 33), (32, 35), (MAX_ROW_BYTES, 0)])
def test_invalid_buffer_cannot_reach_native_codec(native, count, raw_size):
    with pytest.raises(OracleError, match="bounded|byte count"):
        native.decode(bytes(raw_size), "Q8_0", 8, count)


@pytest.mark.parametrize("name,type_id", [("Q8_0", 10000), ("UNKNOWN", 8), ("IQ1_S", 0)])
def test_invalid_type_cannot_trigger_ggml_assert(native, name, type_id):
    with pytest.raises(OracleError, match="unsupported or mismatched"):
        native.layout(name, type_id)


def test_missing_reference_is_explicit(tmp_path):
    with pytest.raises(OracleError, match="missing pinned reference source"):
        NativeGGML(tmp_path / "absent")


def test_modified_codec_source_is_rejected(tmp_path):
    path = tmp_path / "ggml/src/ggml-quants.c"
    path.parent.mkdir(parents=True)
    path.write_text("not the pinned codec")
    with pytest.raises(OracleError, match="source pin mismatch"):
        verify_sources(tmp_path)


def test_missing_library_is_explicit(native, tmp_path):
    with pytest.raises(OracleError, match="missing native GGML library"):
        NativeGGML(native.reference, tmp_path / "absent.so")


def test_cli_missing_assets_fails_and_writes_error_receipt(tmp_path):
    pytest.importorskip("gguf")
    receipt = tmp_path / "parity/failure.json"
    code = main(["--reference", str(tmp_path / "no-reference"),
                 "--checkpoint", str(tmp_path), "--output", str(receipt)])
    assert code == 2
    data = json.loads(receipt.read_text())
    assert data["status"] == "error"
    assert "missing pinned reference source" in data["error"]
    assert set(data["required_formats"]) == set(FORMATS)


def test_short_positional_read_is_rejected(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"abc")
    with path.open("rb") as stream:
        assert read_exact(stream.fileno(), 2, 1) == b"bc"
        with pytest.raises(OracleError, match="short positional read"):
            read_exact(stream.fileno(), 3, 1)


def test_sampling_covers_edges_and_middle_without_duplicates():
    assert representative_indices(1) == [0]
    assert representative_indices(2) == [0, 1]
    assert representative_indices(256) == [0, 128, 255]
    with pytest.raises(OracleError, match="empty dimension"):
        representative_indices(0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_matching_nonfinite_values_never_pass(bad):
    with np.errstate(invalid="ignore"):
        result = compare_arrays(np.array([bad], np.float32), np.array([bad], np.float32))
    assert not result["passed"]
    assert not result["finite"]
    assert result["max_abs_error"] is None


def test_comparison_rejects_broadcasting_and_single_bit_errors():
    with pytest.raises(OracleError, match="shape mismatch"):
        compare_arrays(np.ones((2, 1)), np.ones(2))
    x = np.ones(4, np.float32)
    y = x.copy()
    y[2] = np.nextafter(y[2], np.float32(2))
    assert not compare_arrays(x, y)["passed"]


@pytest.mark.parametrize("kind", FORMATS)
def test_each_real_trained_format_matches_c_at_rows_and_blocks(real_receipt, kind):
    samples = [sample for sample in real_receipt["samples"] if sample["format"] == kind]
    assert samples, f"no independent real-row comparison for {kind}"
    for sample in samples:
        assert sample["native_symbol"] == SYMBOLS.get(kind, "memmove (unquantized F32)")
        assert sample["row_comparison"]["passed"], sample
        assert sample["row_comparison"]["max_abs_error"] == 0
        assert any(block["block_index"] > 0 for block in sample["blocks"])
        assert any(block["block_count"] == 2 for block in sample["blocks"])
        for block in sample["blocks"]:
            assert block["codec_comparison"]["passed"], block
            assert block["row_slice_comparison"]["passed"], block
    # Sampling includes interior and last rows, not just prefix blocks.
    assert any(sample["row"] > 0 for sample in samples)
    if kind in ("IQ1_S", "IQ2_XXS", "IQ3_XXS", "IQ4_XS", "Q2_K", "Q3_K"):
        assert {sample["expert"] for sample in samples} == {0, 128, 255}


def test_real_receipt_provenance_and_bounded_coverage(real_receipt):
    data = real_receipt
    assert data["status"] == "passed"
    assert data["gguf_version"] == GGUF_VERSION
    assert data["oracle"]["commit"] == PIN
    assert data["oracle"]["codec_source_sha256"] == SOURCE_HASHES
    assert len(data["oracle"]["library"]["sha256"]) == 64
    assert len(data["shards"]) == 6
    assert len({sample["shard"] for sample in data["samples"]}) >= 3
    summary = data["summary"]
    assert set(summary["formats"]) == set(FORMATS)
    assert summary["failed_comparisons"] == 0
    assert summary["row_samples"] >= 100
    assert summary["expert_row_samples"] >= 50
    assert summary["raw_row_bytes_read"] < 4 * 1024 * 1024
    assert all(sample["row_bytes"] < MAX_ROW_BYTES for sample in data["samples"])
    # JSON contains only portable finite metrics (no NaN/Inf masking failures).
    json.dumps(data, allow_nan=False)


def test_real_receipt_raw_hash_and_nonzero_expert_offset(real_receipt):
    checkpoint = Path(os.environ.get("GLM_VX_GGUF_CHECKPOINT", DEFAULT_CHECKPOINT))
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    sample = next(s for s in real_receipt["samples"] if s["expert"] == 255 and s["row"] > 0)
    with (directory / sample["shard"]).open("rb") as stream:
        raw = read_exact(stream.fileno(), sample["row_bytes"], sample["row_byte_offset"])
    assert hashlib.sha256(raw).hexdigest() == sample["raw_sha256"]
    assert sample["row_byte_offset"] > sample["tensor_byte_offset"]


def test_incomplete_format_coverage_cannot_be_reported_as_pass(native, tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "only-f32.gguf"
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_tensor("test.weight", np.arange(32, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(OracleError, match="missing required formats"):
        run_parity(native.reference, path)


def test_cli_real_corruption_fails_closed(native, real_receipt, monkeypatch, tmp_path):
    """Exercise the actual CLI, all native calls, summary and serialized failure."""
    import gguf
    original = gguf.quants.dequantize

    def broken(data, kind):
        return original(data, kind) + np.float32(0.125)

    monkeypatch.setattr(gguf.quants, "dequantize", broken)
    path = tmp_path / "corrupt-candidate.json"
    checkpoint = Path(os.environ.get("GLM_VX_GGUF_CHECKPOINT", DEFAULT_CHECKPOINT))
    assert main(["--reference", str(native.reference), "--checkpoint", str(checkpoint),
                 "--output", str(path)]) == 1
    data = json.loads(path.read_text())
    assert data["status"] == "failed"
    assert data["summary"]["failed_comparisons"] > 0
    assert data["summary"]["max_abs_error"] == pytest.approx(0.125, abs=1e-7)
    # C output hashes remain identical to the independent clean run.
    assert [s["row_comparison"]["native_sha256"] for s in data["samples"]] == [
        s["row_comparison"]["native_sha256"] for s in real_receipt["samples"]]
    assert all(not s["row_comparison"]["passed"] for s in data["samples"] if s["format"] != "F32")
