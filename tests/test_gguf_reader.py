"""Synthetic GGUF storage tests; expected quantized values use explicit bytes."""
from contextlib import contextmanager
import struct

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from glm_vx.checkpoint import CheckpointError
from glm_vx.gguf_reader import GGUFStore


def write_gguf(path, tensors, metadata=None):
    """Write real temporary files, with no production conversion helper."""
    writer = gguf.GGUFWriter(path, "glm-dsa")
    for key, value in (metadata or {}).items():
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, value)
    for name, data, qtype in tensors:
        writer.add_tensor(name, data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@contextmanager
def opened(path, **kwargs):
    store = GGUFStore(path, **kwargs)
    try:
        yield store
    finally:
        store.close()


def q8_bytes(shape):
    """GGML Q8_0: fp16 scale followed by 32 signed int8 coefficients."""
    assert shape[-1] % 32 == 0
    nblocks = int(np.prod(shape)) // 32
    encoded = np.empty((nblocks, 34), dtype=np.uint8)
    expected = np.empty((nblocks, 32), dtype=np.float32)
    for i in range(nblocks):
        scale = (i % 7 + 1) / 8
        coeffs = ((np.arange(32, dtype=np.int16) * 9 + i * 13) % 256 - 128).astype(np.int8)
        encoded[i, :2] = np.frombuffer(struct.pack("<e", scale), dtype=np.uint8)
        encoded[i, 2:] = coeffs.view(np.uint8)
        expected[i] = scale * coeffs.astype(np.float32)
    return encoded.reshape(*shape[:-1], shape[-1] // 32 * 34), expected.reshape(shape)


@pytest.mark.parametrize("dtype,qtype", [(np.float32, gguf.GGMLQuantizationType.F32), (np.float16, gguf.GGMLQuantizationType.F16)])
def test_float_tensors_rows_experts_and_owned_results(tmp_path, dtype, qtype):
    values = (np.arange(3 * 7 * 32).reshape(3, 7, 32) / 8 - 40).astype(dtype)
    path = write_gguf(tmp_path / "float.gguf", [("experts", values, qtype), ("matrix", values[1], qtype)])
    store = GGUFStore(path)
    assert store.shape("experts") == values.shape
    all_values = store.read("experts")
    selected = store.read("experts", expert=2, rows=slice(1, 5))
    matrix = store.read("matrix", rows=slice(-3, None))
    for result in (all_values, selected, matrix):
        assert result.dtype == np.float32
        assert result.flags.owndata
        assert not result.flags.writeable
        assert result.base is None
    store.close()
    store.close()
    np.testing.assert_array_equal(all_values, values.astype(np.float32))
    np.testing.assert_array_equal(selected, values[2, 1:5].astype(np.float32))
    np.testing.assert_array_equal(matrix, values[1, -3:].astype(np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        store.read("matrix")


def test_q8_rows_and_experts_use_bounded_decode_chunks(tmp_path, monkeypatch):
    encoded, expected = q8_bytes((3, 7, 64))
    path = write_gguf(tmp_path / "q8.gguf", [("experts", encoded, gguf.GGMLQuantizationType.Q8_0)])
    calls = []
    original = gguf.quants.dequantize

    def recording(data, qtype):
        calls.append(data.copy())
        return original(data, qtype)

    monkeypatch.setattr(gguf.quants, "dequantize", recording)
    with opened(path, decode_rows=2) as store:
        assert store.shape("experts") == expected.shape
        selected = store.read("experts", expert=1, rows=slice(1, 6))
        np.testing.assert_array_equal(selected, expected[1, 1:6])
        assert [len(call) for call in calls] == [2, 2, 1]
        np.testing.assert_array_equal(np.concatenate(calls), encoded[1, 1:6])
        calls.clear()
        np.testing.assert_array_equal(store.read("experts"), expected)
        assert sum(len(call) for call in calls) == 21
        assert max(len(call) for call in calls) == 2
    assert selected.flags.owndata and not selected.flags.writeable
    np.testing.assert_array_equal(selected, expected[1, 1:6])


def split_fixture(tmp_path, *, names=("a", "b"), counts=(2, 2), indices=(0, 1), totals=(2, 2)):
    paths = []
    for index, name in enumerate(names):
        path = tmp_path / f"split-{index + 1:05}-of-00002.gguf"
        data = np.full((2, 32), index + 1, dtype=np.float32)
        paths.append(write_gguf(path, [(name, data, gguf.GGMLQuantizationType.F32)], {
            "split.no": indices[index], "split.count": counts[index], "split.tensors.count": totals[index]
        }))
    return paths


@pytest.mark.parametrize("entry", ["first", "second", "directory"])
def test_complete_split_checkpoint(tmp_path, entry):
    paths = split_fixture(tmp_path)
    path = {"first": paths[0], "second": paths[1], "directory": tmp_path}[entry]
    with opened(path) as store:
        assert store.paths == paths
        assert set(store.tensors) == {"a", "b"}
        np.testing.assert_array_equal(store.read("a"), np.ones((2, 32)))
        np.testing.assert_array_equal(store.read("b"), np.full((2, 32), 2))


def test_missing_split_file_is_rejected(tmp_path):
    paths = split_fixture(tmp_path)
    paths[1].unlink()
    with pytest.raises((CheckpointError, FileNotFoundError)):
        GGUFStore(paths[0])


@pytest.mark.parametrize("options,error", [
    ({"counts": (2, 3)}, "index/count"),
    ({"indices": (0, 0)}, "index/count"),
    ({"totals": (2, 3)}, "tensor count"),
    ({"totals": (3, 3)}, "incomplete tensor manifest"),
    ({"names": ("same", "same")}, "duplicate tensor"),
])
def test_inconsistent_split_manifest_rejected(tmp_path, options, error):
    paths = split_fixture(tmp_path, **options)
    with pytest.raises(CheckpointError, match=error):
        GGUFStore(paths[0])


def test_single_filename_cannot_hide_missing_split(tmp_path):
    path = write_gguf(tmp_path / "single.gguf", [("a", np.ones((2, 32), np.float32), gguf.GGMLQuantizationType.F32)], {"split.count": 2})
    with pytest.raises(CheckpointError, match="missing GGUF split"):
        GGUFStore(path)


@pytest.mark.parametrize("count", [0, 1025])
def test_invalid_split_count_rejected_before_open(tmp_path, count):
    with pytest.raises(CheckpointError, match="invalid split .*count"):
        GGUFStore(tmp_path / f"split-00001-of-{count:05}.gguf")


@pytest.mark.parametrize("file_count", [0, 2])
def test_ambiguous_or_empty_directory_rejected(tmp_path, file_count):
    for i in range(file_count):
        write_gguf(tmp_path / f"model{i}.gguf", [("a", np.ones((2, 32), np.float32), gguf.GGMLQuantizationType.F32)])
    with pytest.raises(CheckpointError, match="directory must identify"):
        GGUFStore(tmp_path)


@pytest.mark.parametrize("qtype", [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.Q8_0])
def test_decode_bound_applies_to_float32_output_before_allocation(tmp_path, monkeypatch, qtype):
    encoded, expected = q8_bytes((3, 7, 64))
    data = encoded if qtype == gguf.GGMLQuantizationType.Q8_0 else expected.astype(np.float16 if qtype == gguf.GGMLQuantizationType.F16 else np.float32)
    path = write_gguf(tmp_path / "bounded.gguf", [("experts", data, qtype)])
    with opened(path, max_decode_bytes=2 * 64 * 4) as store:
        np.testing.assert_array_equal(store.read("experts", expert=1, rows=slice(2, 4)), expected[1, 2:4])
        # No allocation or dequantization is allowed when the byte bound fails.
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                pytest.fail("oversized read allocated or decoded before checking its byte bound")
            patch.setattr(np, "empty", forbidden)
            patch.setattr(gguf.quants, "dequantize", forbidden)
            for selection in ({}, {"expert": 1}, {"expert": 1, "rows": slice(2, 5)}):
                with pytest.raises(CheckpointError, match="decode exceeds byte bound"):
                    store.read("experts", **selection)
        empty = store.read("experts", expert=1, rows=slice(4, 2))
        assert empty.shape == (0, 64)
        assert empty.flags.owndata and not empty.flags.writeable
    with opened(path, max_decode_bytes=2 * 64 * 4 - 1) as store:
        with pytest.raises(CheckpointError, match="decode exceeds byte bound"):
            store.read("experts", expert=1, rows=slice(2, 4))


@pytest.mark.parametrize("selection,error", [
    ({"expert": -1}, "invalid expert"), ({"expert": 3}, "invalid expert"),
    ({"expert": True}, "invalid expert"), ({"expert": 1.0}, "invalid expert"),
    ({"rows": slice(0, 1)}, "row slice requires a matrix"),
    ({"expert": 1, "rows": [1, 2]}, "row slice requires a matrix"),
    ({"expert": 1, "rows": slice(None, None, 2)}, "contiguous"),
    ({"expert": 1, "rows": slice(None, None, -1)}, "contiguous"),
])
def test_invalid_read_selection(tmp_path, selection, error):
    encoded, _ = q8_bytes((3, 7, 32))
    path = write_gguf(tmp_path / "selection.gguf", [("experts", encoded, gguf.GGMLQuantizationType.Q8_0)])
    with opened(path) as store:
        with pytest.raises(CheckpointError, match=error):
            store.read("experts", **selection)


@pytest.mark.parametrize("option,value", [("decode_rows", 0), ("decode_rows", -1), ("decode_rows", True), ("decode_rows", 1.5), ("max_decode_bytes", 0), ("max_decode_bytes", 3), ("max_decode_bytes", True), ("max_decode_bytes", 4.5)])
def test_invalid_decode_configuration(tmp_path, option, value):
    with pytest.raises(ValueError):
        GGUFStore(tmp_path / "unused.gguf", **{option: value})


@pytest.mark.parametrize("threads", [1, 4])
def test_iq1_s_independent_known_patterns(tmp_path, threads):
    # Independent GGML C reference, pinned when these vectors were audited:
    # https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/ggml/src/ggml-common.h#L1131
    # https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/ggml/src/ggml-quants.c#L2651
    # IQ1S_DELTA is 0.125; output = fp16_d * (2 * scale + 1)
    # * (signed int8 grid entry +/- delta). These literal entries come from
    # the C uint64 table in little-endian byte order, not gguf-py internals.
    grid = {
        0: (-1, -1, -1, -1, -1, -1, -1, -1),
        1: (1, -1, -1, -1, -1, -1, -1, -1),
        2: (0, 0, -1, -1, -1, -1, -1, -1),
        3: (-1, 1, -1, -1, -1, -1, -1, -1),
        256: (-1, 0, 1, 0, 1, -1, 0, -1),
        512: (0, 0, 1, -1, -1, 0, 1, -1),
        1024: (0, 1, -1, 0, 0, 0, 0, 0),
        2047: (1, 1, 1, 1, 1, 1, 1, 1),
    }
    encoded = np.zeros((2, 50), dtype=np.uint8)
    expected = np.empty((2, 256), dtype=np.float32)
    for row in range(2):
        d = (row + 1) / 2
        encoded[row, :2] = np.frombuffer(struct.pack("<e", d), dtype=np.uint8)
        for group in range(8):
            indices = (0, 1, 2, 3) if group % 2 == 0 else (256, 512, 1024, 2047)
            negative_delta = (group + row) % 2
            qh = (group << 12) | (negative_delta << 15)
            for lane, index in enumerate(indices):
                encoded[row, 2 + 4 * group + lane] = index & 255
                qh |= (index >> 8) << (3 * lane)
                start = group * 32 + lane * 8
                delta = -0.125 if negative_delta else 0.125
                expected[row, start:start + 8] = d * (2 * group + 1) * (np.array(grid[index]) + delta)
            encoded[row, 34 + group * 2:36 + group * 2] = np.frombuffer(struct.pack("<H", qh), dtype=np.uint8)
    path = write_gguf(tmp_path / "iq1s.gguf", [("weight", encoded, gguf.GGMLQuantizationType.IQ1_S)])
    with opened(path, decode_rows=1, decode_threads=threads) as store:
        assert store.shape("weight") == (2, 256)
        np.testing.assert_array_equal(store.read("weight"), expected)
        np.testing.assert_array_equal(store.read("weight", rows=slice(1, 2)), expected[1:2])


@pytest.mark.parametrize("index", [0, 3])
def test_out_of_range_split_selector_rejected(tmp_path, index):
    split_fixture(tmp_path)
    with pytest.raises(CheckpointError, match="split"):
        GGUFStore(tmp_path / f"split-{index:05}-of-00002.gguf")


def test_split_requires_total_tensor_manifest(tmp_path):
    for i in range(2):
        write_gguf(tmp_path / f"split-{i + 1:05}-of-00002.gguf", [(f"a{i}", np.ones((2, 32), np.float32), gguf.GGMLQuantizationType.F32)], {"split.no": i, "split.count": 2})
    with pytest.raises(CheckpointError, match="split|manifest"):
        GGUFStore(tmp_path)


@pytest.mark.parametrize("header", [b"", b"GGUF", b"BAD!" + bytes(20), struct.pack("<4sIQQ", b"GGUF", 4, 0, 0), struct.pack("<4sIQQ", b"GGUF", 3, 1000001, 0), struct.pack("<4sIQQ", b"GGUF", 3, 0, 100001)])
def test_invalid_headers_rejected(tmp_path, header):
    path = tmp_path / "invalid.gguf"
    path.write_bytes(header)
    with pytest.raises(CheckpointError, match="GGUF header"):
        GGUFStore(path)


@pytest.mark.parametrize("qtype,dtype", [(gguf.GGMLQuantizationType.F32, np.float32), (gguf.GGMLQuantizationType.F16, np.float16)])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_float_data_rejected(tmp_path, qtype, dtype, value):
    data = np.zeros((2, 32), dtype=dtype)
    data[1, 0] = value
    path = write_gguf(tmp_path / "nonfinite.gguf", [("weight", data, qtype)])
    with opened(path) as store:
        np.testing.assert_array_equal(store.read("weight", rows=slice(0, 1)), np.zeros((1, 32)))
        with pytest.raises(CheckpointError, match="nonfinite"):
            store.read("weight")


@pytest.mark.parametrize("row_slice", [slice(None), slice(-100, 100), slice(-3, -1), slice(100, 200), slice(5, 2)])
def test_quantized_row_slice_clipping_and_empty_ranges(tmp_path, row_slice):
    encoded, expected = q8_bytes((7, 64))
    path = write_gguf(tmp_path / "rows.gguf", [("matrix", encoded, gguf.GGMLQuantizationType.Q8_0)])
    with opened(path, decode_rows=1, decode_threads=threads) as store:
        np.testing.assert_array_equal(store.read("matrix", rows=row_slice), expected[row_slice])


def test_inconsistent_model_metadata_across_splits_rejected(tmp_path):
    for i in range(2):
        write_gguf(tmp_path / f"split-{i + 1:05}-of-00002.gguf", [(f"a{i}", np.ones((2, 32), np.float32), gguf.GGMLQuantizationType.F32)], {
            "split.no": i, "split.count": 2, "split.tensors.count": 2,
            "glm-dsa.embedding_length": 32 + i,
        })
    with pytest.raises(CheckpointError, match="inconsistent GGUF metadata"):
        GGUFStore(tmp_path)


def test_failed_split_load_closes_already_opened_mappings(tmp_path, monkeypatch):
    paths = split_fixture(tmp_path, names=("same", "same"))
    original = gguf.GGUFReader
    readers = []

    def tracking_reader(*args, **kwargs):
        reader = original(*args, **kwargs)
        readers.append(reader)
        return reader

    monkeypatch.setattr(gguf, "GGUFReader", tracking_reader)
    with pytest.raises(CheckpointError, match="duplicate tensor"):
        GGUFStore(paths[0])
    assert len(readers) == 2
    for reader in readers:
        assert reader.data._mmap.closed
        assert reader.tensors == []
        assert not reader.fields


@pytest.mark.parametrize("scale", [float("inf"), float("nan")])
def test_nonfinite_quantized_data_rejected(tmp_path, scale):
    encoded, _ = q8_bytes((2, 32))
    encoded[1, :2] = np.frombuffer(struct.pack("<e", scale), dtype=np.uint8)
    path = write_gguf(tmp_path / "nonfinite-q8.gguf", [("weight", encoded, gguf.GGMLQuantizationType.Q8_0)])
    with opened(path) as store:
        assert np.isfinite(store.read("weight", rows=slice(0, 1))).all()
        with np.errstate(invalid="ignore"):
            with pytest.raises(CheckpointError, match="nonfinite"):
                store.read("weight")


def test_parallel_decode_matches_reference_rows_and_closes_workers(tmp_path):
    encoded, expected = q8_bytes((3, 16, 64))
    path = write_gguf(tmp_path / 'parallel.gguf', [('weight', encoded, gguf.GGMLQuantizationType.Q8_0)])
    with opened(path, decode_threads=4, decode_rows=2) as store:
        np.testing.assert_array_equal(store.read('weight', expert=1), expected[1])
        pool=store.pool
    assert pool._shutdown


@pytest.mark.parametrize('threads',[0,33,True,1.5])
def test_invalid_decoder_worker_count(tmp_path,threads):
    with pytest.raises(ValueError,match='decode_threads'):
        GGUFStore(tmp_path/'absent.gguf',decode_threads=threads)
