"""Synthetic-byte tests; no real model weights or Transformers dependency."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from glm_vx.checkpoint import (CheckpointError, SafetensorsFile, SafetensorsCheckpoint,
                               decode_e4m3fn, dequantize_block_fp8)
from glm_vx.config import GLMConfig, ConfigError, MODEL_REVISION


def write_safe(path, entries, extra=b""):
    header, payload = {}, bytearray()
    for name, dtype, shape, data in entries:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + payload + extra)


def write_bad(path, header, payload=b""):
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class SafetensorsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_float_formats_and_owned_lifetime(self):
        values = np.array([1.0, -2.0, 0.0, float("inf"), float("nan")], np.float32)
        bf16 = (values.view(np.uint32) >> 16).astype("<u2")
        write_safe(self.path, [("f32", "F32", [5], values.astype("<f4").tobytes()),
                               ("f16", "F16", [5], values.astype("<f2").tobytes()),
                               ("bf16", "BF16", [5], bf16.tobytes())])
        with SafetensorsFile(self.path) as f:
            arrays = [f.tensor(k) for k in ("f32", "f16", "bf16")]
            for a in arrays:
                self.assertEqual(a.dtype, np.float32)
                np.testing.assert_equal(a, values)
        arrays[0][0] = 3
        with SafetensorsFile(self.path) as f:
            self.assertEqual(f.tensor("f32")[0], 1)

    def test_fp8_all_codes_against_scalar_formula(self):
        codes = np.arange(256, dtype=np.uint8)
        actual = decode_e4m3fn(codes)
        expected = []
        for c in range(256):
            sign, exp, mant = (-1 if c >= 128 else 1), (c >> 3) & 15, c & 7
            value = mant / 512 if exp == 0 else (1 + mant / 8) * 2.0 ** (exp - 7)
            expected.append(float("nan") if exp == 15 and mant == 7 else sign * value)
        np.testing.assert_equal(actual, np.array(expected, dtype=np.float32))
        self.assertEqual(actual[126], 448)
        self.assertEqual(actual[1], 2**-9)
        self.assertTrue(np.signbit(actual[128]))
        with self.assertRaises(CheckpointError):
            decode_e4m3fn(np.array([1], dtype=np.int16))

    def test_fp8_scaling_partial_blocks_and_row_window(self):
        # 0x38 is 1.0, 0x40 is 2.0. Shapes deliberately cross partial blocks.
        codes = np.full((3, 5), 0x38, np.uint8)
        codes[2, 4] = 0x40
        scales = np.array([[2, 3, 4], [5, 6, 7]], dtype="<f4")
        write_safe(self.path, [("x.weight", "F8_E4M3", codes.shape, codes.tobytes()),
                               ("x.weight_scale_inv", "F32", scales.shape, scales.tobytes())])
        expected = np.array([[2, 2, 3, 3, 4], [2, 2, 3, 3, 4], [5, 5, 6, 6, 14]], np.float32)
        with SafetensorsCheckpoint(self.root, block_size=(2, 2)) as c:
            np.testing.assert_equal(c.tensor("x.weight"), expected)
            np.testing.assert_equal(c["x.weight"], expected)
            np.testing.assert_equal(c.tensor("x.weight", rows=slice(1, 3)), expected[1:3])
            self.assertEqual(c.tensor("x.weight", rows=slice(0, 0)).shape, (0, 5))
        np.testing.assert_equal(dequantize_block_fp8(codes, scales, (2, 2)), expected)
        with self.assertRaises(CheckpointError):
            dequantize_block_fp8(codes, np.ones((1, 3)), (2, 2))
        for bad in (0, -1, np.nan, np.inf):
            with self.assertRaises(CheckpointError):
                dequantize_block_fp8(codes, np.full((2, 3), bad), (2, 2))

    def test_fp8_missing_scale_and_wrong_scale_shape_dtype(self):
        for scales in ([], [("x.weight_scale_inv", "F32", [1], struct.pack("<f", 1))],
                       [("x.weight_scale_inv", "BF16", [1, 1], struct.pack("<H", 0x3f80))]):
            write_safe(self.path, [("x.weight", "F8_E4M3", [1, 1], b"\x38")] + scales)
            with SafetensorsCheckpoint(self.root) as c, self.assertRaises(CheckpointError):
                c.tensor("x.weight")

    def test_bounded_rows_scalar_empty_and_strict_slice(self):
        write_safe(self.path, [("x", "F32", [4, 3], np.arange(12, dtype="<f4").tobytes()),
                               ("scalar", "F32", [], struct.pack("<f", 7)),
                               ("empty", "F32", [0, 2], b"")])
        with SafetensorsFile(self.path, max_tensor_bytes=12) as f:
            with self.assertRaises(CheckpointError):
                f.tensor("x")
            np.testing.assert_equal(f.tensor("x", rows=slice(1, 2)), [[3, 4, 5]])
            self.assertEqual(f.tensor("scalar").shape, ())
            self.assertEqual(f.tensor("empty").shape, (0, 2))
            for rows in (slice(-1, 2), slice(0, 5), slice(2, 1), slice(0, 1, 2), slice(True, 2)):
                with self.assertRaises(CheckpointError):
                    f.tensor("x", rows=rows)
            with self.assertRaises(CheckpointError):
                f.tensor("scalar", rows=slice(0, 1))
        with self.assertRaises(CheckpointError):
            f.tensor("x")

    def test_malformed_headers(self):
        base = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        cases = [({"x": dict(base, dtype="I64")}, b"\0" * 4),
                 ({"x": dict(base, shape=[-1])}, b"\0" * 4),
                 ({"x": dict(base, shape=[True])}, b"\0" * 4),
                 ({"x": dict(base, shape=[2])}, b"\0" * 4),
                 ({"x": dict(base, data_offsets=[-1, 3])}, b"\0" * 4),
                 ({"x": dict(base, data_offsets=[0, 8])}, b"\0" * 4),
                 ({"x": dict(base, data_offsets=[1, 5])}, b"\0" * 5),
                 ({"x": base, "y": base}, b"\0" * 4),
                 ({"x": base}, b"\0" * 5),
                 ({"__metadata__": {"a": 5}}, b""),
                 ({"x": dict(base, extra=1)}, b"\0" * 4),
                 (b'{"x":{},"x":{}}', b""),
                 (b'{"x":NaN}', b""),
                 (b'[]', b"")]
        for header, payload in cases:
            with self.subTest(header=header):
                write_bad(self.path, header, payload)
                with self.assertRaises(CheckpointError):
                    SafetensorsFile(self.path)
        for payload in (b"", b"a" * 7, struct.pack("<Q", 1000000000), struct.pack("<Q", 10) + b"{}"):
            self.path.write_bytes(payload)
            with self.assertRaises(CheckpointError):
                SafetensorsFile(self.path)

    def test_lazy_manifest_paths_and_lru(self):
        index = self.root / "model.safetensors.index.json"
        wm = {"x": "one.safetensors", "y": "two.safetensors"}
        index.write_text(json.dumps({"metadata": {"total_size": 8}, "weight_map": wm}))
        # Construction and manifest checks need no shards.
        with SafetensorsCheckpoint(self.root, max_open_shards=1) as c:
            self.assertEqual(c.validate_manifest({"x": (1,)})["headers_checked"], 0)
            with self.assertRaises(FileNotFoundError):
                c.tensor("x")
            write_safe(self.root / wm["x"], [("x", "F32", [1], struct.pack("<f", 1))])
            write_safe(self.root / wm["y"], [("y", "F32", [1], struct.pack("<f", 2))])
            x = c.tensor("x")
            y = c.tensor("y")
            self.assertEqual(len(c._shards), 1)
            self.assertEqual(x[0] + y[0], 3)
            with self.assertRaises(CheckpointError):
                c.validate_manifest({"x": (2,)}, inspect_shards=True)
            self.assertEqual(c.validate_manifest({"x": (1,)}, inspect_shards=True)["headers_checked"], 1)
        with self.assertRaises(CheckpointError):
            c.tensor("x")
        for shard in ("../outside.safetensors", "/a.safetensors", "a\\b.safetensors", "a.bin"):
            index.write_text(json.dumps({"weight_map": {"x": shard}}))
            with self.assertRaises(CheckpointError):
                SafetensorsCheckpoint(self.root)
        outside = self.root.parent / (self.root.name + "-outside.safetensors")
        try:
            outside.write_bytes(b"test")
            (self.root / "link.safetensors").symlink_to(outside)
            index.write_text(json.dumps({"weight_map": {"x": "link.safetensors"}}))
            with self.assertRaises(CheckpointError):
                SafetensorsCheckpoint(self.root)
        finally:
            outside.unlink(missing_ok=True)

    def test_index_tensor_disagreement_and_duplicate_keys(self):
        index = self.root / "model.safetensors.index.json"
        write_safe(self.path, [("y", "F32", [1], struct.pack("<f", 1))])
        index.write_text(json.dumps({"weight_map": {"x": self.path.name}}))
        with SafetensorsCheckpoint(self.root) as c, self.assertRaises(CheckpointError):
            c.tensor("x")
        index.write_text('{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}')
        with self.assertRaises(CheckpointError):
            SafetensorsCheckpoint(self.root)


class ConfigTests(unittest.TestCase):
    def test_pinned_official_config_and_index(self):
        root = Path(__file__).resolve().parents[1] / "metadata"
        c = GLMConfig.from_file(root / "config.json")
        self.assertEqual(c.num_hidden_layers, 78)
        self.assertEqual(c.head_dim, 64)
        self.assertEqual(c.qk_head_dim, 256)
        self.assertEqual(c.qk_norm_eps, 1e-6)
        self.assertEqual(GLMConfig.from_dict(c.to_dict()), c)
        self.assertEqual([i for i in range(78) if c.index_source(i) == i], [0, 1, 2] + list(range(6, 78, 4)))
        self.assertEqual(c.index_source(77), 74)
        self.assertEqual(c.mlp_layer_types[:4], ("dense", "dense", "dense", "sparse"))
        shapes = c.tensor_shapes()
        self.assertEqual(shapes["model.layers.0.self_attn.kv_b_proj.weight"], (28672, 512))
        self.assertNotIn("model.layers.3.self_attn.indexer.wk.weight", shapes)
        self.assertEqual(shapes["model.layers.3.mlp.experts.255.down_proj.weight"], (6144, 2048))
        with SafetensorsCheckpoint(root) as checkpoint:
            report = checkpoint.validate_manifest(shapes)
            self.assertEqual(report["index_tensors"], 118629)
            self.assertEqual(report["shards"], 141)
        self.assertEqual(json.loads((root / "sources.json").read_text())["revision"], MODEL_REVISION)

    def test_reject_unsupported_configuration(self):
        raw = json.loads((Path(__file__).resolve().parents[1] / "metadata/config.json").read_text())
        for patch in ({"model_type": "llama"}, {"attention_bias": True}, {"hidden_act": "gelu"},
                      {"num_key_value_heads": 1}, {"qk_rope_head_dim": 63}, {"n_group": 3}, {"n_group": 2}, {"norm_topk_prob": False},
                      {"rope_parameters": {"rope_type": "yarn", "factor": 4}},
                      {"indexer_types": ["shared"] * 78}, {"hidden_size": True},
                      {"mlp_layer_types": ["dense"]}, {"rms_norm_eps": float("nan")},
                      {"quantization_config": {"quant_method": "int4"}},
                      {"mystery_architecture_flag": True}, {"tie_word_embeddings": True},
                      {"qk_head_dim": 99}, {"weight_block_size": [0, 1]},
                      {"layer_types": ["full_attention"] * 78}):
            with self.subTest(patch=patch), self.assertRaises(ConfigError):
                GLMConfig.from_dict(dict(raw, **patch))

    def test_small_config_schedule(self):
        c = GLMConfig(vocab_size=10, eos_token_id=(1,), pad_token_id=0, num_hidden_layers=4,
                      hidden_size=8, n_routed_experts=4, num_experts_per_tok=2)
        self.assertEqual(c.indexer_types, ("full", "full", "full", "shared"))
        self.assertEqual(c.index_source(3), 2)
        with self.assertRaises(IndexError):
            c.index_source(-1)


if __name__ == "__main__":
    unittest.main()
