"""Checkpoint -> packed native projections -> full model/cache integration."""
import numpy as np
import pytest
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from glm_vx.model import Model
from kernels.backend import VxBackend
from test_gguf_checkpoint import fixture_data, fixture_metadata, packed_weights
from test_batched_prefill import compare_cache


def write_mixed_fixture(path, c, w):
    import gguf
    writer = gguf.GGUFWriter(path, 'glm-dsa')
    for key, value in fixture_metadata(c).items():
        add = writer.add_bool if isinstance(value, bool) else writer.add_float32 if isinstance(value, float) else writer.add_uint32
        add('glm-dsa.' + key, value)
    for name, value in packed_weights(c, w).items():
        value = np.ascontiguousarray(value, dtype=np.float32)
        if value.ndim >= 2 and value.shape[-1] % 32 == 0:
            kind = gguf.GGMLQuantizationType.Q8_0
            writer.add_tensor(name, gguf.quants.quantize(value, kind), raw_dtype=kind)
        else:
            writer.add_tensor(name, value)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()


def test_mixed_quantized_model_prefill_decode_cache_exact(tmp_path):
    c, w = fixture_data()
    path = tmp_path/'mixed.gguf'
    write_mixed_fixture(path, c, w)
    plain, packed = GGUFCheckpoint(path, c), GGUFCheckpoint(path, c)
    try:
        a = Model(c, plain, VxBackend())
        b = Model(c, packed, VxBackend(), packed_weights=True)
        ca, cb = a.new_cache(), b.new_cache()
        tokens = [1, 7, 11, 8, 13, 2, 19, 2]
        for token in tokens:
            np.testing.assert_array_equal(a.forward(token, ca), b.forward(token, cb))
            compare_cache(ca, cb)
        # Batched projections retain packed weights; scalar decoding resumes exactly.
        np.testing.assert_array_equal(a.prefill([8, 4, 3], ca)[-1], b.prefill_chunk([8, 4, 3], cb))
        np.testing.assert_array_equal(a.forward(9, ca), b.forward(9, cb))
        compare_cache(ca, cb)
    finally:
        plain.close(); packed.close()


def test_packed_full_forward_never_expands_linear_weights(tmp_path, monkeypatch):
    c, w = fixture_data();path=tmp_path/'mixed.gguf';write_mixed_fixture(path,c,w)
    cp = GGUFCheckpoint(path, c)
    try:
        model = Model(c, cp, VxBackend(), packed_weights=True)
        original = cp.tensor
        def bounded(name, *, rows=None):
            if rows is None and name not in ('model.norm.weight',) and not name.endswith('kv_b_proj.weight'):
                source, expert = cp._resolve(name)
                if len(cp.store.shape(source)) >= 2:
                    pytest.fail('expanded a packed projection: ' + name)
            return original(name, rows=rows)
        monkeypatch.setattr(cp, 'tensor', bounded)
        cache=model.new_cache()
        for token in [1, 2, 17]: assert np.isfinite(model.forward(token, cache)).all()
        assert cache.position == 3
    finally: cp.close()


def test_explicit_mode_requires_compatible_checkpoint_backend():
    from glm_vx.backend import NumpyBackend
    c,w=fixture_data()
    with pytest.raises(ValueError, match='compatible'):
        Model(c,w,VxBackend(),packed_weights=True)
    with pytest.raises(ValueError, match='compatible'):
        Model(c,w,NumpyBackend(),packed_weights=True)


def test_packed_mlp_does_not_dispatch_to_expanded_resident_interface(tmp_path, monkeypatch):
    c, w = fixture_data()
    path = tmp_path/'mixed.gguf'
    write_mixed_fixture(path, c, w)
    cp = GGUFCheckpoint(path, c)
    class DualBackend(VxBackend):
        def mlp(self, *args):
            pytest.fail('packed mode dispatched to expanded resident MLP')
    try:
        baseline = Model(c, cp, VxBackend(), packed_weights=True)
        dual = Model(c, cp, DualBackend(), packed_weights=True)
        ca, cb = baseline.new_cache(), dual.new_cache()
        # Includes dense, shared and routed expert paths, not just _expert.
        for token in (1, 7):
            np.testing.assert_array_equal(baseline.forward(token, ca), dual.forward(token, cb))
            compare_cache(ca, cb)
    finally:
        cp.close()
