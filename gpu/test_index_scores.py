"""Fused DSA wrapper/transfer tests using compiled CPU SIMT, not GPU timings."""
import numpy as np
import pytest
from gpu.test_backend import backend
from gpu.test_resident_mlp import trace, assert_released
from gpu.testing.index_reference import inputs, ordered_f32, reference_f64


@pytest.mark.parametrize('scale', [None, .125, -.5, 0.0])
def test_fused_hook_numerics_and_exact_transfer_contract(backend, monkeypatch, scale):
    args = inputs()
    events = trace(backend, monkeypatch)
    result = backend.index_scores(*args, scale)
    effective = args[0].shape[1]**-.5 if scale is None else scale
    np.testing.assert_allclose(result, ordered_f32(*args, effective), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result, reference_f64(*args, effective), rtol=3e-5, atol=3e-6)
    assert result.dtype == np.float32 and result.shape == (5,)
    assert events == [('upload', x.nbytes) for x in args] + [('launch', 'index_scores', 128), ('download', 20)]
    assert not backend.cache
    assert_released(backend)


def test_signed_weights_apply_after_relu_without_second_head_scaling(backend):
    q = np.array([[1., 0], [-1., 0]], 'f4')
    keys = np.array([[2., 0], [-3., 0], [0., 0]], 'f4')
    weights = np.array([-2., .5], 'f4')
    np.testing.assert_array_equal(backend.index_scores(q, keys, weights, .25), [-1., .375, 0.])


def test_noncontiguous_inputs_and_mutating_keys_are_uploaded_fresh(backend):
    q, keys, weights = inputs(dim=66)
    q, keys = q[:, ::2], keys[:, ::2]
    for _ in range(2):
        np.testing.assert_allclose(backend.index_scores(q, keys, weights),
                                  ordered_f32(q, keys, weights, 33**-.5), rtol=3e-5, atol=3e-6)
        keys += .1
    assert not backend.cache
    assert_released(backend)


def test_empty_tokens_have_no_device_effects(backend, monkeypatch):
    events = trace(backend, monkeypatch)
    result = backend.index_scores(*inputs(tokens=0))
    assert result.shape == (0,) and result.dtype == np.float32
    assert events == [] and not backend.ctx.tensors


@pytest.mark.parametrize('index,shape', [(0, (3,)), (0, (0, 33)), (0, (3, 0)),
    (1, (5,)), (1, (5, 32)), (2, (3, 1)), (2, (2,))])
def test_invalid_shapes_fail_before_allocating(backend, index, shape):
    args = list(inputs()); args[index] = np.empty(shape, 'f4')
    with pytest.raises(ValueError, match='index_scores'):
        backend.index_scores(*args)
    assert not backend.ctx.tensors


@pytest.mark.parametrize('index', [0, 1, 2])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_before_allocating(backend, index, bad):
    args = inputs(); args[index].flat[0] = bad
    with pytest.raises(ValueError, match='finite float32'):
        backend.index_scores(*args)
    assert not backend.ctx.tensors


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, 1e40, -1e40])
def test_nonfinite_float32_scale_rejected_even_when_empty(backend, bad):
    with pytest.raises(ValueError, match='scale must remain finite'):
        backend.index_scores(*inputs(tokens=0), bad)
    assert not backend.ctx.tensors


def test_int32_launch_overflow_rejected_without_scanning_or_copying(backend):
    # Zero-stride view carries a large logical token count without allocating it.
    q, _, weights = inputs(heads=1, dim=1)
    keys = np.lib.stride_tricks.as_strided(np.ones(1, 'f4'), shape=(1 << 26, 1), strides=(0, 0))
    with pytest.raises(ValueError, match='int32-indexable'):
        backend.index_scores(q, keys, weights)
    assert not backend.ctx.tensors


@pytest.mark.parametrize('failure_at', ['upload', 'launch', 'download'])
def test_failures_release_all_created_temporaries(backend, monkeypatch, failure_at):
    method = {'upload': '_upload', 'launch': '_launch', 'download': '_read'}[failure_at]
    original = getattr(backend, method)
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure_at != 'upload' or calls == 2:
            raise RuntimeError('injected ' + failure_at)
        return original(*args, **kwargs)
    monkeypatch.setattr(backend, method, fail)
    with pytest.raises(RuntimeError, match='injected ' + failure_at):
        backend.index_scores(*inputs())
    assert_released(backend)


def test_sync_failure_keeps_pending_storage_alive(backend, monkeypatch):
    def fail(): raise RuntimeError('injected sync failure')
    monkeypatch.setattr(backend.stream, 'synchronize', fail)
    with pytest.raises(RuntimeError, match='sync failure'):
        backend.index_scores(*inputs())
    assert all(not tensor.closed for tensor in backend.ctx.tensors)


def test_closed_backend_rejects_empty_and_nonempty(backend):
    backend.close()
    for tokens in (0, 5):
        with pytest.raises(RuntimeError, match='closed'):
            backend.index_scores(*inputs(tokens=tokens))


def test_fused_phase_removes_per_head_transfers(backend, monkeypatch):
    q, keys, weights = inputs()
    scale = 33**-.5
    events = trace(backend, monkeypatch)
    per_head = [np.maximum(backend.matvec(keys, head) * scale, 0.) for head in q]
    old = backend.matvec(np.stack(per_head).T, weights)
    assert sum(e[0] == 'launch' for e in events) == len(q) + 1
    assert sum(e[0] == 'upload' for e in events) == 2 * len(q) + 2
    assert sum(e[0] == 'download' for e in events) == len(q) + 1
    events.clear()
    def forbidden(*args, **kwargs):
        pytest.fail('fused index scoring must not call per-head matvec')
    monkeypatch.setattr(backend, 'matvec', forbidden)
    new = backend.index_scores(q, keys, weights, scale)
    np.testing.assert_allclose(new, old, rtol=3e-5, atol=3e-6)
    assert sum(e[0] == 'launch' for e in events) == 1
    assert sum(e[0] == 'upload' for e in events) == 3
    assert sum(e[0] == 'download' for e in events) == 1
    assert_released(backend)
