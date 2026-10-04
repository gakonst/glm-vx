"""Resident MLP numerics and transfer contract under CPU SIMT, not GPU timing."""
import numpy as np
import pytest

import gpu.backend as adapter
from gpu.testing.context import Context, Tensor


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(adapter, 'CUDAContext', Context)
    value = adapter.GPUBackend(weight_cache_bytes=1024 * 1024)
    try:
        yield value
    finally:
        value.close()


def inputs(hidden=7, width=33, output=5, *, readonly=False):
    rng = np.random.default_rng(619)
    weights = [rng.normal(0, .15, shape).astype('f4') for shape in
               ((width, hidden), (width, hidden), (output, width))]
    for weight in weights:
        weight.flags.writeable = not readonly
    return (*weights, rng.normal(size=hidden).astype('f4'))


def oracle(gate, up, down, x):
    g = gate.astype('f8') @ x.astype('f8')
    u = up.astype('f8') @ x.astype('f8')
    return down.astype('f8') @ (g / (1 + np.exp(-g)) * u)


def trace(backend, monkeypatch):
    events = []
    tensor, download, launch = backend.ctx.tensor, Tensor.download, backend._launch

    def tracked_tensor(shape, dtype='float32', data=None):
        value = tensor(shape, dtype, data)
        if data is not None:
            events.append(('upload', value.nbytes))
        return value

    def tracked_download(value):
        events.append(('download', value.nbytes))
        return download(value)

    def tracked_launch(name, grid, block, args):
        events.append(('launch', name, block))
        return launch(name, grid, block, args)

    monkeypatch.setattr(backend.ctx, 'tensor', tracked_tensor)
    monkeypatch.setattr(Tensor, 'download', tracked_download)
    monkeypatch.setattr(backend, '_launch', tracked_launch)
    return events


def assert_released(backend):
    cached = [tensor for _, tensor in backend.cache.values()]
    assert not backend.pinned
    assert backend.cache_bytes <= backend.cache_limit
    assert all(not tensor.closed if tensor in cached else tensor.closed
               for tensor in backend.ctx.tensors)


@pytest.mark.parametrize('shape', [(7, 33, 5), (33, 67, 9)])
def test_mlp_numerical_parity_and_mutable_weights(backend, shape):
    args = inputs(*shape)
    np.testing.assert_allclose(backend.mlp(*args), oracle(*args), rtol=3e-5, atol=2e-6)
    gate, up, down, x = args
    np.testing.assert_allclose(backend.mlp(*args),
        backend.matvec(down, backend.swiglu(backend.matvec(gate, x), backend.matvec(up, x))),
        rtol=3e-6, atol=2e-6)
    gate += .1
    np.testing.assert_allclose(backend.mlp(*args), oracle(*args), rtol=3e-5, atol=2e-6)
    assert not backend.cache
    assert_released(backend)


def test_one_input_upload_on_cache_hit_and_only_final_download(backend, monkeypatch):
    args = inputs(readonly=True)
    events = trace(backend, monkeypatch)
    backend.mlp(*args)
    expected_launches = [('launch', 'matvec', 128), ('launch', 'matvec', 128),
                         ('launch', 'swiglu', 256), ('launch', 'matvec', 128)]
    assert events == [('upload', value.nbytes) for value in args] + expected_launches + [('download', args[2].shape[0] * 4)]
    cached = [tensor for _, tensor in backend.cache.values()]
    assert len(cached) == 3
    events.clear()
    np.testing.assert_allclose(backend.mlp(*args), oracle(*args), rtol=3e-5, atol=2e-6)
    assert events == [('upload', args[-1].nbytes)] + expected_launches + [('download', args[2].shape[0] * 4)]
    assert [tensor for _, tensor in backend.cache.values()] == cached
    assert_released(backend)


@pytest.mark.parametrize('cache_limit', [0, 64, 128])
def test_cache_pressure_preserves_all_live_weight_inputs(backend, cache_limit):
    args = inputs(4, 4, 4, readonly=True)
    backend.cache_limit = cache_limit
    for _ in range(2):
        np.testing.assert_allclose(backend.mlp(*args), oracle(*args), rtol=3e-5, atol=2e-6)
        assert_released(backend)


@pytest.mark.parametrize('failure_at', [1, 2, 3, 4])
def test_launch_failure_releases_temporaries_and_unpins_weights(backend, monkeypatch, failure_at):
    calls = 0
    launch = backend._launch

    def fail(name, grid, block, args):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise RuntimeError('injected launch failure')
        return launch(name, grid, block, args)

    monkeypatch.setattr(backend, '_launch', fail)
    with pytest.raises(RuntimeError, match='injected launch failure'):
        backend.mlp(*inputs(readonly=True))
    assert calls == failure_at
    assert_released(backend)


def test_download_failure_releases_temporaries(backend, monkeypatch):
    def fail(_):
        raise RuntimeError('injected download failure')
    monkeypatch.setattr(backend, '_read', fail)
    with pytest.raises(RuntimeError, match='injected download failure'):
        backend.mlp(*inputs(readonly=True))
    assert_released(backend)


def test_sync_failure_does_not_free_pending_storage(backend, monkeypatch):
    def fail():
        raise RuntimeError('injected sync failure')
    monkeypatch.setattr(backend.stream, 'synchronize', fail)
    with pytest.raises(RuntimeError, match='injected sync failure'):
        backend.mlp(*inputs(readonly=True))
    assert backend.pinned
    assert all(not tensor.closed for tensor in backend.ctx.tensors)


@pytest.mark.parametrize('bad_index,bad_shape', [(0, (3,)), (1, (2, 7)), (2, (5, 2)), (3, (1, 7))])
def test_invalid_shapes_rejected_before_device_allocation(backend, bad_index, bad_shape):
    args = list(inputs())
    args[bad_index] = np.zeros(bad_shape, dtype='f4')
    with pytest.raises(ValueError, match='MLP shape mismatch'):
        backend.mlp(*args)
    assert not backend.ctx.tensors


@pytest.mark.parametrize('tuned_shape,expected', [([33, 7], [64, 64, 128]), ([5, 33], [128, 128, 64])])
def test_each_projection_honors_its_matching_tuning_shape(backend, monkeypatch, tuned_shape, expected):
    backend.tuning = {'shape': tuned_shape, 'selected_block': 64}
    events = trace(backend, monkeypatch)
    args = inputs()
    np.testing.assert_allclose(backend.mlp(*args), oracle(*args), rtol=3e-5, atol=2e-6)
    assert [event[2] for event in events if event[:2] == ('launch', 'matvec')] == expected


def test_resident_mlp_real_device():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    ctx.close()
    value = adapter.GPUBackend()
    try:
        args = inputs(33, 67, 9, readonly=True)
        np.testing.assert_allclose(value.mlp(*args), oracle(*args), rtol=5e-4, atol=3e-5)
    finally:
        value.close()


def test_cached_resident_chain_reduces_activation_transfers(backend, monkeypatch):
    gate, up, down, x = args = inputs(readonly=True)
    backend.mlp(*args)
    events = trace(backend, monkeypatch)
    expected = backend.matvec(down, backend.swiglu(backend.matvec(gate,x),backend.matvec(up,x)))
    assert sum(e[0]=='upload' for e in events)==5
    assert sum(e[0]=='download' for e in events)==4
    events.clear()
    actual = backend.mlp(*args)
    assert sum(e[0]=='upload' for e in events)==1
    assert sum(e[0]=='download' for e in events)==1
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=2e-6)
