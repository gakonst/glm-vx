"""Selection order is exact, including cancellation-scale ties and signed zeros."""
import ctypes as C
import itertools

import numpy as np
import pytest
from kernels.backend import VxBackend


@pytest.fixture(scope='module')
def backend():
    return VxBackend()


def expected(x, k):
    return np.lexsort((np.arange(len(x), dtype=np.int64), -x))[:k]


def test_exhaustive_small_ties(backend):
    for n in range(7):
        for values in itertools.product((-1., 0., 1.), repeat=n):
            x = np.array(values, np.float32)
            for k in range(n + 1):
                np.testing.assert_array_equal(backend.topk(x, k), expected(x, k))


@pytest.mark.parametrize('n', [1, 2, 3, 31, 32, 33, 127, 128, 129, 2047, 2048, 2049, 4096, 1048576])
def test_large_and_boundary_selection(backend, n):
    rng = np.random.default_rng(2309 + n)
    x = rng.normal(size=n).astype(np.float32)
    x[::3] = 0  # frequent ties cross selection and heap boundaries
    for k in sorted({0, 1, min(7, n), min(2048, n), n if n < 10000 else 4096}):
        np.testing.assert_array_equal(backend.topk(x, k), expected(x, k))


def test_extreme_finite_values_and_signed_zero(backend):
    m = np.finfo(np.float32).max
    s = np.nextafter(np.float32(0), np.float32(1))
    x = np.array([0., -0., m, -m, m, s, -s, 1., -1., 0.], np.float32)
    for k in range(len(x) + 1):
        np.testing.assert_array_equal(backend.topk(x, k), expected(x, k))


@pytest.mark.parametrize('order', ['ascending', 'descending', 'equal'])
def test_ordered_adversarial_input(backend, order):
    x = np.arange(4096, dtype=np.float32)
    if order == 'descending': x = x[::-1].copy()
    if order == 'equal': x.fill(7)
    for k in (1, 17, 2048, 4096):
        np.testing.assert_array_equal(backend.topk(x, k), expected(x, k))


def test_native_guards_and_canaries(backend):
    x = np.arange(33, dtype=np.float32)
    result = np.full(35, -997, dtype=np.int32)
    fn = backend.lib.glm_vx_topk
    ptr = C.cast(result.ctypes.data + result.itemsize, C.POINTER(C.c_int32))
    xp = x.ctypes.data_as(C.POINTER(C.c_float))
    for n, k in [(-1, 0), (33, -1), (33, 34)]:
        assert fn(ptr, xp, n, k) != 0
        assert np.all(result == -997)
    assert fn(ptr, xp, 33, 0) == 0
    assert np.all(result == -997)
    assert fn(ptr, xp, 33, 33) == 0
    assert result[0] == result[-1] == -997
    np.testing.assert_array_equal(result[1:-1], expected(x, 33))
