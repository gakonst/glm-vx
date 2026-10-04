"""Exercise ordered reduction and row tails against a scalar float32 oracle.

Usage: python kernels/test_matvec_rows.py [library ...]
"""
import ctypes as C
from pathlib import Path
import sys
import numpy as np


def main(paths):
    p = C.POINTER(C.c_float)
    rng = np.random.default_rng(20261004)
    for path in paths:
        lib = C.CDLL(str(Path(path).resolve()))
        fn = lib.glm_vx_matvec
        fn.argtypes = [p, p, p, C.c_int32, C.c_int32]
        fn.restype = C.c_int32
        count = 0
        for rows in (0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 15, 16, 17, 31):
            for cols in (0, 1, 3, 7, 8, 17, 63, 256, 513):
                for adversarial in (False, True):
                    w = rng.normal(size=(rows, cols)).astype('f4')
                    x = rng.normal(size=cols).astype('f4')
                    if adversarial:
                        # Magnitude cancellation and signed zeros expose reduction
                        # reassociation or fused multiply-add changes.
                        values = np.array([2**24, 1, -(2**24), -0., 0.125, -0.25, 3.1], 'f4')
                        w[:] = np.resize(values, (rows, cols))
                        x[:] = np.resize(np.array([1., 1., 1., -1., 0.3], 'f4'), cols)
                    expected = np.zeros(rows, 'f4')
                    for c in range(cols):
                        expected = np.add(expected, np.multiply(w[:, c], x[c], dtype=np.float32), dtype=np.float32)
                    storage = np.full(rows + 2, -12345.0, 'f4')
                    out = storage[1:-1]
                    assert fn(out.ctypes.data_as(p), w.ctypes.data_as(p), x.ctypes.data_as(p), rows, cols) == 0
                    np.testing.assert_array_equal(out.view('u4'), expected.view('u4'))
                    np.testing.assert_array_equal(storage[[0, -1]], [-12345., -12345.])
                    count += 1
        for rows, cols in ((-1, 8), (8, -1)):
            assert fn(None, None, None, rows, cols) == -1
        print(f'{path}: {count} bit-exact cases, output canaries and negative-dimension checks passed')

if __name__ == '__main__':
    main(sys.argv[1:] or ['kernels/build/libglm_vx.so'])
