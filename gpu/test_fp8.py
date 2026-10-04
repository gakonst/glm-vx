"""FP8 numerical checks. No CUDA C or expanded-f32 GPU fallback is used."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

GPU_DIR = Path(__file__).resolve().parent
FP = ctypes.POINTER(ctypes.c_float)
U8P = ctypes.POINTER(ctypes.c_uint8)


def decode_oracle(byte: int) -> float:
    """Independent mathematical definition of finite-only E4M3 (bias 7)."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    value = math.ldexp(mantissa / 8, -6) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return math.copysign(value, -1 if byte & 128 else 1)


@pytest.fixture(scope="module")
def vx_reference(tmp_path_factory):
    compiler = os.environ.get("VXC") or shutil.which("vxc")
    local = GPU_DIR.parent.parent / "vx-toolchain" / "bin" / "vxc"
    if compiler is None and local.exists():
        compiler = str(local)
    if not compiler:
        pytest.skip("Vx compiler unavailable; set VXC to compile the actual Vx FP8 decoder")
    cc = os.environ.get("CC") or shutil.which("cc")
    if not cc:
        pytest.skip("C object linker unavailable (needed to link the Vx-emitted object)")
    build = tmp_path_factory.mktemp("vx-fp8-reference")
    source = build / "reference.vx"
    # This prefix is copied verbatim from production source. No translated decoder.
    source.write_text((GPU_DIR / "fp8.vx").read_text().split("// GPU ENTRY:", 1)[0])
    obj, shared = build / "reference.o", build / "reference.so"
    subprocess.run([compiler, str(source), "--action", "emit-obj", "-o", str(obj)], check=True, capture_output=True, text=True)
    subprocess.run([cc, "-shared", str(obj), "-o", str(shared)], check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(shared))
    lib.glm_vx_fp8_decode.argtypes = [ctypes.c_uint8]
    lib.glm_vx_fp8_decode.restype = ctypes.c_float
    lib.glm_vx_fp8_matvec_reference.argtypes = [FP, U8P, FP, FP, ctypes.c_int32, ctypes.c_int32]
    lib.glm_vx_fp8_matvec_reference.restype = ctypes.c_int32
    return lib


def reference_call(lib, weight, scales, x):
    n, k = weight.shape
    # Sentinels detect writes past output, including the N=0 case.
    padded = np.full(n + 2, np.float32(-12345.0))
    out = padded[1:-1]
    status = lib.glm_vx_fp8_matvec_reference(out.ctypes.data_as(FP), weight.ctypes.data_as(U8P), scales.ctypes.data_as(FP), x.ctypes.data_as(FP), n, k)
    assert status == 0
    assert padded[0] == padded[-1] == -12345.0
    return out


def fixture_data(n, k):
    rng = np.random.default_rng(n * 1009 + k)
    weight = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    weight[weight == 127] = 126
    weight[weight == 255] = 254
    scales = rng.uniform(0.01, 0.7, size=((n + 127) // 128, (k + 127) // 128)).astype(np.float32)
    x = rng.uniform(-1, 1, size=k).astype(np.float32)
    return weight, scales, x


def projection_oracle(weight, scales, x):
    n, k = weight.shape
    lut = np.array([decode_oracle(i) for i in range(256)], dtype=np.float32)
    # Match scalar fp32 accumulation independently of the Vx implementation.
    out = np.zeros(n, dtype=np.float32)
    row_tiles = np.arange(n) // 128
    for col in range(k):
        out += (lut[weight[:, col]] * scales[row_tiles, col // 128]) * x[col]
    return out


def test_vx_decoder_all_256(vx_reference):
    for byte in range(256):
        actual, expected = vx_reference.glm_vx_fp8_decode(byte), decode_oracle(byte)
        if math.isnan(expected):
            assert math.isnan(actual), f"0x{byte:02x} must be NaN"
        else:
            assert actual == expected, f"0x{byte:02x}: {actual} != {expected}"
            if expected == 0:
                assert math.copysign(1, actual) == math.copysign(1, expected)


@pytest.mark.parametrize("n,k", [(0, 0), (0, 129), (1, 0), (1, 1), (3, 31), (4, 32), (5, 33), (127, 127), (128, 128), (129, 129), (130, 257), (257, 255), (256, 256)])
def test_vx_block_scales_and_partial_tiles(vx_reference, n, k):
    weight, scales, x = fixture_data(n, k)
    actual = reference_call(vx_reference, weight, scales, x)
    np.testing.assert_array_equal(actual, projection_oracle(weight, scales, x))


def test_scale_boundaries_exact(vx_reference):
    # 0x38 == 1.0. Unique scales expose transposition and floor-vs-ceil errors.
    weight = np.full((129, 129), 0x38, dtype=np.uint8)
    scales = np.array([[1, 2], [3, 5]], dtype=np.float32)
    x = np.ones(129, dtype=np.float32)
    actual = reference_call(vx_reference, weight, scales, x)
    np.testing.assert_array_equal(actual[:128], np.full(128, 130, dtype=np.float32))
    assert actual[128] == 389


def test_nan_encoding_propagation(vx_reference):
    weight = np.array([[127], [255], [126], [254]], dtype=np.uint8)
    actual = reference_call(vx_reference, weight, np.ones((1, 1), dtype=np.float32), np.ones(1, dtype=np.float32))
    assert np.isnan(actual[:2]).all()
    np.testing.assert_array_equal(actual[2:], [448, -448])


def test_reference_rejects_negative_dimensions(vx_reference):
    null_f, null_b = FP(), U8P()
    for n, k in [(-1, 1), (1, -1)]:
        assert vx_reference.glm_vx_fp8_matvec_reference(null_f, null_b, null_f, null_f, n, k) == -1


def checked_ptx():
    import hashlib
    import json
    ptx = Path(os.environ.get("VX_FP8_PTX", str(GPU_DIR / "build" / "fp8.ptx")))
    if not ptx.exists():
        pytest.fail("Build FP8 PTX first: python3 gpu/toolchain/build_ptx.py --source gpu/fp8.vx --output gpu/build/fp8.ptx")
    evidence = json.loads(ptx.with_suffix(".json").read_text())
    assert evidence["source_sha256"] == hashlib.sha256((GPU_DIR / "fp8.vx").read_bytes()).hexdigest(), "PTX source is stale; rebuild"
    assert evidence["ptx_sha256"] == hashlib.sha256(ptx.read_bytes()).hexdigest(), "PTX does not match its build manifest"
    return ptx


def test_offline_ptx_structure():
    import re
    path = Path(os.environ.get("VX_FP8_PTX", str(GPU_DIR / "build" / "fp8.ptx")))
    if not path.exists():
        pytest.skip("FP8 PTX not built; run gpu/toolchain/build_ptx.py for offline validation")
    ptx = checked_ptx().read_text()
    assert ".visible .entry glm_vx_gpu_fp8_matvec(" in ptx
    assert re.search(r"ld\.global\.[su]8\b", ptx), "kernel must load FP8 bytes directly"
    assert "shfl.sync.down.b32" in ptx
    assert all(register in ptx for register in ("%tid.x", "%ctaid.x", "%ntid.x"))
    assert not re.search(r"\bcall(?:\.uni)?\b|\.extern\s+\.func|\.local\b", ptx)


@pytest.fixture(scope="module")
def cuda_fp8():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip(f"Real FP8 GPU validation unavailable: {exc}")
    with ctx:
        with ctx.load_ptx(checked_ptx()) as module:
            yield ctx, module.kernel("glm_vx_gpu_fp8_matvec")


def gpu_call(cuda_fp8, weight, scales, x, threads):
    ctx, kernel = cuda_fp8
    n, k = weight.shape
    # Guard output bounds on the device, including rows in a partial final CTA.
    sentinel = np.full(n + 2, -12345.0, dtype=np.float32)
    with ctx.tensor((n + 2,), "float32", sentinel).allocation as output_buffer:
        from gpu.runtime import DeviceTensor
        out = DeviceTensor(output_buffer, (n,), "float32", offset=4)
        with ctx.tensor(weight.shape, "uint8", weight).allocation as wb:
            with ctx.tensor(scales.shape, "float32", scales).allocation as sb:
                with ctx.tensor(x.shape, "float32", x).allocation as xb:
                    kernel.launch((n + threads // 32 - 1) // (threads // 32), threads,
                                  [out, wb, sb, xb, ctypes.c_int32(n), ctypes.c_int32(k)])
                    ctx.synchronize()
                    padded = np.frombuffer(output_buffer.download(), dtype=np.float32)
    assert padded[0] == padded[-1] == -12345.0
    return padded[1:-1]


@pytest.mark.parametrize("n,k,threads", [(1, 1, 32), (3, 31, 128), (5, 33, 128), (127, 127, 256), (128, 128, 128), (129, 129, 128), (130, 257, 256)])
def test_real_gpu_fp8(cuda_fp8, vx_reference, n, k, threads):
    weight, scales, x = fixture_data(n, k)
    expected = reference_call(vx_reference, weight, scales, x)
    actual = gpu_call(cuda_fp8, weight, scales, x, threads)
    # Warp reduction changes fp32 summation order relative to the scalar oracle.
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-3)


def test_real_gpu_all_encodings(cuda_fp8):
    weight = np.arange(256, dtype=np.uint8).reshape(256, 1)
    scales = np.ones((2, 1), dtype=np.float32)
    x = np.ones(1, dtype=np.float32)
    actual = gpu_call(cuda_fp8, weight, scales, x, 128)
    expected = np.array([decode_oracle(i) for i in range(256)], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
