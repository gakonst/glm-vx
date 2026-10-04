"""Contract tests with a fake driver, not evidence of real CUDA execution.

Run: python3 -m unittest gpu.test_runtime -v
The fake library exercises ctypes output pointers, launch parameter lifetimes,
context stacks, copies, resource ownership, and validation. It executes no PTX
and does not provide a runtime fallback. Real numerical checks require NVIDIA.
"""
import ctypes as C
import gc
import struct
import unittest
from unittest import mock
import weakref

from gpu.runtime import (CUDADriver, CUDAContext, CUDAError, CUDAUnavailable,
                         DeviceTensor, GLMKernels)


def value(x):
    return x.value if hasattr(x, "value") else x


def write(pointer, ctype, data):
    C.cast(pointer, C.POINTER(ctype))[0] = data


class FakeFunction:
    def __init__(self, library, name):
        self.library, self.name = library, name

    def __call__(self, *args):
        self.library.calls.append((self.name, args))
        failure = self.library.errors.get(self.name)
        if failure is not None:
            return failure
        return self.library.dispatch(self.name, args)


class FakeLibrary:
    """In-memory API mock: no numerical kernels or CPU fallback implementations."""
    def __init__(self, *, devices=1, capability=(8, 0)):
        self.calls, self.errors, self.functions = [], {}, {}
        self.devices, self.capability = devices, capability
        self.stack = [777]  # A preexisting external current context must survive.
        self.next_handle, self.next_address = 1000, 0x100000000
        self.allocations, self.kernel_names = {}, {}
        self.launches = []
        self.on_launch = None

    def __getattr__(self, name):
        if name.startswith("cu"):
            if name not in self.functions:
                self.functions[name] = FakeFunction(self, name)
            return self.functions[name]
        raise AttributeError(name)

    def handle(self, pointer):
        self.next_handle += 1
        write(pointer, C.c_void_p, self.next_handle)
        return self.next_handle

    def storage(self, pointer, count):
        pointer, count = value(pointer), value(count)
        for start, allocation in self.allocations.items():
            if start <= pointer and pointer + count <= start + len(allocation):
                return allocation, pointer - start
        raise AssertionError("mock saw an invalid device byte range")

    def dispatch(self, name, args):
        if name == "cuDeviceGetCount":
            write(args[0], C.c_int32, self.devices)
        elif name == "cuDeviceGet":
            write(args[0], C.c_int32, value(args[1]))
        elif name == "cuDeviceGetName":
            C.memmove(args[0], b"MOCK ONLY - NO GPU\x00", 19)
        elif name == "cuDriverGetVersion":
            write(args[0], C.c_int32, 12000)
        elif name == "cuDeviceGetAttribute":
            attrs = {1: 1024, 2: 1024, 3: 1024, 4: 64, 5: (1 << 31) - 1,
                     6: 65535, 7: 65535, 8: 49152, 75: self.capability[0], 76: self.capability[1]}
            write(args[0], C.c_int32, attrs[value(args[1])])
        elif name == "cuCtxCreate_v2":
            self.stack.append(self.handle(args[0]))
        elif name == "cuCtxPushCurrent_v2":
            self.stack.append(value(args[0]))
        elif name == "cuCtxPopCurrent_v2":
            write(args[0], C.c_void_p, self.stack.pop())
        elif name in ("cuModuleLoadDataEx", "cuStreamCreate", "cuEventCreate"):
            self.handle(args[0])
        elif name == "cuModuleGetFunction":
            self.kernel_names[self.handle(args[0])] = args[2].decode()
        elif name == "cuFuncGetAttribute":
            write(args[0], C.c_int32, 1024 if args[1] == 0 else 1024)
        elif name == "cuMemAlloc_v2":
            address = self.next_address
            self.next_address += value(args[1]) + 256
            self.allocations[address] = bytearray(value(args[1]))
            write(args[0], C.c_uint64, address)
        elif name == "cuMemFree_v2":
            del self.allocations[value(args[0])]
        elif name == "cuMemcpyHtoD_v2":
            allocation, offset = self.storage(args[0], args[2])
            allocation[offset:offset + value(args[2])] = C.string_at(args[1], value(args[2]))
        elif name == "cuMemcpyDtoH_v2":
            allocation, offset = self.storage(args[1], args[2])
            raw = bytes(allocation[offset:offset + value(args[2])])
            C.memmove(args[0], raw, len(raw))
        elif name == "cuLaunchKernel":
            self.launches.append((self.kernel_names[value(args[0])], args[1:7], value(args[8])))
            if self.on_launch is not None:
                self.on_launch(args)
        elif name == "cuEventElapsedTime":
            write(args[0], C.c_float, 1.25)
        elif name in ("cuGetErrorName", "cuGetErrorString"):
            write(args[1], C.c_char_p, b"MOCK_CUDA_ERROR")
        return 0

    def names(self):
        return [name for name, _ in self.calls]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.library = FakeLibrary()
        self.driver = CUDADriver(_library=self.library)
        self.context = CUDAContext(_driver=self.driver)
        self.addCleanup(self.context.close)

    def module(self):
        return self.context.load_ptx_bytes(".version 7.6\n.target sm_80\n.address_size 64\n")

    def suite(self):
        return GLMKernels(self.module())

    def test_binding_signatures_are_explicit(self):
        self.assertEqual(self.driver.cuMemAlloc_v2.argtypes[1], C.c_size_t)
        self.assertEqual(len(self.driver.cuLaunchKernel.argtypes), 11)
        self.assertEqual(self.driver.cuLaunchKernel.restype, C.c_int32)

    def test_missing_driver_fails_closed(self):
        with mock.patch("gpu.runtime.C.CDLL", side_effect=OSError("missing")):
            with self.assertRaises(CUDAUnavailable):
                CUDAContext()

    def test_missing_symbol_fails_closed(self):
        with self.assertRaises(CUDAUnavailable):
            CUDADriver(_library=object())

    def test_no_device_and_initialization_errors_fail_closed(self):
        with self.assertRaises(CUDAUnavailable):
            CUDAContext(_driver=CUDADriver(_library=FakeLibrary(devices=0)))
        library = FakeLibrary()
        library.errors["cuInit"] = 100
        with self.assertRaises(CUDAUnavailable):
            CUDAContext(_driver=CUDADriver(_library=library))
        self.assertNotIn("cuCtxCreate_v2", library.names())

    def test_invalid_ordinal_and_capability(self):
        with self.assertRaises(ValueError):
            CUDAContext(1, _driver=self.driver)
        with self.assertRaises(CUDAUnavailable):
            CUDAContext(minimum_compute_capability=(9, 0), _driver=self.driver)
        with self.assertRaises(TypeError):
            CUDAContext(True, _driver=self.driver)

    def test_context_restores_prior_stack(self):
        self.assertEqual(self.library.stack, [777])
        self.context.synchronize()
        self.assertEqual(self.library.stack, [777])
        self.library.errors["cuCtxSynchronize"] = 719
        with self.assertRaises(CUDAError):
            self.context.synchronize()
        self.assertEqual(self.library.stack, [777])
        del self.library.errors["cuCtxSynchronize"]

    def test_context_manager_cleanup_and_idempotence(self):
        with CUDAContext(_driver=self.driver) as ctx:
            tensor = ctx.tensor((2,), data=struct.pack("ff", 1, 2))
            stream = ctx.stream()
        self.assertTrue(ctx.closed)
        self.assertTrue(tensor.closed)
        self.assertTrue(stream.closed)
        ctx.close()
        tensor.close()
        with self.assertRaises(RuntimeError):
            tensor.download()

    def test_tensor_upload_roundtrip_size_and_dtype(self):
        raw = struct.pack("4f", 1, 2, 3, 4)
        tensor = self.context.tensor((2, 2), data=raw)
        self.assertEqual(tensor.download(), raw)
        self.assertEqual(tensor.nbytes, 16)
        with self.assertRaises(ValueError):
            tensor.upload(b"short")
        for shape in ((-1,), (True,), (1.5,), ()):
            with self.assertRaises((ValueError, TypeError)):
                self.context.tensor(shape)
        with self.assertRaises(ValueError):
            self.context.tensor((1,), "bogus")
        with self.assertRaises(ValueError):
            self.context.tensor((1 << 62,), "float64")
        with self.assertRaises(ValueError):
            tensor.upload(memoryview(bytearray(32))[::2])

    def test_empty_tensor_has_owned_dummy_storage_and_no_copy(self):
        tensor = self.context.tensor((2, 0), data=b"")
        self.assertEqual(tensor.nbytes, 0)
        self.assertEqual(tensor.size, 0)
        self.assertEqual(tensor.allocation.nbytes, 1)
        self.assertEqual(tensor.download(), b"")
        self.assertNotIn("cuMemcpyHtoD_v2", self.library.names())

    def test_tensor_views_hold_owner_and_check_alignment_range(self):
        tensor = self.context.tensor((4,), data=struct.pack("4f", 1, 2, 3, 4))
        allocation_ref = weakref.ref(tensor.allocation)
        view = tensor.view((2,), offset=4)
        pointer = tensor.pointer
        del tensor
        gc.collect()
        self.assertIsNotNone(allocation_ref())
        self.assertEqual(view.pointer, pointer + 4)
        self.assertEqual(view.download(), struct.pack("2f", 2, 3))
        with self.assertRaises(ValueError):
            view.view((3,))
        with self.assertRaises(ValueError):
            DeviceTensor(view.allocation, (1,), offset=1)
        with self.assertRaises(ValueError):
            DeviceTensor(view.allocation, (5,))
        view.close()
        with self.assertRaises(RuntimeError):
            _ = view.pointer

    def test_buffers_validate_ranges_and_zero_length(self):
        buffer = self.context.allocate(8)
        buffer.upload(b"abcd", offset=4)
        self.assertEqual(buffer.download(offset=4), b"abcd")
        self.assertEqual(buffer.download(offset=8, nbytes=0), b"")
        for action in (lambda: buffer.upload(b"x", offset=8),
                       lambda: buffer.upload(b"", offset=-1),
                       lambda: buffer.download(offset=9),
                       lambda: buffer.download(nbytes=9),
                       lambda: self.context.allocate(0)):
            with self.assertRaises(ValueError):
                action()

    def test_failed_upload_releases_new_allocation(self):
        before = len(self.library.allocations)
        with self.assertRaises(ValueError):
            self.context.tensor((10,), data=b"bad")
        self.assertEqual(len(self.library.allocations), before)

    def test_launch_copies_pointer_and_scalar_parameters_while_alive(self):
        kernel = self.module().kernel("test")
        tensor = self.context.tensor((8,))
        view = tensor.view((2,), offset=4)
        stream = self.context.stream()
        def inspect(args):
            gc.collect()
            params = args[9]
            self.assertEqual(C.cast(params[0], C.POINTER(C.c_uint64))[0], view.pointer)
            self.assertEqual(C.cast(params[1], C.POINTER(C.c_int32))[0], 123)
            self.assertAlmostEqual(C.cast(params[2], C.POINTER(C.c_float))[0], 0.25)
            self.assertEqual(args[1:7], (2, 1, 1, 32, 1, 1))
            self.assertEqual(value(args[8]), stream.handle.value)
        self.library.on_launch = inspect
        kernel.launch((2,), (32,), [view, C.c_int32(123), C.c_float(0.25)], stream=stream)
        self.assertEqual(len(self.library.launches), 1)

    def test_pending_launch_resources_retained_until_synchronized_close(self):
        module = self.module()
        tensor = self.context.tensor((4,))
        allocation = weakref.ref(tensor.allocation)
        module_ref = weakref.ref(module)
        module.kernel("test").launch(1, 32, [tensor])
        del tensor, module
        gc.collect()
        self.assertIsNotNone(allocation())
        self.assertIsNotNone(module_ref())
        self.library.calls.clear()
        self.context.close()
        names = self.library.names()
        self.assertLess(names.index("cuCtxSynchronize"), names.index("cuMemFree_v2"))
        self.assertLess(names.index("cuCtxSynchronize"), names.index("cuModuleUnload"))
        self.assertEqual(names[-1], "cuCtxDestroy_v2")

    def test_close_sync_failure_does_not_free_any_resource(self):
        tensor = self.context.tensor((4,))
        self.library.errors["cuCtxSynchronize"] = 719
        with self.assertRaises(CUDAError) as error:
            tensor.close()
        self.assertEqual(error.exception.code, 719)
        self.assertFalse(tensor.closed)
        self.assertNotIn("cuMemFree_v2", self.library.names())
        with self.assertRaises(CUDAError):
            self.context.close()
        self.assertFalse(self.context.closed)
        self.assertNotIn("cuCtxDestroy_v2", self.library.names())
        del self.library.errors["cuCtxSynchronize"]
        tensor.close()

    def test_launch_validation_and_closed_module(self):
        module = self.module()
        kernel = module.kernel("test")
        for grid, block in ((0, 32), (1, 1025), ((1, 1, 1, 1), 32), (True, 32), (1, (1024, 2))):
            with self.assertRaises((ValueError, TypeError)):
                kernel.launch(grid, block)
        for arg in (1, 2.0, None, C.c_void_p(123), b"data"):
            with self.assertRaises(TypeError):
                kernel.launch(1, 32, [arg])
        with self.assertRaises(ValueError):
            kernel.launch(1, 32, shared_bytes=49152)
        self.assertEqual(self.library.launches, [])
        module.close()
        with self.assertRaises(RuntimeError):
            kernel.launch(1, 32)

    def test_cross_context_arguments_streams_and_events_rejected(self):
        kernel = self.module().kernel("test")
        with CUDAContext(_driver=self.driver) as other:
            tensor, stream, event = other.tensor((1,)), other.stream(), other.event().record()
            for action in (lambda: kernel.launch(1, 32, [tensor]),
                           lambda: kernel.launch(1, 32, stream=stream),
                           lambda: self.context.event().record(stream),
                           lambda: self.context.stream().wait_event(event)):
                with self.assertRaises(ValueError):
                    action()
        self.assertEqual(self.library.launches, [])

    def test_stream_and_event_ordering_and_timing(self):
        stream = self.context.stream()
        start, end = self.context.event(), self.context.event()
        with self.assertRaises(ValueError):
            start.synchronize()
        with self.assertRaises(ValueError):
            stream.wait_event(start)
        start.record(stream)
        stream.wait_event(start)
        end.record(stream)
        self.assertTrue(stream.query())
        self.assertTrue(end.query())
        self.assertEqual(start.elapsed_ms(end), 1.25)
        self.library.errors["cuStreamQuery"] = 600
        self.library.errors["cuEventQuery"] = 600
        self.assertFalse(stream.query())
        self.assertFalse(end.query())
        self.library.errors["cuStreamQuery"] = 719
        with self.assertRaises(CUDAError):
            stream.query()
        untimed = self.context.event(timing=False).record()
        with self.assertRaises(ValueError):
            start.elapsed_ms(untimed)
        stream.close()
        with self.assertRaises(RuntimeError):
            stream.synchronize()

    def test_ptx_errors_propagate_without_execution(self):
        with self.assertRaises(ValueError):
            self.context.load_ptx_bytes(b"")
        with self.assertRaises(ValueError):
            self.context.load_ptx_bytes(b"a\x00b")
        with self.assertRaises(ValueError):
            self.context.load_ptx_bytes(b"\x00")
        self.library.errors["cuModuleLoadDataEx"] = 218
        with self.assertRaises(CUDAError) as error:
            self.module()
        self.assertEqual(error.exception.operation, "cuModuleLoadDataEx")
        self.assertEqual(error.exception.code, 218)
        self.assertEqual(self.library.launches, [])

    def test_fork_reuse_rejected(self):
        with mock.patch("gpu.runtime.os.getpid", return_value=-1):
            with self.assertRaises(RuntimeError):
                self.context.synchronize()

    def test_wrappers_launch_shapes_and_abi(self):
        kernels = self.suite()
        x = self.context.tensor((2, 64))
        residual = self.context.tensor((2, 64))
        weight = self.context.tensor((64,))
        stream = self.context.stream()
        def inspect(args):
            if self.library.kernel_names[value(args[0])].endswith("residual_rmsnorm"):
                self.assertEqual(C.cast(args[9][5], C.POINTER(C.c_int32))[0], 2)
                self.assertEqual(C.cast(args[9][6], C.POINTER(C.c_int32))[0], 64)
                self.assertAlmostEqual(C.cast(args[9][7], C.POINTER(C.c_float))[0], 1e-6)
        self.library.on_launch = inspect
        out, residual_out = kernels.residual_rmsnorm(x, residual, weight, stream=stream)
        self.assertEqual(out.shape, (2, 64))
        self.assertEqual(residual_out.shape, (2, 64))
        kernels.swiglu(x, residual, out=x, stream=stream)
        trig = self.context.tensor((32,))
        kernels.rope_glm(x, trig, trig, stream=stream)
        kernels.matvec(self.context.tensor((9, 64)), weight, stream=stream)
        self.assertEqual([entry[1] for entry in self.library.launches],
                         [(2, 1, 1, 256, 1, 1), (1, 1, 1, 256, 1, 1),
                          (1, 1, 1, 256, 1, 1), (3, 1, 1, 128, 1, 1)])
        self.assertTrue(all(entry[2] == stream.handle.value for entry in self.library.launches))

    def test_wrappers_reject_shape_dtype_alias_and_nonfinite_scalars(self):
        kernels = self.suite()
        x = self.context.tensor((2, 64))
        weight = self.context.tensor((64,))
        for eps in (0, -1, float("nan"), float("inf"), 1e300, 1e-300):
            with self.assertRaises(ValueError):
                kernels.residual_rmsnorm(x, x, weight, eps)
        with self.assertRaises(ValueError):
            kernels.residual_rmsnorm(x, x, weight, out=x, residual_out=x)
        with self.assertRaises(ValueError):
            kernels.rope_glm(x, self.context.tensor((32,)), self.context.tensor((32,)), out=x)
        with self.assertRaises(ValueError):
            kernels.swiglu(x, self.context.tensor((2, 64), "int32"))
        with self.assertRaises(ValueError):
            kernels.matvec(x, self.context.tensor((63,)))
        base = self.context.tensor((129,))
        with self.assertRaises(ValueError):
            kernels.swiglu(base.view((128,)), base.view((128,)), out=base.view((128,), offset=4))
        self.assertEqual(self.library.launches, [])

    def test_mla_abi_scratch_and_same_stream_sequence(self):
        kernels = self.suite()
        q, qr = self.context.tensor((3, 33)), self.context.tensor((3, 8))
        cache, cr = self.context.tensor((10, 33)), self.context.tensor((10, 8))
        selected = self.context.tensor((3, 5), "int32")
        stream = self.context.stream()
        seen = []
        def inspect(args):
            params = args[9]
            if self.library.kernel_names[value(args[0])].endswith("mla_partial"):
                seen.extend(C.cast(params[i], C.POINTER(C.c_int32))[0] for i in range(7, 14))
                self.assertEqual(C.cast(params[14], C.POINTER(C.c_float))[0], 0.5)
        self.library.on_launch = inspect
        out = kernels.mla_decode(q, qr, cache, cr, selected, splits=2, count=4, scale=0.5, stream=stream)
        self.assertEqual(out.shape, (3, 33))
        self.assertEqual(seen, [3, 33, 8, 10, 4, 5, 2])
        self.assertEqual([entry[0] for entry in self.library.launches],
                         ["glm_vx_gpu_mla_partial", "glm_vx_gpu_mla_merge"])
        self.assertEqual(self.library.launches[0][1], (6, 1, 1, 64, 1, 1))
        self.assertTrue(all(entry[2] == stream.handle.value for entry in self.library.launches))

    def test_mla_empty_inputs_and_validation(self):
        kernels = self.suite()
        q, qr = self.context.tensor((2, 32)), self.context.tensor((2, 0))
        cache, cr = self.context.tensor((0, 32)), self.context.tensor((0, 0))
        selected = self.context.tensor((0,), "int32")
        out = kernels.mla_decode(q, qr, cache, cr, selected, splits=1)
        self.assertEqual(out.shape, (2, 32))  # Shape only; mock never computes zeros.
        self.library.launches.clear()
        for changes in ({"count": 1}, {"splits": 0}, {"scale": float("nan")}, {"out": q}):
            with self.assertRaises(ValueError):
                kernels.mla_decode(q, qr, cache, cr, selected, **changes)
        self.assertEqual(self.library.launches, [])


    def test_launch_error_propagates_and_preserves_ownership(self):
        tensor = self.context.tensor((1,))
        kernel = self.module().kernel("test")
        self.library.errors["cuLaunchKernel"] = 700
        with self.assertRaises(CUDAError) as error:
            kernel.launch(1, 32, [tensor])
        self.assertEqual(error.exception.code, 700)
        self.assertFalse(tensor.closed)
        self.assertEqual(self.library.stack, [777])

    def test_batched_extensions_and_router_abi(self):
        kernels = self.suite()
        x, weight = self.context.tensor((63,)), self.context.tensor((63,))
        seen = []
        def inspect(args):
            params = args[9]
            name = self.library.kernel_names[value(args[0])]
            if name.endswith("rmsnorm_rows"):
                seen.append((name, C.cast(params[3], C.POINTER(C.c_int32))[0],
                             C.cast(params[4], C.POINTER(C.c_int32))[0]))
                self.assertAlmostEqual(C.cast(params[5], C.POINTER(C.c_float))[0], 1e-6)
            elif name.endswith("softmax_rows"):
                seen.append((name, C.cast(params[2], C.POINTER(C.c_int32))[0],
                             C.cast(params[3], C.POINTER(C.c_int32))[0]))
            elif name.endswith("rmsnorm"):
                seen.append((name, C.cast(params[3], C.POINTER(C.c_int32))[0]))
                self.assertAlmostEqual(C.cast(params[4], C.POINTER(C.c_float))[0], 1e-6)
            elif name.endswith("softmax"):
                seen.append((name, C.cast(params[2], C.POINTER(C.c_int32))[0]))
            else:
                seen.append((name, C.cast(params[4], C.POINTER(C.c_int32))[0],
                             C.cast(params[5], C.POINTER(C.c_int32))[0]))
                self.assertEqual(C.cast(params[6], C.POINTER(C.c_float))[0], 2.5)
        self.library.on_launch = inspect
        kernels.rmsnorm(x, weight, out=x)
        kernels.softmax(x, out=x)
        batched = self.context.tensor((2, 3, 63))
        kernels.rmsnorm_rows(batched, weight, out=batched)
        kernels.softmax_rows(batched, out=batched)
        indices, weights = kernels.router(x, weight, 8, 2.5)
        self.assertEqual(indices.dtype, "int32")
        self.assertEqual(indices.shape, (8,))
        self.assertEqual(weights.shape, (8,))
        self.assertEqual(seen, [("glm_vx_gpu_rmsnorm", 63),
                                ("glm_vx_gpu_softmax", 63),
                                ("glm_vx_gpu_rmsnorm_rows", 6, 63),
                                ("glm_vx_gpu_softmax_rows", 6, 63),
                                ("glm_vx_gpu_router", 63, 8)])
        self.assertEqual(self.library.launches[2][1], (6, 1, 1, 256, 1, 1))
        self.assertEqual(self.library.launches[3][1], (6, 1, 1, 256, 1, 1))
        self.assertEqual(self.library.launches[-1][1], (1, 1, 1, 32, 1, 1))

    def test_vector_extensions_validate_limits_and_overlap(self):
        kernels = self.suite()
        x = self.context.tensor((8,))
        for action in (lambda: kernels.router(x, x, 9),
                       lambda: kernels.router(x, x, 0),
                       lambda: kernels.router(x, x, 8, weights=x),
                       lambda: kernels.rmsnorm(x, x, out=x),
                       lambda: kernels.softmax(self.context.tensor((0,))),
                       lambda: kernels.router(self.context.tensor((257,)), self.context.tensor((257,)), 1)):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(self.library.launches, [])


    def test_fp8_optional_module_loader_and_packed_abi(self):
        kernels = self.suite()
        packed = self.context.tensor((129, 257), "uint8")
        scales = self.context.tensor((2, 3))
        x = self.context.tensor((257,))
        with self.assertRaises(RuntimeError):
            kernels.fp8_matvec(packed, scales, x)
        fp8_module = self.module()
        with mock.patch.object(self.context, "load_ptx", return_value=fp8_module) as loader:
            self.assertIs(kernels.load_fp8("fp8.ptx"), kernels)
            loader.assert_called_once_with("fp8.ptx")
        stream = self.context.stream()
        def inspect(args):
            params = args[9]
            self.assertEqual(C.cast(params[1], C.POINTER(C.c_uint64))[0], packed.pointer)
            self.assertEqual(C.cast(params[2], C.POINTER(C.c_uint64))[0], scales.pointer)
            self.assertEqual(C.cast(params[4], C.POINTER(C.c_int32))[0], 129)
            self.assertEqual(C.cast(params[5], C.POINTER(C.c_int32))[0], 257)
        self.library.on_launch = inspect
        out = kernels.fp8_matvec(packed, scales, x, stream=stream)
        self.assertEqual(out.shape, (129,))
        self.assertEqual(self.library.launches[-1],
                         ("glm_vx_gpu_fp8_matvec", (33, 1, 1, 128, 1, 1), stream.handle.value))
        self.assertEqual(packed.dtype, "uint8")

    def test_fp8_constructor_ownership_shapes_and_overlap(self):
        kernels = GLMKernels(self.module(), fp8_module=self.module())
        packed = self.context.tensor((128, 128), "uint8")
        scales, x = self.context.tensor((1, 1)), self.context.tensor((128,))
        for action in (lambda: kernels.fp8_matvec(packed, scales, x, out=x),
                       lambda: kernels.fp8_matvec(packed, self.context.tensor((1,)), x),
                       lambda: kernels.fp8_matvec(self.context.tensor((128, 128)), scales, x)):
            with self.assertRaises(ValueError):
                action()
        with CUDAContext(_driver=self.driver) as other:
            with self.assertRaises(ValueError):
                GLMKernels(self.module(), fp8_module=other.load_ptx_bytes(".version 7.6"))
        self.assertEqual(self.library.launches, [])


if __name__ == "__main__":
    unittest.main()
