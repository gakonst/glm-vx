"""A small, fail-closed NVIDIA CUDA Driver API runtime (Python stdlib only).

There is no CPU implementation, simulator, PyTorch, or CuPy dependency. Creating
``CUDAContext`` requires libcuda and a real visible NVIDIA device. PTX is JIT
loaded by that driver; a successful load does not prove a kernel is correct.

Typical low-level use::

    import ctypes as C
    from gpu.runtime import CUDAContext
    with CUDAContext(0) as ctx:
        with ctx.load_ptx("gpu/build/kernels.ptx") as module:
            x = ctx.tensor((1024,), "float32", host_bytes)
            y = ctx.tensor((1024,), "float32")
            with ctx.stream() as stream:
                module.kernel("my_kernel").launch(
                    (4,), (256,), [x, y, C.c_uint32(1024)], stream=stream)
                stream.synchronize()
                result_bytes = y.download()

``DeviceTensor`` describes contiguous, native-endian storage. Shapes are nonempty
sequences of nonnegative integers. Empty tensors own a one-byte allocation. Views retain their owning ``DeviceBuffer`` and
validate alignment and their full byte range. Uploads require exactly the tensor
byte count; buffer uploads/downloads also support checked byte offsets. Python
buffer objects (including ``array.array`` and memoryviews) can supply host bytes.
No implicit dtype conversion or initialization is performed.

Kernel pointer parameters must be DeviceTensor/DeviceBuffer objects; scalar
parameters must be explicit fixed-width ctypes numbers, e.g. c_int32/c_float.
The runtime keeps parameter storage alive until cuLaunchKernel returns (CUDA
copies parameter values then). The context strongly owns all CUDA resources
until explicit close or context exit. Closing a buffer, module, stream, or event
synchronizes the entire context first; this prevents pending work from using
freed resources. Upload/download are blocking and also synchronize the context.
Launches remain asynchronous. Use streams/events for GPU execution dependencies
and timing. Do not use this adapter's context concurrently through external CUDA
APIs, and do not reuse objects after close or across fork.

Every operation pushes this context and restores the caller's prior context.
Host-side operations are serialized per context. CUDA errors propagate without
fallback; a failed synchronization prevents unsafe resource destruction. Always
use a context manager or call close explicitly: no __del__ cleanup is attempted.
Mock tests validate the adapter contract only, never real GPU execution.
"""
from __future__ import annotations

import contextlib
import ctypes as C
import math
import os
from pathlib import Path
import threading

__all__ = ["CUDAError", "CUDAUnavailable", "CUDADriver", "CUDAContext", "DeviceBuffer",
           "DeviceTensor", "Module", "Kernel", "Stream", "Event", "GLMKernels"]


class CUDAError(RuntimeError):
    """A CUDA Driver API call failed; ``code`` preserves its CUresult."""
    def __init__(self, operation: str, code: int, detail: str = ""):
        self.operation, self.code = operation, code
        super().__init__(f"{operation} failed (CUDA {code})" + (f": {detail}" if detail else ""))


class CUDAUnavailable(RuntimeError):
    """No usable CUDA driver/device exists. There is deliberately no fallback."""


_P = C.c_void_p
_U64 = C.c_uint64
_U32 = C.c_uint32
_I32 = C.c_int32
_SIZE = C.c_size_t


class CUDADriver:
    """Load and bind NVIDIA's driver library. ``_library`` is for mock tests only."""
    def __init__(self, *, _library=None):
        if _library is None:
            name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
            try:
                loader = C.WinDLL if os.name == "nt" else C.CDLL
                _library = loader(name)
            except OSError as exc:
                raise CUDAUnavailable(f"Cannot load NVIDIA CUDA driver {name}: {exc}") from exc
        self.library = _library
        specs = {
            "cuInit": [_U32], "cuDriverGetVersion": [C.POINTER(_I32)],
            "cuDeviceGetCount": [C.POINTER(_I32)], "cuDeviceGet": [C.POINTER(_I32), _I32],
            "cuDeviceGetName": [_P, _I32, _I32],
            "cuDeviceGetAttribute": [C.POINTER(_I32), _I32, _I32],
            "cuCtxCreate_v2": [C.POINTER(_P), _U32, _I32], "cuCtxDestroy_v2": [_P],
            "cuCtxPushCurrent_v2": [_P], "cuCtxPopCurrent_v2": [C.POINTER(_P)],
            "cuCtxSynchronize": [],
            "cuMemAlloc_v2": [C.POINTER(_U64), _SIZE], "cuMemFree_v2": [_U64],
            "cuMemcpyHtoD_v2": [_U64, _P, _SIZE], "cuMemcpyDtoH_v2": [_P, _U64, _SIZE],
            "cuModuleLoadDataEx": [C.POINTER(_P), _P, _U32, _P, _P],
            "cuModuleUnload": [_P], "cuModuleGetFunction": [C.POINTER(_P), _P, C.c_char_p],
            "cuFuncGetAttribute": [C.POINTER(_I32), _I32, _P],
            "cuLaunchKernel": [_P, _U32, _U32, _U32, _U32, _U32, _U32, _U32, _P,
                               C.POINTER(_P), C.POINTER(_P)],
            "cuStreamCreate": [C.POINTER(_P), _U32], "cuStreamDestroy_v2": [_P],
            "cuStreamSynchronize": [_P], "cuStreamQuery": [_P],
            "cuStreamWaitEvent": [_P, _P, _U32],
            "cuEventCreate": [C.POINTER(_P), _U32], "cuEventDestroy_v2": [_P],
            "cuEventRecord": [_P, _P], "cuEventSynchronize": [_P], "cuEventQuery": [_P],
            "cuEventElapsedTime": [C.POINTER(C.c_float), _P, _P],
            "cuGetErrorName": [_I32, C.POINTER(C.c_char_p)],
            "cuGetErrorString": [_I32, C.POINTER(C.c_char_p)],
        }
        try:
            for name, argtypes in specs.items():
                fn = getattr(self.library, name)
                fn.argtypes, fn.restype = argtypes, _I32
                setattr(self, name, fn)
        except AttributeError as exc:
            raise CUDAUnavailable(f"NVIDIA driver is missing a required CUDA symbol: {exc}") from exc

    def check(self, code, operation):
        if code != 0:
            parts = []
            for name in ("cuGetErrorName", "cuGetErrorString"):
                value = C.c_char_p()
                if getattr(self, name)(code, C.byref(value)) == 0 and value.value:
                    parts.append(value.value.decode("utf8", "replace"))
            raise CUDAError(operation, int(code), ": ".join(parts))

    def call(self, name, *args):
        self.check(getattr(self, name)(*args), name)


def _integer(value, name, *, minimum=0, maximum=(1 << 63) - 1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _host_bytes(data):
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("host data must support the Python buffer protocol") from exc
    if not view.c_contiguous:
        raise ValueError("host data must be C-contiguous")
    return view.tobytes()


class CUDAContext:
    """Own one CUDA context and its resources; use ``with CUDAContext() as ctx``.

    ``device`` is a visible CUDA device ordinal. ``minimum_compute_capability``
    can require e.g. (8, 0); GLMKernels requires this for its sm80 PTX. ``_driver``
    is an internal test injection, not an alternate execution backend.
    """
    def __init__(self, device=0, *, minimum_compute_capability=None, _driver=None):
        _integer(device, "device", maximum=(1 << 31) - 1)
        self.driver = _driver if _driver is not None else CUDADriver()
        self._lock = threading.RLock()
        self._resources = []
        self._closed = True
        self.handle = _P()
        self._pid = os.getpid()
        d = self.driver
        try:
            d.call("cuInit", 0)
            count = _I32()
            d.call("cuDeviceGetCount", C.byref(count))
        except CUDAError as exc:
            raise CUDAUnavailable(f"CUDA initialization failed: {exc}") from exc
        if count.value == 0:
            raise CUDAUnavailable("No visible NVIDIA CUDA devices; CPU fallback is disabled")
        if device >= count.value:
            raise ValueError(f"CUDA device ordinal {device} is outside [0, {count.value})")
        self.device = _I32()
        d.call("cuDeviceGet", C.byref(self.device), device)
        name = C.create_string_buffer(256)
        d.call("cuDeviceGetName", name, len(name), self.device)
        version = _I32()
        d.call("cuDriverGetVersion", C.byref(version))
        self.name = name.value.decode("utf8", "replace")
        self.driver_version = version.value
        self.compute_capability = (self._attribute(75), self._attribute(76))
        self.max_threads_per_block = self._attribute(1)
        self.max_block = tuple(self._attribute(n) for n in (2, 3, 4))
        self.max_grid = tuple(self._attribute(n) for n in (5, 6, 7))
        self.max_shared_bytes = self._attribute(8)
        if minimum_compute_capability is not None:
            minimum = tuple(minimum_compute_capability)
            if len(minimum) != 2 or any(type(n) is not int or n < 0 for n in minimum):
                raise ValueError("minimum_compute_capability must contain two nonnegative integers")
            if self.compute_capability < minimum:
                raise CUDAUnavailable(f"Device {self.name} has compute capability {self.compute_capability}; requires {minimum}")
        d.call("cuCtxCreate_v2", C.byref(self.handle), 0, self.device)
        try:
            previous = _P()
            d.call("cuCtxPopCurrent_v2", C.byref(previous))
        except BaseException:
            d.call("cuCtxDestroy_v2", self.handle)
            raise
        self._closed = False

    def _attribute(self, attribute):
        result = _I32()
        self.driver.call("cuDeviceGetAttribute", C.byref(result), attribute, self.device)
        return result.value

    def _ensure_open(self):
        if os.getpid() != self._pid:
            raise RuntimeError("CUDA objects cannot be reused after fork")
        if self._closed:
            raise RuntimeError("CUDA context is closed")

    @contextlib.contextmanager
    def _activate(self):
        with self._lock:
            self._ensure_open()
            self.driver.call("cuCtxPushCurrent_v2", self.handle)
            try:
                yield
            finally:
                previous = _P()
                self.driver.call("cuCtxPopCurrent_v2", C.byref(previous))

    def _synchronize_current(self):
        self.driver.call("cuCtxSynchronize")

    def synchronize(self):
        """Block until all context work finishes, surfacing asynchronous CUDA errors."""
        with self._activate():
            self._synchronize_current()

    def allocate(self, nbytes):
        return DeviceBuffer(self, nbytes)

    def tensor(self, shape, dtype="float32", data=None):
        """Allocate uninitialized contiguous device storage; optionally upload bytes."""
        shape, dtype, nbytes = DeviceTensor._layout(shape, dtype)
        buffer = self.allocate(max(1, nbytes))
        try:
            tensor = DeviceTensor(buffer, shape, dtype)
            if data is not None:
                tensor.upload(data)
            return tensor
        except BaseException:
            buffer.close()
            raise

    def load_ptx(self, path):
        """Read a PTX file and JIT load it into this context (not CPU compilation)."""
        return Module(self, Path(path).read_bytes())

    def load_ptx_bytes(self, ptx):
        """JIT load PTX bytes or text. Driver errors propagate without fallback."""
        if isinstance(ptx, str):
            ptx = ptx.encode("utf8")
        return Module(self, ptx)

    def stream(self, *, nonblocking=True):
        return Stream(self, nonblocking=nonblocking)

    def event(self, *, timing=True):
        return Event(self, timing=timing)

    @property
    def closed(self):
        return self._closed

    def close(self):
        """Synchronize, release owned resources in reverse order, destroy context.

        Idempotent after success. If synchronization/destruction fails, propagate
        the error and retain remaining ownership so callers may inspect/retry.
        """
        with self._lock:
            if self._closed:
                return
            with self._activate():
                self._synchronize_current()
                for resource in list(reversed(self._resources)):
                    resource._destroy_current()
            self.driver.call("cuCtxDestroy_v2", self.handle)
            self._closed = True
            self.handle = _P()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class _Resource:
    def _setup(self, context):
        self.context = context
        self._closed = False
        context._resources.append(self)

    @property
    def closed(self):
        return self._closed or self.context.closed

    def _ensure_open(self):
        self.context._ensure_open()
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _mark_closed(self):
        self._closed = True
        self.context._resources.remove(self)

    def close(self):
        with self.context._lock:
            if self._closed:
                return
            with self.context._activate():
                self.context._synchronize_current()
                self._destroy_current()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class DeviceBuffer(_Resource):
    """A context-owned device allocation; close synchronizes pending GPU work."""
    def __init__(self, context, nbytes):
        _integer(nbytes, "nbytes", minimum=1, maximum=min((1 << 63) - 1, C.c_size_t(-1).value))
        self.nbytes = nbytes
        self._pointer = _U64()
        with context._activate():
            context.driver.call("cuMemAlloc_v2", C.byref(self._pointer), nbytes)
            self._setup(context)

    @property
    def pointer(self):
        self._ensure_open()
        return self._pointer.value

    def _range(self, offset, nbytes):
        _integer(offset, "offset")
        _integer(nbytes, "nbytes")
        if offset + nbytes > self.nbytes:
            raise ValueError("transfer or view exceeds device allocation")

    def upload(self, data, *, offset=0):
        """Blocking byte upload. Synchronizes all context work before copying."""
        raw = _host_bytes(data)
        self._range(offset, len(raw))
        with self.context._activate():
            self._ensure_open()
            self.context._synchronize_current()
            if raw:
                host = C.create_string_buffer(raw, len(raw))
                self.context.driver.call("cuMemcpyHtoD_v2", self._pointer.value + offset, host, len(raw))
        return self

    def download(self, *, offset=0, nbytes=None):
        """Blocking download to bytes. Synchronizes all context work first."""
        _integer(offset, "offset")
        if nbytes is None:
            nbytes = self.nbytes - offset
        self._range(offset, nbytes)
        with self.context._activate():
            self._ensure_open()
            self.context._synchronize_current()
            host = C.create_string_buffer(nbytes)
            if nbytes:
                self.context.driver.call("cuMemcpyDtoH_v2", host, self._pointer.value + offset, nbytes)
            return host.raw

    def _destroy_current(self):
        self.context.driver.call("cuMemFree_v2", self._pointer)
        self._mark_closed()
        self._pointer = _U64()


_DTYPES = {"float16": 2, "float32": 4, "float64": 8, "int8": 1, "uint8": 1,
           "int16": 2, "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8}


class DeviceTensor:
    """A contiguous typed view retaining its DeviceBuffer owner.

    ``offset`` is in bytes from the start of ``allocation``. The tensor's close
    closes that entire allocation, invalidating sibling views; usually let the
    owning CUDAContext close it. ``view`` offsets are relative to this view and
    cannot extend beyond its range. Dtype strings are explicit full names.
    """
    @staticmethod
    def _layout(shape, dtype):
        if isinstance(shape, (str, bytes)):
            raise TypeError("shape must be a sequence of nonnegative integers")
        shape = tuple(shape)
        if not shape:
            raise ValueError("shape must have at least one dimension")
        for dimension in shape:
            _integer(dimension, "shape dimension")
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r}; use one of {tuple(_DTYPES)}")
        nbytes = math.prod(shape) * _DTYPES[dtype]
        _integer(nbytes, "tensor byte count")
        return shape, dtype, nbytes

    def __init__(self, allocation, shape, dtype="float32", *, offset=0):
        if not isinstance(allocation, DeviceBuffer):
            raise TypeError("allocation must be a DeviceBuffer")
        shape, dtype, nbytes = self._layout(shape, dtype)
        allocation._ensure_open()
        allocation._range(offset, nbytes)
        if offset % _DTYPES[dtype]:
            raise ValueError("tensor offset must be aligned to its dtype size")
        self.allocation = allocation
        self.shape, self.dtype, self.nbytes, self.offset = shape, dtype, nbytes, offset

    @property
    def context(self):
        return self.allocation.context

    @property
    def pointer(self):
        return self.allocation.pointer + self.offset

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def closed(self):
        return self.allocation.closed

    def view(self, shape, dtype=None, *, offset=0):
        shape, dtype, nbytes = self._layout(shape, self.dtype if dtype is None else dtype)
        _integer(offset, "offset")
        if offset + nbytes > self.nbytes:
            raise ValueError("view exceeds parent tensor byte range")
        return DeviceTensor(self.allocation, shape, dtype, offset=self.offset + offset)

    def upload(self, data):
        raw = _host_bytes(data)
        if len(raw) != self.nbytes:
            raise ValueError(f"tensor upload needs exactly {self.nbytes} bytes, received {len(raw)}")
        self.allocation.upload(raw, offset=self.offset)
        return self

    def download(self):
        return self.allocation.download(offset=self.offset, nbytes=self.nbytes)

    def close(self):
        self.allocation.close()


class Module(_Resource):
    """A loaded PTX module; kernels retain it and reject launches after unload."""
    def __init__(self, context, ptx):
        raw = _host_bytes(ptx)
        if not raw.rstrip(b"\x00") or b"\x00" in raw.rstrip(b"\x00"):
            raise ValueError("PTX must be nonempty text without interior NUL bytes")
        image = C.create_string_buffer(raw.rstrip(b"\x00"))
        self.handle = _P()
        self._kernels = {}
        with context._activate():
            context.driver.call("cuModuleLoadDataEx", C.byref(self.handle), image, 0, None, None)
            self._setup(context)

    def kernel(self, name):
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("kernel name must be nonempty text without NUL")
        with self.context._activate():
            self._ensure_open()
            if name not in self._kernels:
                self._kernels[name] = Kernel(self, name)
            return self._kernels[name]

    def _destroy_current(self):
        self.context.driver.call("cuModuleUnload", self.handle)
        self._mark_closed()
        self.handle = _P()


def _dimensions(value, name, limits):
    if isinstance(value, int):
        value = (value,)
    value = tuple(value)
    if not 1 <= len(value) <= 3:
        raise ValueError(f"{name} requires one to three dimensions")
    value = value + (1,) * (3 - len(value))
    for i, dimension in enumerate(value):
        _integer(dimension, f"{name}[{i}]", minimum=1, maximum=min(limits[i], (1 << 32) - 1))
    return value


def _stream_handle(context, stream):
    if stream is None:
        return _P()
    if not isinstance(stream, Stream):
        raise TypeError("stream must be a Stream or None")
    if stream.context is not context:
        raise ValueError("stream belongs to a different CUDA context")
    stream._ensure_open()
    return stream.handle


_SCALARS = (C.c_int8, C.c_uint8, C.c_int16, C.c_uint16, C.c_int32, C.c_uint32,
            C.c_int64, C.c_uint64, C.c_float, C.c_double)


class Kernel:
    """Configure and asynchronously launch a PTX entry with explicit ctypes ABI.

    ``launch(grid, block, args, shared_bytes=0, stream=None)`` uses one-to-three
    dimensional grid/block tuples (or an integer for x). No raw integer pointers
    or untyped Python scalar arguments are accepted. Generic kernels cannot
    infer their PTX signature: callers must supply the exact order and widths.
    """
    def __init__(self, module, name):
        self.module, self.context, self.name = module, module.context, name
        self.handle = _P()
        with self.context._activate():
            module._ensure_open()
            self.context.driver.call("cuModuleGetFunction", C.byref(self.handle), module.handle, name.encode())
            result = _I32()
            self.context.driver.call("cuFuncGetAttribute", C.byref(result), 0, self.handle)
            self.max_threads = result.value
            self.context.driver.call("cuFuncGetAttribute", C.byref(result), 1, self.handle)
            self.static_shared_bytes = result.value

    def launch(self, grid, block, args=(), *, shared_bytes=0, stream=None):
        grid = _dimensions(grid, "grid", self.context.max_grid)
        block = _dimensions(block, "block", self.context.max_block)
        if math.prod(block) > min(self.max_threads, self.context.max_threads_per_block):
            raise ValueError("block exceeds kernel/device thread limit")
        _integer(shared_bytes, "shared_bytes", maximum=self.context.max_shared_bytes)
        if shared_bytes + self.static_shared_bytes > self.context.max_shared_bytes:
            raise ValueError("static plus dynamic shared memory exceeds device limit")
        with self.context._activate():
            self.module._ensure_open()
            stream_handle = _stream_handle(self.context, stream)
            storage = []
            for arg in args:
                if isinstance(arg, (DeviceTensor, DeviceBuffer)):
                    if arg.context is not self.context:
                        raise ValueError("kernel argument belongs to a different CUDA context")
                    storage.append(_U64(arg.pointer))
                elif type(arg) in _SCALARS:
                    # Copy so mutation of a caller-owned ctypes scalar cannot alter
                    # this launch's host parameter storage during the driver call.
                    storage.append(type(arg)(arg.value))
                else:
                    raise TypeError("kernel arguments require device buffers/tensors or explicit ctypes numeric scalars")
            params = (_P * len(storage))(*(C.cast(C.pointer(value), _P).value for value in storage))
            self.context.driver.call("cuLaunchKernel", self.handle, *grid, *block, shared_bytes,
                                     stream_handle, params, None)
            # Both storage and params remain live until cuLaunchKernel has copied
            # their values. Device resources remain context-owned after return.
        return self


class Stream(_Resource):
    """CUDA stream; nonblocking streams are the default (no legacy-stream waits)."""
    def __init__(self, context, *, nonblocking=True):
        if type(nonblocking) is not bool:
            raise TypeError("nonblocking must be bool")
        self.handle = _P()
        with context._activate():
            context.driver.call("cuStreamCreate", C.byref(self.handle), 1 if nonblocking else 0)
            self._setup(context)

    def synchronize(self):
        with self.context._activate():
            self._ensure_open()
            self.context.driver.call("cuStreamSynchronize", self.handle)

    def query(self):
        """Return False for CUDA_ERROR_NOT_READY, propagate all other errors."""
        with self.context._activate():
            self._ensure_open()
            code = self.context.driver.cuStreamQuery(self.handle)
            if code == 600:
                return False
            self.context.driver.check(code, "cuStreamQuery")
            return True

    def wait_event(self, event):
        """Enqueue a dependency on an already-recorded event from this context."""
        with self.context._activate():
            self._ensure_open()
            _check_event(self.context, event)
            event._ensure_recorded()
            self.context.driver.call("cuStreamWaitEvent", self.handle, event.handle, 0)

    def _destroy_current(self):
        self.context.driver.call("cuStreamDestroy_v2", self.handle)
        self._mark_closed()
        self.handle = _P()


def _check_event(context, event):
    if not isinstance(event, Event):
        raise TypeError("event must be an Event")
    if event.context is not context:
        raise ValueError("event belongs to a different CUDA context")
    event._ensure_open()


class Event(_Resource):
    """CUDA event for ordering/timing; elapsed_ms waits for the end event."""
    def __init__(self, context, *, timing=True):
        if type(timing) is not bool:
            raise TypeError("timing must be bool")
        self.handle, self.timing, self._recorded = _P(), timing, False
        with context._activate():
            context.driver.call("cuEventCreate", C.byref(self.handle), 0 if timing else 2)
            self._setup(context)

    def _ensure_recorded(self):
        self._ensure_open()
        if not self._recorded:
            raise ValueError("event has not been recorded")

    def record(self, stream=None):
        with self.context._activate():
            self._ensure_open()
            self.context.driver.call("cuEventRecord", self.handle, _stream_handle(self.context, stream))
            self._recorded = True
        return self

    def synchronize(self):
        with self.context._activate():
            self._ensure_recorded()
            self.context.driver.call("cuEventSynchronize", self.handle)

    def query(self):
        with self.context._activate():
            self._ensure_recorded()
            code = self.context.driver.cuEventQuery(self.handle)
            if code == 600:
                return False
            self.context.driver.check(code, "cuEventQuery")
            return True

    def elapsed_ms(self, end):
        """Return driver event time in milliseconds, synchronizing ``end`` first."""
        with self.context._activate():
            self._ensure_recorded()
            _check_event(self.context, end)
            end._ensure_recorded()
            if not self.timing or not end.timing:
                raise ValueError("elapsed time requires timing-enabled events")
            self.context.driver.call("cuEventSynchronize", end.handle)
            elapsed = C.c_float()
            self.context.driver.call("cuEventElapsedTime", C.byref(elapsed), self.handle, end.handle)
            return elapsed.value

    def _destroy_current(self):
        self.context.driver.call("cuEventDestroy_v2", self.handle)
        self._mark_closed()
        self.handle = _P()


_INT_MAX = (1 << 31) - 1


def _i32(value, name, *, minimum=0):
    return _I32(_integer(value, name, minimum=minimum, maximum=_INT_MAX))


def _f32(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = C.c_float(value)
    if not math.isfinite(converted.value) or (positive and converted.value <= 0):
        raise ValueError(f"{name} must be finite float32" + (" and positive" if positive else ""))
    return converted


def _overlap(a, b):
    return a.nbytes > 0 and b.nbytes > 0 and max(a.pointer, b.pointer) < min(a.pointer + a.nbytes, b.pointer + b.nbytes)


def _separate(output, inputs, *, exact_alias=False):
    for source in inputs:
        if _overlap(output, source):
            exact = output.pointer == source.pointer and output.nbytes == source.nbytes
            if not (exact_alias and exact):
                raise ValueError("unsupported overlapping device tensor ranges")


class GLMKernels:
    """Validated float32 wrappers for the entries in ``gpu/kernels.h``.

    Construct with ``GLMKernels(ctx.load_ptx(path))``. For packed FP8, supply
    ``fp8_module=ctx.load_ptx(fp8_path)`` or call ``load_fp8(fp8_path)``. PTX targets sm80, so this
    wrapper requires compute capability >= 8.0. All methods enqueue launches
    asynchronously on ``stream`` (None selects the legacy default stream).
    Optional output tensors allow reuse; omitted outputs are context-owned
    allocations. No tensors are converted, read back for validation, or evaluated
    on the CPU. Inputs/intermediates must be finite except documented softmax
    masks; callers supply causal MLA selection IDs. IDs outside [0,tokens) are masked by the kernel.

    Shapes: RMSNorm x/residual [rows,dim], weight [dim]; SwiGLU arbitrary equal
    shapes; RoPE x [heads,even_dim], cosine/sine [dim/2]; matvec weight [rows,cols],
    x [cols]. MLA q_latent [heads,rank], q_rope [heads,rope_dim], cache_latent
    [tokens,rank], cache_rope [tokens,rope_dim], selected int32 [count] (shared) or
    [heads,stride] (per-head). Optional count selects a prefix of each list.
    Partial scratch has shapes [heads,splits,rank] and [heads,splits,2].

    ``residual_rmsnorm`` returns (out,residual_out), ``mla_partial`` returns
    (numerator,stats), and other methods return out. ``mla_decode`` submits partial
    then merge on the same stream; optionally supply numerator/stats/out to reuse
    scratch. Automatically allocated scratch lives until context close, so reuse
    explicit scratch in long-running decode loops. Launch dimensions and all
    index arithmetic are checked for signed 32-bit overflow.
    """
    def __init__(self, module, *, fp8_module=None):
        if not isinstance(module, Module):
            raise TypeError("module must be a CUDA Module")
        module._ensure_open()
        if module.context.compute_capability < (8, 0):
            raise CUDAUnavailable("GLM Vx kernels require compute capability >= 8.0 (sm80 PTX)")
        self.module, self.context = module, module.context
        self._entries = {name: module.kernel("glm_vx_gpu_" + name) for name in (
            "residual_rmsnorm", "swiglu", "rope_glm", "matvec", "mla_partial", "mla_merge")}
        self._fp8_kernel = None
        if fp8_module is not None:
            self._attach_fp8(fp8_module)

    def _tensor(self, tensor, name, *, shape=None, ndim=None, dtype="float32", index64=False):
        if not isinstance(tensor, DeviceTensor):
            raise TypeError(f"{name} must be a DeviceTensor")
        tensor.allocation._ensure_open()
        if tensor.context is not self.context:
            raise ValueError(f"{name} belongs to a different CUDA context")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if (shape is not None and tensor.shape != tuple(shape)) or (ndim is not None and len(tensor.shape) != ndim):
            raise ValueError(f"{name} has incompatible shape {tensor.shape}")
        _integer(tensor.size, f"{name} element count", maximum=(1 << 63) - 1 if index64 else _INT_MAX)
        for dim in tensor.shape:
            _integer(dim, f"{name} dimension", maximum=_INT_MAX)
        return tensor

    def _out(self, tensor, shape, name="out"):
        return self.context.tensor(shape) if tensor is None else self._tensor(tensor, name, shape=shape)

    def _launch(self, name, grid, block, args, stream):
        _integer(grid * block, "launch stride", minimum=1, maximum=_INT_MAX)
        if name not in self._entries:
            self._entries[name] = self.module.kernel("glm_vx_gpu_" + name)
        self._entries[name].launch(grid, block, args, stream=stream)

    def residual_rmsnorm(self, x, residual, weight, eps=1e-6, *, out=None, residual_out=None, stream=None):
        self._tensor(x, "x", ndim=2)
        rows, dim = x.shape
        _i32(rows, "rows", minimum=1)
        _i32(dim, "dim", minimum=1)
        _integer(dim + 255, "rounded dim", maximum=_INT_MAX)
        self._tensor(residual, "residual", shape=x.shape)
        self._tensor(weight, "weight", shape=(dim,))
        eps = _f32(eps, "eps", positive=True)
        out = self._out(out, x.shape)
        residual_out = self._out(residual_out, x.shape, "residual_out")
        _separate(out, [residual_out, weight])
        _separate(residual_out, [weight])
        for destination in (out, residual_out):
            _separate(destination, [x, residual], exact_alias=True)
        self._launch("residual_rmsnorm", rows, 256,
                     [out, residual_out, x, residual, weight, _I32(rows), _I32(dim), eps], stream)
        return out, residual_out

    def swiglu(self, gate, up, *, out=None, stream=None):
        self._tensor(gate, "gate")
        self._tensor(up, "up", shape=gate.shape)
        n = gate.size
        grid = max(1, min((n + 255) // 256, 65535))
        # Kernel increments once after its final iteration as well.
        _integer(((n + grid * 256 - 1) // (grid * 256) + 1) * grid * 256 - 1,
                 "grid-stride index upper bound", maximum=_INT_MAX)
        out = self._out(out, gate.shape)
        _separate(out, [gate, up], exact_alias=True)
        self._launch("swiglu", grid, 256, [out, gate, up, _I32(n)], stream)
        return out

    def rope_glm(self, x, cosine, sine, *, out=None, stream=None):
        self._tensor(x, "x", ndim=2)
        heads, dim = x.shape
        _i32(heads, "heads", minimum=1)
        _i32(dim, "dim", minimum=1)
        if dim % 2:
            raise ValueError("RoPE dim must be even")
        self._tensor(cosine, "cosine", shape=(dim // 2,))
        self._tensor(sine, "sine", shape=(dim // 2,))
        n = heads * (dim // 2)
        grid = max(1, min((n + 255) // 256, 65535))
        _integer(((n + grid * 256 - 1) // (grid * 256) + 1) * grid * 256 - 1,
                 "grid-stride index upper bound", maximum=_INT_MAX)
        out = self._out(out, x.shape)
        _separate(out, [x, cosine, sine])
        self._launch("rope_glm", grid, 256, [out, x, cosine, sine, _I32(heads), _I32(dim)], stream)
        return out

    def matvec(self, weight, x, *, out=None, stream=None):
        self._tensor(weight, "weight", ndim=2)
        rows, cols = weight.shape
        _i32(rows, "rows", minimum=1)
        _integer(cols + 63, "warp loop upper bound", maximum=_INT_MAX)
        self._tensor(x, "x", shape=(cols,))
        out = self._out(out, (rows,))
        _separate(out, [weight, x])
        self._launch("matvec", (rows + 3) // 4, 128, [out, weight, x, _I32(rows), _I32(cols)], stream)
        return out

    def mla_partial(self, q_latent, q_rope, cache_latent, cache_rope, selected, *,
                    splits=4, scale=1.0, count=None, numerator=None, stats=None, stream=None):
        self._tensor(q_latent, "q_latent", ndim=2)
        heads, rank = q_latent.shape
        _i32(heads, "heads", minimum=1)
        _integer(rank, "rank", minimum=1, maximum=1024)
        self._tensor(q_rope, "q_rope", ndim=2)
        rope_dim = q_rope.shape[1]
        self._tensor(q_rope, "q_rope", shape=(heads, rope_dim))
        self._tensor(cache_latent, "cache_latent", ndim=2)
        tokens = cache_latent.shape[0]
        self._tensor(cache_latent, "cache_latent", shape=(tokens, rank))
        self._tensor(cache_rope, "cache_rope", shape=(tokens, rope_dim))
        self._tensor(selected, "selected", dtype="int32")
        if len(selected.shape) == 1:
            capacity, stride = selected.shape[0], 0
        elif len(selected.shape) == 2 and selected.shape[0] == heads:
            capacity = stride = selected.shape[1]
        else:
            raise ValueError("selected must have shape [count] or [heads,stride]")
        if count is None:
            count = capacity
        _integer(count, "count", maximum=capacity)
        _i32(splits, "splits", minimum=1)
        _integer(heads * splits * max(rank, 2), "MLA scratch index upper bound", maximum=_INT_MAX)
        _integer(count + 2 * splits, "MLA selection loop upper bound", maximum=_INT_MAX)
        block = 32 * ((rank + 31) // 32)
        _integer(rope_dim + 2 * block, "MLA rope loop upper bound", maximum=_INT_MAX)
        scale = _f32(scale, "scale")
        numerator = self._out(numerator, (heads, splits, rank), "numerator")
        stats = self._out(stats, (heads, splits, 2), "stats")
        inputs = [q_latent, q_rope, cache_latent, cache_rope, selected]
        _separate(numerator, inputs + [stats])
        _separate(stats, inputs)
        self._launch("mla_partial", heads * splits, block,
                     [numerator, stats, *inputs, _I32(heads), _I32(rank), _I32(rope_dim), _I32(tokens),
                      _I32(count), _I32(stride), _I32(splits), scale], stream)
        return numerator, stats

    def mla_merge(self, numerator, stats, *, out=None, stream=None):
        self._tensor(numerator, "numerator", ndim=3)
        heads, splits, rank = numerator.shape
        for value, name in ((heads, "heads"), (splits, "splits"), (rank, "rank")):
            _i32(value, name, minimum=1)
        _integer(rank + 255, "rounded rank", maximum=_INT_MAX)
        self._tensor(stats, "stats", shape=(heads, splits, 2))
        out = self._out(out, (heads, rank))
        _separate(out, [numerator, stats])
        self._launch("mla_merge", heads, 256, [out, numerator, stats, _I32(heads), _I32(rank), _I32(splits)], stream)
        return out

    def mla_decode(self, q_latent, q_rope, cache_latent, cache_rope, selected, *, splits=4,
                   scale=1.0, count=None, numerator=None, stats=None, out=None, stream=None):
        """Run partial+merge on one stream. Caller owns causal token selection.

        Output must not overlap any input or scratch; preallocated scratch is
        recommended for repeated calls. No synchronization occurs on return.
        """
        self._tensor(q_latent, "q_latent", ndim=2)
        if out is not None:
            self._tensor(out, "out", shape=q_latent.shape)
            inputs = [q_latent, q_rope, cache_latent, cache_rope, selected]
            scratch = [t for t in (numerator, stats) if t is not None]
            # Validate kinds before address-range inspection.
            for tensor in inputs + scratch:
                if not isinstance(tensor, DeviceTensor):
                    raise TypeError("MLA inputs must be DeviceTensor objects")
            _separate(out, inputs + scratch)
        numerator, stats = self.mla_partial(q_latent, q_rope, cache_latent, cache_rope, selected,
            splits=splits, scale=scale, count=count, numerator=numerator, stats=stats, stream=stream)
        return self.mla_merge(numerator, stats, out=out, stream=stream)

    def rmsnorm(self, x, weight, eps=1e-6, *, out=None, stream=None):
        """Normalize one float32 vector using weight[n]; exact out==x allowed.

        Uses the original single-vector ABI. Use rmsnorm_rows for batched input.
        """
        self._tensor(x, "x", ndim=1)
        n = x.size
        _i32(n, "n", minimum=1)
        _integer(n + 255, "rounded n", maximum=_INT_MAX)
        self._tensor(weight, "weight", shape=x.shape)
        eps = _f32(eps, "eps", positive=True)
        out = self._out(out, x.shape)
        _separate(out, [x], exact_alias=True)
        _separate(out, [weight])
        self._launch("rmsnorm", 1, 256, [out, x, weight, _I32(n), eps], stream)
        return out

    def softmax(self, x, *, out=None, stream=None):
        """Stable softmax of one nonempty vector; exact in-place allowed.

        Negative-infinity masks require at least one finite element. NaN,
        positive infinity and all-masked inputs are invalid; callers enforce
        this device-value contract. Use softmax_rows for batched inputs.
        """
        self._tensor(x, "x", ndim=1)
        n = x.size
        _i32(n, "n", minimum=1)
        _integer(n + 255, "rounded n", maximum=_INT_MAX)
        out = self._out(out, x.shape)
        _separate(out, [x], exact_alias=True)
        self._launch("softmax", 1, 256, [out, x, _I32(n)], stream)
        return out

    def rmsnorm_rows(self, x, weight, eps=1e-6, *, out=None, stream=None):
        """RMSNorm over the final dimension of x[...,dim], with weight[dim].

        Leading dimensions flatten to rows; a vector has one row. Returns the
        same shape as x. Exact out==x is allowed; output/weight overlap is not.
        """
        self._tensor(x, "x")
        dim, rows = x.shape[-1], math.prod(x.shape[:-1])
        _i32(rows, "rows", minimum=1)
        _i32(dim, "dim", minimum=1)
        _integer(dim + 255, "rounded dim", maximum=_INT_MAX)
        self._tensor(weight, "weight", shape=(dim,))
        eps = _f32(eps, "eps", positive=True)
        out = self._out(out, x.shape)
        _separate(out, [x], exact_alias=True)
        _separate(out, [weight])
        self._launch("rmsnorm_rows", rows, 256, [out, x, weight, _I32(rows), _I32(dim), eps], stream)
        return out

    def softmax_rows(self, x, *, out=None, stream=None):
        """Stable rowwise softmax over x's final dimension; in-place allowed.

        Leading dimensions flatten to rows; a vector has one row. Negative
        infinity masks are accepted when each row contains a finite element.
        NaN, positive infinity, and all-masked rows are invalid. Values stay on
        device; callers must enforce this value-domain contract.
        """
        self._tensor(x, "x")
        dim, rows = x.shape[-1], math.prod(x.shape[:-1])
        _i32(rows, "rows", minimum=1)
        _i32(dim, "dim", minimum=1)
        _integer(dim + 255, "rounded dim", maximum=_INT_MAX)
        out = self._out(out, x.shape)
        _separate(out, [x], exact_alias=True)
        self._launch("softmax_rows", rows, 256, [out, x, _I32(rows), _I32(dim)], stream)
        return out

    def router(self, logits, bias, k, scale=1.0, *, indices=None, weights=None, stream=None):
        """Return (int32 indices[k], float32 weights[k]) for one expert router.

        Supports 1..256 experts, 1..min(8,experts) selected experts. Ranking uses
        sigmoid(logits)+bias, breaking ties by lower index; weights normalize
        selected original sigmoid values using sum+1e-20 then multiply by scale.
        Inputs must be finite. Outputs cannot overlap each other or either input.
        """
        self._tensor(logits, "logits", ndim=1)
        experts = logits.size
        _integer(experts, "experts", minimum=1, maximum=256)
        _integer(k, "k", minimum=1, maximum=min(8, experts))
        self._tensor(bias, "bias", shape=logits.shape)
        scale = _f32(scale, "scale")
        indices = self.context.tensor((k,), "int32") if indices is None else self._tensor(indices, "indices", shape=(k,), dtype="int32")
        weights = self._out(weights, (k,), "weights")
        _separate(indices, [weights, logits, bias])
        _separate(weights, [logits, bias])
        self._launch("router", 1, 32, [indices, weights, logits, bias, _I32(experts), _I32(k), scale], stream)
        return indices, weights

    def _attach_fp8(self, module):
        if not isinstance(module, Module):
            raise TypeError("fp8_module must be a CUDA Module")
        module._ensure_open()
        if module.context is not self.context:
            raise ValueError("FP8 module belongs to a different CUDA context")
        self._fp8_kernel = module.kernel("glm_vx_gpu_fp8_matvec")

    def load_fp8(self, path):
        """Load an optional FP8 PTX module into this context; return self.

        Its entry must be glm_vx_gpu_fp8_matvec. The context owns the module;
        replacing it leaves the previous module owned until explicit/context
        close, preserving other callers' handles and pending launches.
        """
        module = self.context.load_ptx(path)
        try:
            self._attach_fp8(module)
        except BaseException:
            module.close()
            raise
        return self

    def fp8_matvec(self, weight, scales, x, *, out=None, stream=None):
        """Packed e4m3fn projection; weights remain uint8 device storage.

        Requires an optional FP8 module supplied at construction or load_fp8().
        weight:[rows,cols] uint8, scales:[ceil(rows/128),ceil(cols/128)] float32,
        x:[cols] float32, out:[rows] float32. Launch uses 128 threads (four warps),
        one warp per row; decode and scale application happen inside the kernel.
        Indexing is 64-bit in this entry, while each dimension is signed int32.
        FP8 NaN encodings and nonfinite float inputs/intermediates are outside the
        supported finite-input contract; validation does not read device values.
        """
        if self._fp8_kernel is None:
            raise RuntimeError("FP8 module is not loaded; supply fp8_module or call load_fp8(path)")
        self._tensor(weight, "weight", ndim=2, dtype="uint8", index64=True)
        rows, cols = weight.shape
        _i32(rows, "rows", minimum=1)
        self._tensor(scales, "scales", shape=((rows + 127) // 128, (cols + 127) // 128), index64=True)
        self._tensor(x, "x", shape=(cols,))
        out = self._out(out, (rows,))
        _separate(out, [weight, scales, x])
        self._fp8_kernel.launch((rows + 3) // 4, 128,
                               [out, weight, scales, x, _I32(rows), _I32(cols)], stream=stream)
        return out
