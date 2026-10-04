# CPU-only serving validation

Tested source: `949ed34734a79c61c6744b22729b5f10ff204357`. AMD EPYC 4484PX, 12 cores/24 logical CPUs, one model worker and BLAS thread count 1. Concurrency below means HTTP requests, not CPU threads.

The workload is the tiny random-weight GLM-shaped model (four layers, hidden size 32, vocabulary 256). It validates execution and serving mechanics, not trained GLM-5.3 answer quality or full-model performance.

Each cell summarizes three runs, eight requests per run, alternating 8/128-token prompts, 16 generated tokens, and one excluded warmup request. Percentiles are medians of per-run percentiles. Throughput includes prefill and HTTP overhead.

| Backend | Concurrent requests | Output tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | Token gap p95 (ms) |
|---|---:|---:|---:|---:|---:|
| vx | 1 | 94.4 | 128.6 | 260.9 | 3.4 |
| vx | 2 | 93.5 | 173.1 | 385.3 | 13.6 |
| vx | 4 | 93.7 | 392.6 | 968.6 | 14.4 |
| vx | 8 | 100.1 | 621.9 | 1164.5 | 19.0 |
| numpy-reference | 1 | 256.6 | 51.7 | 104.2 | 0.9 |
| numpy-reference | 2 | 278.5 | 99.2 | 112.8 | 3.2 |
| numpy-reference | 4 | 269.0 | 118.1 | 216.3 | 20.7 |
| numpy-reference | 8 | 254.9 | 252.9 | 457.2 | 71.0 |

**192/192 measured requests succeeded.** All corresponding generated-token hashes matched across both backends, every concurrency setting and repeat.

The compiled Vx CPU path is slower than the NumPy reference in this workload, and additional HTTP concurrency does not materially raise Vx throughput. This is consistent with the current single model owner and scalar CPU kernels. These measurements do not establish production scalability.

**Correctness:** 170 tests and 37 subtests passed; 50 additional compiled native-kernel checks passed. The GPU-source CPU SIMT tests also ran, but 14 actual GPU tests were explicitly skipped because CUDA is unavailable. No GPU executed.

Reproduce from the repository root:

```sh
source ../vx-toolchain/env.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests kernels/test_backend.py gpu -q -rs
.venv/bin/python kernels/test_kernels.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/cpu-validation/run.py
```

The runner starts each server on an automatically assigned loopback port and stops it afterward. Raw per-request results, output hashes, hardware metadata and test receipts are in this directory. Results use a shared machine and sequential backend runs; no confidence intervals or extrapolation to a full trained checkpoint are claimed.
