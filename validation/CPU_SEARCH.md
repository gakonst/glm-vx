# CPU compiler candidate search

`validation/cpu_search.py` builds explicit O0, O1, O2 and O3 candidates in new,
isolated directories, requires fixed CPU semantic tests before benchmarking,
and selects the fastest passing candidate on matched synthetic workloads.
It never replaces the default library or enables a candidate for serving.
**Every receipt, candidate, and selected winner has `release_eligible=false`.**
The global trained strict numerical gate has not passed; this search cannot
supply model proof, release approval, trained throughput, or any GPU claim.

Run from an updated checkout with the existing Vx toolchain and Python venv:

```bash
source ../vx-toolchain/env.sh
PYTHONPATH=$PWD ../glm-vx/.venv/bin/python -m validation.cpu_search \
  --python ../glm-vx/.venv/bin/python \
  --reference ../glm-vx-reference \
  --output-dir build/cpu-search/run-001 --repeats 7
```

Use a fresh output directory each time; existing evidence is never overwritten.
The Python executable path preserves venv symlinks. The toolchain environment
must already supply Vx's LLVM dependencies. `--vxc` fingerprints the supplied
executable; a wrapper's hash is only a launcher hash. On the documented local
installation, set `VX_LLVM_BIN="$VX_TOOLCHAIN_ROOT/llvm22/usr/lib/llvm-22/bin"`
and pass `--vxc "$VX_TOOLCHAIN_ROOT/release/bin/vxc"` to pin the actual release
binary. Repeated `--toolchain-file /path/to/file` options also pin and recheck
LLVM/linker tools or environment scripts. Transitive LLVM/linker and Python-package binaries are not all attested;
retain the pinned toolchain/dependency environment alongside the receipt. Builds invoke the existing build
script with structured argv and an absolute `VX_BUILD_DIR`; default O0 and
`kernels/build/libglm_vx.so` remain unchanged. Test and benchmark children get
that candidate's `GLM_VX_LIBRARY`, this checkout's `PYTHONPATH`, deterministic
seeds, and one BLAS/OpenMP thread. No shell command strings are interpolated.

## Required gates and provenance

Each candidate must build successfully and pass the fixed model, packed-model,
backend, batch-prefill, top-k, trace, cache, scheduler and HTTP lifecycle tests. Selected synthetic packed-codec tests
also run against the independently pinned native GGML codec reference. The
reference's source hashes and library fingerprint are recorded and rechecked;
missing reference assets fail the run. No trained GGUF rows are accessed.
The exact required files/node IDs are constants in the runner and are included
in its receipt. Missing files, empty collection, missing required test modules,
failures, duplicate identities, collection errors, **any skipped test**, or timeout prevent eligibility.
Pytest uses the explicit root `pyproject.toml` with `addopts` cleared, so ambient
configuration cannot silently select only a subset of every candidate.

One historical model test hardcodes the default library. The runner explicitly
deselects it and runs `test_cpu_candidate_gate.py` instead: that test checks the
candidate path/hash and exercises the independent expanded-model oracle using
that candidate. This is a documented replacement, not a silently skipped test.
Every candidate must report the same test identities as O0 before benchmarking;
partial collection cannot make another candidate appear eligible. Root pytest
configuration files are included in the source freeze. GPU tests are
intentionally outside this CPU gate.

All repository implementation/test/benchmark source files and lookup data in
`glm_vx`, `kernels`, `tests`, `validation`, and `benchmarks` are hashed before and
after the search, and checked between candidates/stages. Generated build,
cache and evidence directories are excluded. The compiler launcher, Python executable, native reference,
default library, and each actual candidate `.so` are fingerprinted. Tests and
benchmarks bind the candidate's hash. A final source/compiler/Python/library drift
check revokes **every** eligibility decision and removes the winner.

The receipt retains structured command argv, exit statuses, timeout state,
logs and log hashes, JUnit evidence, snapshots, actual library paths/hashes,
benchmark samples, rankings, and failure reasons. Timed-out process groups
are terminated so compiler/test children cannot continue producing artifacts.
A failed candidate may be excluded while others continue; O0 itself must pass
all gates to provide a common baseline. Missing/changed baseline evidence
fails the overall selection. A failed search exits nonzero and writes a failed
receipt when it has created its evidence directory.

## Matched measurement and interpretation

Every passing candidate benchmarks the same fixed seeded resident F32 matrix
cases `(batch, output, input) = (8,256,6144), (8,1024,6144), (16,512,2048)` and
the same synthetic 256-hidden-width model prefill prompts of 8 and 32 tokens.
Every run uses fresh KV cache state; initial warmup is excluded. Both O0 and
candidate outputs must be finite; matching NaNs or infinities fail. Each measured
output must exactly equal the O0 output. At least three (default seven)
positive finite timing samples per case are required. Workload IDs, sample
counts, exact-output hashes, library hashes, and reported medians are checked
again by the parent runner before ranking. Each benchmark worker also records
its actual Python/package versions, OS, architecture and CPU affinity count.

The score is the equal-case geometric mean of O0 median/candidate median.
Larger scores win; exact ties prefer the lower optimization level. The selected
artifact is the isolated `.so` path and SHA-256 in `receipt.json`; it is never
copied to the default path or auto-promoted. A successful receipt establishes
only that candidate's eligibility for these measured CPU workloads.

This is a bounded sequential search, not statistically conclusive hardware
tuning. CPU contention, frequency changes and thermal effects can change the
winner. Re-run under controlled conditions before using the measurements.
Packed codec correctness is gated, but the performance score uses matrix and
prefill workloads; it is not a packed-kernel performance ranking. There is no
trained-model, cold-page, serving-load, HTTP-latency, or GPU measurement here.

Runner failure-path tests (no compiler or trained weights required):

```bash
PYTHONPATH=$PWD ../glm-vx/.venv/bin/python -m pytest tests/test_cpu_search.py
```

## Recorded smoke search (2026-10-04)

A real four-build run on parent commit `3d0ec82` used `vxc 0.0.2`, the shared
main venv, and three repeats. Every candidate passed **115 tests, zero skips**
(plus four subtests); the one documented default-library test was deselected
and replaced by the candidate-bound test. The source snapshot covered 76 files,
with no changes; the default library's SHA-256 remained unchanged. The runner
and replacement-test suite separately passed 27 tests.

| Candidate | Matrix 8×256×6144 ms | Matrix 8×1024×6144 ms | Matrix 16×512×2048 ms | Prefill 8 ms | Prefill 32 ms | Geomean speedup |
|---|---:|---:|---:|---:|---:|---:|
| O0 | 6.286 | 23.028 | 7.881 | 34.236 | 108.806 | 1.000× |
| O1 | 4.653 | 17.813 | 5.023 | 24.357 | 82.913 | 1.383× |
| O2 | 4.099 | 16.430 | 5.101 | 24.485 | 84.515 | 1.430× |
| O3 | 3.815 | 15.056 | 5.730 | 24.093 | 87.958 | 1.435× |

O3 was the measured winner, with artifact SHA-256
`cff6023afc49cd7c03b8d30b4f91937494c06cfc1f68c5014bab75dda45d3441`.
O2 and O3 were close; these timings do not establish a reliable ordering under
other load conditions. The complete local receipt and per-candidate logs,
JUnit XML, benchmarks, objects and libraries are in
`build/cpu-search/20261004-initial/`. The selected library is
`O3/build/libglm_vx.so` under that directory. **Release eligibility is false.**
Re-run after merging newer model/backend changes; this evidence binds only
the recorded frozen source hashes and the measured artifact.
