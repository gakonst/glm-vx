# Local Vx toolchain

Installed official `vx-lang/Vx` release `v0.0.2` (published 2026-09-21), portable Linux x86_64 artifact. Its downloaded SHA-256 matched the published checksum. This is the release compiler, not a build of current upstream HEAD. Upstream inspected: `48da4a1aea4cda37599648865531e5b62b82ae29`.

All new toolchain files are contained in this directory. No sudo, package installation into system directories, profile changes, or upstream source edits were performed. Debian packages were downloaded and extracted locally. Space used is about 2.7 GiB including archives and development packages.

## Commands

Tool API mount root `/linux-paradigm` corresponds to native filesystem `/srv/nanocodex/workspace`. Use native absolute paths *inside shell commands*; use the logical mount path for tool `workdir` routing.

```bash
# Works from any current directory, no manual environment setup:
/srv/nanocodex/workspace/vx-toolchain/bin/vxc --run /srv/nanocodex/workspace/vx-toolchain/samples/hello.vx
/srv/nanocodex/workspace/vx-toolchain/bin/vxc --run /srv/nanocodex/workspace/vx-toolchain/samples/matmul.vx

# Environment for direct LLVM commands, linking, or optional upstream builds:
source /srv/nanocodex/workspace/vx-toolchain/env.sh
llvm-config --version  # 22.1.8
z3 --version           # 4.8.12

# Rebuild a real native object and linked executable:
/srv/nanocodex/workspace/vx-toolchain/samples/build-hello.sh
/srv/nanocodex/workspace/vx-toolchain/samples/run-hello-native.sh
```

Local LLVM tools: `llvm22/usr/lib/llvm-22/bin/{clang,clang++,llvm-config,mlir-translate,opt,llc,lld}`. Z3 is `llvm22/usr/bin/z3`. `env.sh` also exposes the host's pre-existing Rust 1.97.1 under `/srv/nanocodex/.rustup/toolchains/1.97-x86_64-unknown-linux-gnu/bin`.

## Verified execution

- `logs/hello-jit.log`: Vx compiled via LLVM IR, object generation and linking, executed native binary, printed `42`, compiler exit 0.
- `logs/docker-smoke.log`: upstream `examples/docker_smoke.vx` compiled and executed, printed `24`, exit 0 (this upstream example is scalar arithmetic).
- `logs/matmul-jit.log`: actual `Tensor<f32,[4,4]>` matrix multiplication of tensors filled with 2 and 3 printed `24`, exit 0.
- `logs/hello-object.log`: `vxc -c` generated `samples/hello.o`, verified ELF x86-64 relocatable.
- `logs/hello-aot.log` and `logs/hello-native.log`: object linked with Clang and local runtime/MLIR libraries; the resulting ELF executable ran and printed `42`, exit 0 with environment loaded.
- `logs/help.txt`: compiler CLI from installed binary.

## Limitations

- CPU portable release selected. CUDA GPU execution, accelerator hardware, Enzyme autodiff, full upstream test suite, and GLM model serving are not validated here.
- `VX_DISABLE_CUDA=1` in the environment intentionally selects CPU behavior for a future source build. No CUDA toolkit/GPU was provisioned.
- `z3` executable works, but no full seam-verification suite was run.
- Direct standalone executable launch requires sourcing `env.sh` because the local LLVM shared libraries depend on locally extracted `libz3.so.4`. The provided `run-hello-native.sh` does that automatically. The executable is not a self-contained distributable.
- In this release, `vxc SOURCE -o OUTPUT` still selects default run-jit and does not create OUTPUT. Use `vxc -c SOURCE -o OUTPUT.o` followed by native linking (see `samples/build-hello.sh`).
- No claim that current upstream HEAD compiles from source: dependencies were installed but the runnable compiler is the official v0.0.2 binary.

## Sources

Read upstream `README.md`, `docs/INSTALL.md`, `config.template`, `setup.sh`, `www/install.sh`, `scripts/provision/setup_linux.sh`, and relevant packaging/build code.

- https://github.com/vx-lang/Vx/releases/tag/v0.0.2
- https://github.com/vx-lang/Vx/blob/main/docs/INSTALL.md
- https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-22/main/binary-amd64/Packages.gz

`downloads/vx-release.json`, `downloads/vx.sha256`, and `downloads/packages.json` retain version and checksum provenance. LLVM packages were checked against SHA-256 values in the official HTTPS package index.

## Shared-library C ABI confirmed

`samples/abi.vx` defines an ordinary top-level function (no `main`, no `extern` or attribute):

```rust
fn vx_scale(data: *mut f32, n: i64, scale: f32) -> i32 {
    unsafe {
        for i in 0..n {
            data[i] = data[i] * scale;
        }
    }
    return 0;
}
```

Verified commands from this directory:

```bash
source env.sh
bin/vxc -c samples/abi.vx -o samples/abi.o
clang++ -shared samples/abi.o -o samples/libabi.so
```

`nm -g` confirms unmangled exported `vx_scale` and an additional `_mlir_vx_scale`. Python ctypes using `(POINTER(c_float), c_longlong, c_float) -> c_int` called `vx_scale`, returned zero, and changed `[1,2,3,4]` to `[3,6,9,12]`. `logs/abi-ctypes.log` records actual execution. The pointer-only kernel needs no Vx or MLIR runtime library linkage and the generated object linked directly as a shared library. Libraries listed above for hello printing are needed only when those operations produce corresponding runtime symbol references. Read `docs/lang/abi.md` and upstream pointer tests for ABI/source context.
