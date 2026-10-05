#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${VXC:=vxc}"
: "${CXX:=g++}"
: "${CLANG:=clang}"
mkdir -p gpu/build
# Capture before compilation, then reject concurrent source changes. Tests verify
# both these exact sources and the resulting binary instead of trusting existence.
sha256sum gpu/kernels.vx gpu/fp8.vx gpu/gemm.vx gpu/testing/simt.cpp gpu/testing/build_simt.sh > gpu/build/simt-inputs.sha256
"$VXC" gpu/kernels.vx --action emit-llvm > gpu/build/kernels-host.mlir
mlir-translate --mlir-to-llvmir gpu/build/kernels-host.mlir > gpu/build/kernels-host.ll
"$CLANG" -O2 -fPIC -c gpu/build/kernels-host.ll -o gpu/build/kernels-host.o
"$VXC" gpu/fp8.vx --action emit-llvm > gpu/build/fp8-host.mlir
mlir-translate --mlir-to-llvmir gpu/build/fp8-host.mlir > gpu/build/fp8-host.ll
"$CLANG" -O2 -fPIC -c gpu/build/fp8-host.ll -o gpu/build/fp8-host.o
"$VXC" gpu/gemm.vx --action emit-llvm > gpu/build/gemm-host.mlir
mlir-translate --mlir-to-llvmir gpu/build/gemm-host.mlir > gpu/build/gemm-host.ll
"$CLANG" -O2 -fPIC -c gpu/build/gemm-host.ll -o gpu/build/gemm-host.o
"$CXX" -std=c++20 -O2 -fPIC -shared -pthread gpu/testing/simt.cpp \
    gpu/build/kernels-host.o gpu/build/fp8-host.o gpu/build/gemm-host.o -lm -o gpu/build/libsimt.so

sha256sum -c gpu/build/simt-inputs.sha256 > /dev/null
cat gpu/build/simt-inputs.sha256 > gpu/build/simt-build.sha256.tmp
sha256sum gpu/build/libsimt.so >> gpu/build/simt-build.sha256.tmp
mv gpu/build/simt-build.sha256.tmp gpu/build/simt-build.sha256
