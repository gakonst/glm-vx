#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${VXC:=vxc}"
: "${CXX:=g++}"
: "${CLANG:=clang}"
mkdir -p gpu/build
"$VXC" gpu/kernels.vx --action emit-llvm > gpu/build/kernels-host.mlir
mlir-translate --mlir-to-llvmir gpu/build/kernels-host.mlir > gpu/build/kernels-host.ll
"$CLANG" -O2 -fPIC -c gpu/build/kernels-host.ll -o gpu/build/kernels-host.o
"$VXC" gpu/fp8.vx --action emit-llvm > gpu/build/fp8-host.mlir
mlir-translate --mlir-to-llvmir gpu/build/fp8-host.mlir > gpu/build/fp8-host.ll
"$CLANG" -O2 -fPIC -c gpu/build/fp8-host.ll -o gpu/build/fp8-host.o
"$CXX" -std=c++20 -O2 -fPIC -shared -pthread gpu/testing/simt.cpp \
    gpu/build/kernels-host.o gpu/build/fp8-host.o -lm -o gpu/build/libsimt.so
