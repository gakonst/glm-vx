#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LLAMA_CPP_SOURCE:?Set LLAMA_CPP_SOURCE to the pinned llama.cpp source tree}"
: "${LLAMA_CPP_BUILD:=$LLAMA_CPP_SOURCE/build}"
: "${CXX:=c++}"
mkdir -p build/parity
"$CXX" -O2 -std=c++17 validation/native/llama_trace.cpp \
  -I "$LLAMA_CPP_SOURCE/include" -I "$LLAMA_CPP_SOURCE/ggml/include" \
  -L "$LLAMA_CPP_BUILD/bin" -Wl,-rpath,"$LLAMA_CPP_BUILD/bin" \
  -lllama -lggml -lggml-base -o build/parity/llama-trace
