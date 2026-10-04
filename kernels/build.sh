#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${VXC:=vxc}"
: "${CC:=cc}"
mkdir -p build
# No fallback implementation: this object must be emitted by the Vx compiler.
"$VXC" kernels.vx --action emit-obj -o build/kernels.o
"$CC" -shared -o build/libglm_vx.so build/kernels.o -lm
nm -D --defined-only build/libglm_vx.so | grep ' glm_vx_'
