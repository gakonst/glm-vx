#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${VXC:=vxc}"
: "${CC:=cc}"
# Default stays O0. Optimized packed candidates are explicit and independently tested.
: "${VX_OPT_LEVEL:=0}"
: "${VX_BUILD_DIR:=build}"
case "$VX_OPT_LEVEL" in 0|1|2|3) ;; *) echo 'VX_OPT_LEVEL must be 0..3' >&2; exit 2 ;; esac
mkdir -p "$VX_BUILD_DIR"
# No fallback implementation: this object must be emitted by the Vx compiler.
"$VXC" kernels.vx -O "$VX_OPT_LEVEL" --action emit-obj -o "$VX_BUILD_DIR/kernels.o"
"$VXC" packed.vx -O "$VX_OPT_LEVEL" --action emit-obj -o "$VX_BUILD_DIR/packed.o"
"$CC" -shared -o "$VX_BUILD_DIR/libglm_vx.so" "$VX_BUILD_DIR/kernels.o" "$VX_BUILD_DIR/packed.o" -lm
nm -D --defined-only "$VX_BUILD_DIR/libglm_vx.so" | grep ' glm_vx_'
