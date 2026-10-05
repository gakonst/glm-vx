#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ../vx-toolchain/env.sh
PYTHON="${PYTHON:-../glm-vx/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
E=evidence/vx-row-tile
for opt in 0 3; do
  suffix=''; target=build-o3
  if [ "$opt" = 0 ]; then suffix=-o0; target=build; fi
  vxc "$E/probe.vx" -O "$opt" --action emit-obj -o "$E/probe$suffix.o" > "$E/probe$suffix-build.txt" 2>&1
  cc -shared "$E/probe$suffix.o" -o "$E/probe$suffix.so"
  VX_OPT_LEVEL="$opt" VX_BUILD_DIR="$target" bash kernels/build.sh > "$E/build-o$opt.txt" 2>&1
  "$PYTHON" kernels/test_kernels.py "kernels/$target/libglm_vx.so" > "$E/kernels-o$opt.txt"
  BENCH_OPT="$opt" "$PYTHON" "$E/bench.py"
done
"$PYTHON" kernels/test_matvec_rows.py kernels/build/libglm_vx.so kernels/build-o3/libglm_vx.so > "$E/correctness.txt"
GLM_VX_LIBRARY=kernels/build-o3/libglm_vx.so "$PYTHON" -m unittest kernels.test_backend tests.test_model > "$E/integration.txt" 2>&1
objdump -d "$E/probe.o" > "$E/probe.asm"
objdump -d --disassemble=glm_vx_matvec kernels/build-o3/kernels.o > "$E/kernels-o3.asm"
vxc "$E/types.vx" -O 3 --action emit-obj -o "$E/types.o" > "$E/types-build.txt" 2>&1
cc -shared "$E/types.o" -o "$E/types.so"
objdump -d "$E/types.o" > "$E/types.asm"
"$PYTHON" "$E/verify_types.py" > "$E/types-results.txt"
# In this installed compiler emit-llvm prints LLVM-dialect MLIR to stdout,
# despite accepting -o; preserve the actual output rather than label it LLVM IR.
vxc "$E/probe.vx" -O 3 --action emit-llvm > "$E/ir-build.txt" 2>&1
