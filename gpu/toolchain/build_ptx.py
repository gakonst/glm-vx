#!/usr/bin/env python3
"""Compile ordinary per-thread Vx functions to NVIDIA PTX; never execute on CPU.

Requires vxc v0.0.2 and LLVM 22 mlir-translate/opt/llc in PATH. Optional
VX_GPU_LLVM_BIN points at the directory containing those LLVM executables.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

PREFIX = "glm_vx_gpu_"
ARCH_PTX = {"sm_70": 76, "sm_72": 76, "sm_75": 76, "sm_80": 76,
            "sm_86": 76, "sm_87": 76, "sm_89": 78, "sm_90": 78, "sm_90a": 80}
INDEX = {
    "vx_gpu_thread_idx_x": "tid.x", "vx_gpu_block_idx_x": "ctaid.x",
    "vx_gpu_block_dim_x": "ntid.x", "vx_gpu_grid_dim_x": "nctaid.x",
}
SIGNATURES = {**{name: ("i32", "") for name in INDEX},
    "vx_gpu_shuffle_down_f32": ("float", "float, i32"),
    "vx_gpu_shuffle_idx_f32": ("float", "float, i32"),
    "vx_gpu_barrier": ("void", ""), "vx_gpu_shared_f32": ("ptr", ""),
    "vx_gpu_mma_tf32_m16n8k8": ("void", "ptr, i32, " + ", ".join(["float"] * 10)),
    "expf": ("float", "float"), "sqrtf": ("float", "float")}
DECL = re.compile(r"^declare\s+(\w+)\s+@([\w.$]+)\(([^)]*)\)(?:[ \t]+[^\n]*)?$", re.M)
DEFINE = re.compile(r"^define\s+(\w+)\s+@([\w.$]+)\(([^)]*)\)([^\n]*)\{", re.M)

def adapter_ir(ir: str, shared_floats: int) -> tuple[str, list[str]]:
    if re.search(r'^target (triple|datalayout)', ir, re.M):
        raise ValueError("expected portable IR without a host target; omit --host/--target")
    definitions = list(DEFINE.finditer(ir))
    names = {m[2] for m in definitions}
    entries = [m[2] for m in definitions if m[2].startswith(PREFIX)]
    if not entries:
        raise ValueError(f"no {PREFIX}* entry definitions")
    for m in definitions:
        if m[2] in entries:
            if m[1] != "void":
                raise ValueError(f"GPU entry {m[2]} must return void")
            for arg in filter(None, map(str.strip, m[3].split(','))):
                if not re.fullmatch(r'(?:ptr|i32|i64|float|double) %[-\w.]+', arg):
                    raise ValueError(f"unsupported entry parameter: {arg}")
    used = set()
    def declaration(m: re.Match) -> str:
        name = m[2]
        if name in SIGNATURES:
            expected = SIGNATURES[name]
            actual = (m[1], ', '.join(x.strip() for x in m[3].split(',')))
            if actual != expected:
                raise ValueError(f"ABI mismatch for {name}: {actual}, expected {expected}")
            used.add(name)
            return ''
        if name.startswith('llvm.'):
            return m[0]
        raise ValueError(f"unsupported external @{name}; no CPU/runtime fallback permitted")
    ir = DECL.sub(declaration, ir)
    # Reject formatting outside the deliberately narrow compiler-output contract.
    if re.search(r'^declare(?!.*@llvm\.)', ir, re.M):
        raise ValueError("unrecognized external declaration syntax")
    calls = set(re.findall(r'\b(?:call|invoke)\s+[^\n@]*@([\w.$]+)\(', ir))
    if calls - names - used - {n for n in calls if n.startswith('llvm.')}:
        raise ValueError("unresolved function call in Vx IR")
    def definition(m: re.Match) -> str:
        linkage = 'ptx_kernel' if m[2] in entries else 'internal'
        return f'define {linkage} {m[1]} @{m[2]}({m[3]}){m[4]}{{'
    ir = DEFINE.sub(definition, ir)
    wrappers = []
    for name, intrinsic in INDEX.items():
        if name in used:
            wrappers.append(f'''declare i32 @llvm.nvvm.read.ptx.sreg.{intrinsic}()
define internal i32 @{name}() alwaysinline {{
  %v = call i32 @llvm.nvvm.read.ptx.sreg.{intrinsic}()
  ret i32 %v
}}''')
    for mode in ('down', 'idx'):
        name = f'vx_gpu_shuffle_{mode}_f32'
        if name in used:
            intrinsic = f'llvm.nvvm.shfl.sync.{mode}.f32'
            wrappers.append(f'''declare float @{intrinsic}(i32, float, i32, i32) convergent
define internal float @{name}(float %value, i32 %lane) alwaysinline convergent {{
  %v = call float @{intrinsic}(i32 -1, float %value, i32 %lane, i32 31)
  ret float %v
}}''')
    if 'vx_gpu_barrier' in used:
        wrappers.append('''declare void @llvm.nvvm.barrier0() convergent
define internal void @vx_gpu_barrier() alwaysinline convergent {
  call void @llvm.nvvm.barrier0()
  ret void
}''')
    if 'vx_gpu_shared_f32' in used:
        wrappers.append(f'''@vx_gpu_shared_storage = internal addrspace(3) global [{shared_floats} x float] undef, align 16
define internal ptr @vx_gpu_shared_f32() alwaysinline {{
  %p = addrspacecast ptr addrspace(3) @vx_gpu_shared_storage to ptr
  ret ptr %p
}}''')
    if 'vx_gpu_mma_tf32_m16n8k8' in used:
        if shared_floats < 320:
            raise ValueError('TF32 GEMM requires at least 320 shared floats')
        # Vx has a scalar C ABI, so bridge the aggregate MMA result through a
        # lane-private four-float output region. No arithmetic kernel is replaced.
        conversions = '\n'.join(
            f'  %{x}t = call i32 asm "cvt.rna.tf32.f32 $0, $1;", "=r,f"(float %{x})'
            for x in ('a0', 'a1', 'a2', 'a3', 'b0', 'b1'))
        stores = '\n'.join(f"  %d{i} = extractvalue {{float, float, float, float}} %d, {i}\n"
            f"  %p{i} = getelementptr float, ptr %base, i32 {i}\n"
            f"  store float %d{i}, ptr %p{i}, align 4" for i in range(4))
        wrappers.append('''define internal void @vx_gpu_mma_tf32_m16n8k8(ptr %out, i32 %offset,
    float %a0, float %a1, float %a2, float %a3, float %b0, float %b1,
    float %c0, float %c1, float %c2, float %c3) alwaysinline convergent {
''' + conversions + '''
  %d = call {float, float, float, float} asm sideeffect
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"(i32 %a0t, i32 %a1t, i32 %a2t, i32 %a3t, i32 %b0t, i32 %b1t, float %c0, float %c1, float %c2, float %c3) convergent
''' + '  %base = getelementptr float, ptr %out, i32 %offset\n' + stores + '\n  ret void\n}')
    if 'expf' in used:
        wrappers.append('''declare float @llvm.nvvm.ex2.approx.ftz.f(float)
define internal float @expf(float %x) alwaysinline {
  %base2 = fmul float %x, 0x3FF7154760000000
  %y = call float @llvm.nvvm.ex2.approx.ftz.f(float %base2)
  ret float %y
}''')
    if 'sqrtf' in used:
        wrappers.append('''declare float @llvm.sqrt.f32(float)
define internal float @sqrtf(float %x) alwaysinline {
  %y = call float @llvm.sqrt.f32(float %x)
  ret float %y
}''')
    return 'target triple = "nvptx64-nvidia-cuda"\n' + ir + '\n' + '\n'.join(wrappers) + '\n', entries

def find_tool(name: str, directory: str | None = None) -> str:
    path = shutil.which(str(Path(directory) / name) if directory else name)
    if not path:
        raise ValueError(f"required executable {name} unavailable; configure PATH/VX_GPU_LLVM_BIN")
    return path

def run(command: list[str], output: Path | None = None) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(f"command failed ({result.returncode}): {command!r}\n{result.stderr}\n{result.stdout}")
    if output:
        output.write_text(result.stdout)
    return result.stdout

def validate_ptx(ptx: str, entries: list[str]) -> dict:
    found = re.findall(r'\.visible\s+\.entry\s+(\w+)\(', ptx)
    if set(found) != set(entries):
        raise ValueError(f"PTX entries differ: expected {entries}, got {found}")
    if '.extern .func' in ptx or re.search(r'\bcall(?:\.uni)?\b', ptx):
        raise ValueError("PTX contains a device call; expected fully inlined device-only kernels")
    if 'glm_vx_gpu_gemm_tf32' in entries:
        body = ptx.split('.visible .entry glm_vx_gpu_gemm_tf32(', 1)[1].split('.visible .entry', 1)[0]
        for token in ('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32',
                      'cvt.rna.tf32.f32', 'ld.shared.', 'st.shared.', 'bar.sync'):
            if token not in body:
                raise ValueError(f'TF32 GEMM is missing required PTX operation: {token}')
    registers = sorted(set(re.findall(r'%(?:tid|ctaid|ntid|nctaid)\.x', ptx)))
    if not registers:
        raise ValueError("PTX has no hardware thread/block indexing")
    return {'entries': found, 'hardware_index_registers': registers,
        'tensor_core_mma': 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx,
        'tf32_rna_conversion': 'cvt.rna.tf32.f32' in ptx,
        'warp_shuffle': 'shfl.sync.' in ptx, 'block_barrier': 'bar.sync' in ptx,
        'shared_memory': '.shared ' in ptx, 'device_exp2': 'ex2.approx.ftz.f32' in ptx,
        'device_sqrt': 'sqrt.rn.f32' in ptx}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch', default='sm_80', choices=sorted(ARCH_PTX))
    parser.add_argument('--shared-floats', type=int, default=256)
    parser.add_argument('--vxc', default=os.environ.get('VXC', 'vxc'))
    parser.add_argument('--keep-ir', action='store_true', help='retain LLVM/MLIR beside output')
    parser.add_argument('--ptxas', help='optionally assemble to cubin with this NVIDIA ptxas executable; failure is fatal')
    args = parser.parse_args()
    try:
        ptx_isa = ARCH_PTX[args.arch]
        if not 33 <= args.shared_floats <= 12288:
            raise ValueError('shared-floats must be in [33,12288] (max 48 KiB)')
        vxc = find_tool(args.vxc)
        ptxas = find_tool(args.ptxas) if args.ptxas else None
        bindir = os.environ.get('VX_GPU_LLVM_BIN')
        translate, opt, llc = (find_tool(t, bindir) for t in ('mlir-translate', 'opt', 'llc'))
        versions = {tool: run([path, '--version']).strip() for tool,path in [('vxc',vxc),('mlir-translate',translate),('opt',opt),('llc',llc)]}
        for tool in ('mlir-translate', 'opt', 'llc'):
            if not re.search(r'LLVM version 22\.', versions[tool]):
                raise ValueError(f'{tool} must use LLVM 22; got {versions[tool].splitlines()[0]}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='vx-ptx-') as temp:
            work = Path(temp)
            run([vxc, '--emit-llvm', str(args.source.resolve())], work/'source.mlir')
            ir = run([translate, '--mlir-to-llvmir', str(work/'source.mlir')], work/'source.ll')
            adapted, entries = adapter_ir(ir, args.shared_floats)
            if "@vx_gpu_mma_tf32_m16n8k8" in ir and int(args.arch[3:5]) < 80:
                raise ValueError("TF32 MMA requires sm_80 or newer")
            (work/'device.ll').write_text(adapted)
            run([opt, '-S', '-passes=always-inline,default<O3>,verify', str(work/'device.ll'), '-o', str(work/'optimized.ll')])
            run([llc, '-march=nvptx64', f'-mcpu={args.arch}', f'-mattr=+ptx{ptx_isa}', '-O3', str(work/'optimized.ll'), '-o', str(work/'kernel.ptx')])
            ptx = (work/'kernel.ptx').read_text()
            evidence = validate_ptx(ptx, entries)
            actual_target = re.search(r'^\.target (\w+)', ptx, re.M)
            actual_version = re.search(r'^\.version ([0-9.]+)', ptx, re.M)
            if not actual_target or actual_target[1] != args.arch or not actual_version:
                raise ValueError('PTX header does not match requested architecture')
            evidence.update({'schema': 'glm-vx-ptx-v1', 'arch': args.arch, 'ptx_version': actual_version[1],
                'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                'shared_floats': args.shared_floats, 'gpu_execution_validated': False,
                'tools': {k:v.splitlines()[0] for k,v in versions.items()}})
            evidence['assembly_validated'] = False
            if ptxas:
                version = run([ptxas, '--version']).strip()
                assembled = subprocess.run([ptxas, f'-arch={args.arch}', '-v',
                    str(work/'kernel.ptx'), '-o', str(work/'kernel.cubin')],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if assembled.returncode:
                    raise ValueError(f'NVIDIA ptxas failed ({assembled.returncode}):\n{assembled.stderr}')
                if not (work/'kernel.cubin').is_file():
                    raise ValueError('ptxas succeeded without producing a cubin')
                (work/'ptxas.log').write_text(assembled.stdout + assembled.stderr)
                evidence.update({'assembly_validated': True, 'ptxas_version': version,
                    'cubin_sha256': hashlib.sha256((work/'kernel.cubin').read_bytes()).hexdigest()})
            # Publish only after every compiler and structural check succeeded.
            if args.keep_ir:
                for file in ('source.mlir', 'source.ll', 'device.ll', 'optimized.ll'):
                    shutil.copyfile(work/file, args.output.with_suffix('.'+file))
            if ptxas:
                shutil.copyfile(work/'kernel.cubin', args.output.with_suffix('.cubin'))
                shutil.copyfile(work/'ptxas.log', args.output.with_suffix('.ptxas.log'))
            args.output.write_text(ptx)
            args.output.with_suffix('.json').write_text(json.dumps(evidence, indent=2)+'\n')
            print(json.dumps(evidence, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
