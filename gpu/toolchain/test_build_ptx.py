"""Compiler-contract regressions; integration test compiles actual Vx to PTX."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('build_ptx', HERE/'build_ptx.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

VALID = '''declare i32 @vx_gpu_thread_idx_x()
define void @glm_vx_gpu_test(ptr %0, i32 %1) {
  %3 = call i32 @vx_gpu_thread_idx_x()
  ret void
}
'''

class ContractTests(unittest.TestCase):
    def test_maps_kernel_and_index(self):
        ir, entries = builder.adapter_ir(VALID, 256)
        self.assertEqual(entries, ['glm_vx_gpu_test'])
        self.assertIn('define ptx_kernel void', ir)
        self.assertIn('llvm.nvvm.read.ptx.sreg.tid.x', ir)

    def test_rejects_host_external(self):
        with self.assertRaisesRegex(ValueError, 'unsupported external'):
            builder.adapter_ir(VALID+'declare i32 @printf(ptr, ...)\n', 256)

    def test_rejects_index_abi_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'ABI mismatch'):
            builder.adapter_ir(VALID.replace('declare i32', 'declare i64'), 256)

    def test_rejects_nonvoid_kernel(self):
        with self.assertRaisesRegex(ValueError, 'return void'):
            builder.adapter_ir(VALID.replace('define void', 'define i32'), 256)

    def test_rejects_tensor_descriptor(self):
        with self.assertRaisesRegex(ValueError, 'unsupported entry parameter'):
            builder.adapter_ir(VALID.replace('ptr %0', '{ptr, i64} %0'), 256)

    def test_rejects_host_triple(self):
        with self.assertRaisesRegex(ValueError, 'host target'):
            builder.adapter_ir('target triple = "x86_64-linux-gnu"\n'+VALID, 256)

    def test_rejects_missing_kernel(self):
        with self.assertRaisesRegex(ValueError, 'no glm_vx_gpu_'):
            builder.adapter_ir(VALID.replace('glm_vx_gpu_test', 'main'), 256)

    def test_convergent_ops(self):
        ir, _ = builder.adapter_ir(VALID + 'declare void @vx_gpu_barrier()\n'
            + 'declare float @vx_gpu_shuffle_down_f32(float, i32)\n', 256)
        self.assertIn('alwaysinline convergent', ir)
        self.assertIn('i32 -1, float %value, i32 %lane, i32 31', ir)

    def test_rejects_ptx_calls_and_missing_threads(self):
        entry = '.visible .entry glm_vx_gpu_test() { %tid.x; }'
        with self.assertRaisesRegex(ValueError, 'device call'):
            builder.validate_ptx(entry+'\ncall.uni foo;', ['glm_vx_gpu_test'])
        with self.assertRaisesRegex(ValueError, 'no hardware'):
            builder.validate_ptx(entry.replace('%tid.x', ''), ['glm_vx_gpu_test'])

    def test_real_vx_device_math(self):
        # Missing tools is a test failure, not a skipped/passing compilation.
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'probe.ptx'
            subprocess.run([sys.executable, str(HERE/'build_ptx.py'), '--source',
                str(HERE/'probe.vx'), '--output', str(output)], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            ptx = output.read_text()
            for token in ('.visible .entry glm_vx_gpu_probe', '%tid.x', '%ctaid.x',
                          '%ntid.x', 'ex2.approx.ftz.f32', 'sqrt.rn.f32'):
                self.assertIn(token, ptx)

    def test_real_numerical_kernels(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'kernels.ptx'
            subprocess.run([sys.executable, str(HERE/'build_ptx.py'), '--source',
                str(HERE.parent/'kernels.vx'), '--output', str(output)], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            ptx = output.read_text()
            for name in ('rmsnorm', 'softmax', 'router', 'residual_rmsnorm',
                         'swiglu', 'rope_glm', 'matvec', 'index_scores', 'mla_partial', 'mla_merge',
                         'rmsnorm_rows', 'softmax_rows'):
                self.assertIn('.visible .entry glm_vx_gpu_' + name + '(', ptx)
            for token in ('shfl.sync.idx', 'shfl.sync.down', 'bar.sync', '.shared'):
                self.assertIn(token, ptx)

    def test_mma_adapter_contract(self):
        declaration = 'declare void @vx_gpu_mma_tf32_m16n8k8(ptr, i32, ' + ', '.join(['float'] * 10) + ')\n'
        with self.assertRaisesRegex(ValueError, '320 shared'):
            builder.adapter_ir(VALID + declaration, 256)
        with self.assertRaisesRegex(ValueError, 'ABI mismatch'):
            builder.adapter_ir(VALID + declaration.replace('ptr, i32', 'ptr, i64'), 320)
        ir, _ = builder.adapter_ir(VALID + declaration, 320)
        self.assertIn('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32', ir)
        self.assertEqual(ir.count('cvt.rna.tf32.f32'), 6)
        self.assertIn('sideeffect', ir)
        self.assertIn(') convergent', ir)

    def test_tensor_core_entry_cannot_be_scalar_relabel(self):
        fake = '.visible .entry glm_vx_gpu_gemm_tf32() { %tid.x; }'
        with self.assertRaisesRegex(ValueError, 'missing required PTX'):
            builder.validate_ptx(fake, ['glm_vx_gpu_gemm_tf32'])

    def test_real_vx_tensor_core_gemm(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'gemm.ptx'
            command = [sys.executable, str(HERE/'build_ptx.py'), '--source',
                       str(HERE.parent/'gemm.vx'), '--output', str(output), '--shared-floats', '320']
            subprocess.run(command, check=True, capture_output=True, text=True)
            ptx = output.read_text()
            for token in ('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32',
                          'cvt.rna.tf32.f32', 'ld.shared.', 'st.shared.', 'bar.sync'):
                self.assertIn(token, ptx)
            previous = ptx
            failed = subprocess.run(command + ['--arch', 'sm_75'], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('requires sm_80', failed.stderr)
            self.assertEqual(output.read_text(), previous)

    def test_failed_assembly_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory/'probe.ptx'
            output.write_text('previous build')
            assembler = directory/'failing-ptxas'
            assembler.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo test-assembler; exit 0; fi\nexit 42\n')
            assembler.chmod(0o700)
            result = subprocess.run([sys.executable, str(HERE/'build_ptx.py'), '--source',
                str(HERE/'probe.vx'), '--output', str(output),
                '--ptxas', str(assembler)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('NVIDIA ptxas failed (42)', result.stderr)
            self.assertEqual(output.read_text(), 'previous build')
            self.assertFalse(output.with_suffix('.json').exists())

if __name__ == '__main__':
    unittest.main()
