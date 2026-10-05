"""Require CPU SIMT tests to execute the current Vx source and harness build."""
import hashlib
from pathlib import Path

INPUTS = ('gpu/kernels.vx', 'gpu/fp8.vx', 'gpu/gemm.vx',
          'gpu/testing/simt.cpp', 'gpu/testing/build_simt.sh')
BINARY = 'gpu/build/libsimt.so'


def validate(root):
    root = Path(root)
    manifest = root/'gpu/build/simt-build.sha256'
    instruction = 'rebuild gpu/testing/build_simt.sh before GPU arithmetic tests'
    try:
        rows = [line.split(maxsplit=1) for line in manifest.read_text().splitlines()]
        expected = {path: digest for digest, path in rows}
        if len(rows) != len(INPUTS)+1 or set(expected) != set(INPUTS+(BINARY,)):
            raise ValueError('incomplete or duplicate build manifest')
        for path, digest in expected.items():
            actual = hashlib.sha256((root/path).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError('changed source or binary: '+path)
    except (OSError, ValueError) as exc:
        raise RuntimeError('stale or missing SIMT build: '+instruction) from exc
