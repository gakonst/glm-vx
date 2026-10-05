import hashlib
import pytest
from gpu.testing.fingerprint import INPUTS, BINARY, validate


def fixture(root):
    lines = []
    for name in INPUTS+(BINARY,):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+name)
    manifest = root/'gpu/build/simt-build.sha256'
    manifest.write_text('\n'.join(lines)+'\n')
    return manifest


def test_fresh_complete_manifest(tmp_path):
    fixture(tmp_path); validate(tmp_path)


@pytest.mark.parametrize('name', INPUTS+(BINARY,))
def test_rejects_stale_source_or_binary(tmp_path, name):
    fixture(tmp_path)
    with (tmp_path/name).open('ab') as output: output.write(b'changed')
    with pytest.raises(RuntimeError, match='stale or missing SIMT'): validate(tmp_path)


@pytest.mark.parametrize('change', ['absent', 'partial', 'duplicate'])
def test_rejects_missing_or_incomplete_manifest(tmp_path, change):
    manifest = fixture(tmp_path)
    if change == 'absent': manifest.unlink()
    elif change == 'partial': manifest.write_text(manifest.read_text().splitlines()[0]+'\n')
    else: manifest.write_text(manifest.read_text()+manifest.read_text().splitlines()[0]+'\n')
    with pytest.raises(RuntimeError, match='rebuild'): validate(tmp_path)
