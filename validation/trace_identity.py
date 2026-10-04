"""Declared checkpoint identity and bounded file fingerprints for trace tools.

Shard hashes are reused from verified download metadata; this is not a fresh
hash of the hundreds of GB of weights or an attestation of a trace producer.
"""
import hashlib
import json
from pathlib import Path
from validation.trace_gate import parse_json


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bind_model(directory, contract_path):
    directory, contract_path = Path(directory), Path(contract_path)
    contract = parse_json(contract_path.read_text())
    pins = contract['pins']
    if sha_file(directory/'config.json') != pins['config_sha256']:
        raise ValueError('checkpoint config differs from frozen contract')
    manifest_path = directory/'manifest.json'
    manifest = parse_json(manifest_path.read_text())
    declared = {item['rfilename']: item['lfs']['sha256'] for item in manifest['files']}
    if len(declared) != len(manifest['files']):
        raise ValueError('duplicate checkpoint manifest entry')
    expected = pins['weights_sha256']
    if declared != expected:
        raise ValueError('checkpoint manifest differs from frozen weight pins')
    files, basenames = {}, set()
    for name, digest in expected.items():
        base = Path(name).name
        if base in basenames:
            raise ValueError('ambiguous checkpoint basename')
        basenames.add(base)
        stat = (directory/base).stat()
        if stat.st_size <= 0:
            raise ValueError('empty checkpoint shard')
        files[name] = {'declared_sha256': digest, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return {'config_sha256': pins['config_sha256'], 'contract_sha256': sha_file(contract_path),
            'manifest_sha256': sha_file(manifest_path), 'checkpoint_files': files,
            'verification': 'declared shard pins reused; weight bytes NOT rehashed'}


def bind_loaded(identity, directory, paths):
    """Check the shard set selected by the loader, not only adjacent metadata."""
    expected = {name: (Path(directory)/Path(name).name).resolve()
                for name in identity['checkpoint_files']}
    loaded = [Path(path).resolve() for path in paths]
    if len(loaded) != len(set(loaded)) or set(loaded) != set(expected.values()):
        raise ValueError('loaded checkpoint shards differ from bound identity')
    result = {}
    for name,path in expected.items():
        stat = path.stat(); original = identity['checkpoint_files'][name]
        if (stat.st_size, stat.st_mtime_ns) != (original['bytes'], original['mtime_ns']):
            raise ValueError('loaded checkpoint file metadata changed')
        result[name] = str(path)
    return result
