"""CPU handoff correctness/security regressions; no checkpoint downloads."""
import hashlib
import json
from pathlib import Path
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.handoff import (MAGIC, HandoffError, HandoffReceiver, Limits,
                            OwnedRequest, model_identity)
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.tiny import tiny_config


def make_model(backend='numpy', seed=19, update=None):
    config = tiny_config()
    config.update(indexer_types=['full', 'shared', 'full', 'shared'])
    config.update(update or {})
    if backend == 'vx':
        library = Path(__file__).resolve().parents[1] / 'kernels/build/libglm_vx.so'
        if not library.exists():
            pytest.skip('compiled CPU Vx library is not available')
        from kernels.backend import VxBackend
        engine = VxBackend(library)
    else:
        engine = NumpyBackend()
    return GlmMoeDsaModel(config, tiny_weights(config, seed), engine)


def ready(model=None, *, tokens=(1, 7, 2, 11, 4), request_id='r', limits=Limits()):
    request = OwnedRequest(model or make_model(), request_id, 'prefill', limits=limits)
    request.append_prompt(list(tokens))
    request.seal_prompt()
    return request


def rewrite(payload, update=None, mutate_body=None, canonical=True):
    start = len(MAGIC) + 4
    length = struct.unpack_from('>I', payload, len(MAGIC))[0]
    header = json.loads(payload[start:start + length])
    if update:
        header.update(update)
    body = bytearray(payload[start + length:-32])
    if mutate_body:
        mutate_body(body, header)
    text = json.dumps(header, sort_keys=True, separators=(',', ':') if canonical else None).encode('ascii')
    wire = MAGIC + struct.pack('>I', len(text)) + text + body
    return wire + hashlib.sha256(wire).digest()


def assert_cache_equal(actual, expected):
    assert actual.position == expected.position
    for a, b in zip(actual.layers, expected.layers):
        for field in ('latents', 'rope_keys', 'index_keys', 'selected_indices'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize('backend', ['numpy', 'vx'])
@pytest.mark.parametrize('partitions', [[7], [1, 1, 1, 1, 1, 1, 1], [1, 2, 4], [3, 4]])
def test_partitioned_prefill_transfer_decode_exactly_matches_unsplit(backend, partitions):
    tokens = [1, 7, 2, 11, 4, 3, 16]
    oracle = make_model(backend)
    expected_cache = oracle.new_cache()
    expected_logits = oracle.prefill(tokens, expected_cache)[-1]
    source = OwnedRequest(make_model(backend), 'r', 'prefill')
    offset = 0
    for count in partitions:
        source.append_prompt(tokens[offset:offset + count])
        offset += count
    source.seal_prompt()
    payload = source.transfer('decode', transfer_id='transfer-1')
    assert not source.active
    for action in (source.seal_prompt, lambda: source.append_prompt([1]),
                   source.greedy, lambda: source.transfer('decode')):
        with pytest.raises(HandoffError, match='ownership'):
            action()
    model = make_model(backend)
    calls = []
    original = model.forward
    model.forward = lambda token, cache: (calls.append((token, cache.position)), original(token, cache))[1]
    decoded = HandoffReceiver(model, 'decode').claim(payload, request_id='r', source='prefill')
    assert calls == []  # Claim performs no model forward/recompute.
    np.testing.assert_array_equal(decoded.logits, expected_logits)
    assert_cache_equal(decoded.cache, expected_cache)
    expected_tokens = []
    for _ in range(5):
        token = int(np.argmax(expected_logits))
        expected_tokens.append(token)
        expected_logits = oracle.forward(token, expected_cache)
    assert decoded.greedy(5) == expected_tokens
    assert calls == [(token, len(tokens) + i) for i, token in enumerate(expected_tokens)]
    assert decoded.tokens == tokens + expected_tokens
    assert decoded.prompt_length == len(tokens)
    assert_cache_equal(decoded.cache, expected_cache)
    np.testing.assert_array_equal(decoded.logits, expected_logits)


@pytest.mark.parametrize('split', range(1, 7))
def test_transfer_during_prefill_and_again_during_decode(split):
    tokens = [1, 7, 2, 11, 4, 3, 16]
    expected = ready(tokens=tokens)
    expected_tokens = expected.greedy(5)
    source = OwnedRequest(make_model(), 'r', 'prefill')
    source.append_prompt(tokens[:split])
    next_worker = HandoffReceiver(make_model(), 'prefill2').claim(
        source.transfer('prefill2'), request_id='r', source='prefill')
    assert next_worker.phase == 'prefill'
    with pytest.raises(HandoffError, match='seal'):
        next_worker.greedy()
    next_worker.append_prompt(tokens[split:])
    next_worker.seal_prompt()
    first = next_worker.greedy(2)
    decoded = HandoffReceiver(make_model(), 'decode').claim(
        next_worker.transfer('decode'), request_id='r', source='prefill2')
    assert first + decoded.greedy(3) == expected_tokens
    assert decoded.tokens == expected.tokens
    assert decoded.prompt_length == len(tokens)
    assert_cache_equal(decoded.cache, expected.cache)


def test_payload_is_canonical_and_reconstruction_owns_memory():
    a, b = ready(), ready()
    payload = a.transfer('decode', transfer_id='fixed')
    assert payload == b.transfer('decode', transfer_id='fixed')
    imported = HandoffReceiver(make_model(), 'decode').claim(payload, request_id='r', source='prefill')
    before = imported.cache.layers[0].latents[0].copy()
    a.cache.layers[0].latents[0].fill(99)  # Deliberate bypass for alias detection.
    np.testing.assert_array_equal(imported.cache.layers[0].latents[0], before)
    assert imported.cache.layers[0].latents[0].flags.writeable
    assert not np.shares_memory(imported.cache.layers[0].latents[0], imported.cache.layers[0].latents[1])


@pytest.mark.parametrize('field,value', [
    ('version', 2), ('version', True), ('position', True), ('position', -1),
    ('position', 10**30), ('position', 6), ('phase', 'unknown'),
    ('prompt_length', 0), ('prompt_length', 100), ('prompt_length', True),
    ('request_id', '../bad'), ('source', 'decode'), ('transfer_id', ''),
    ('extra', 1), ('identity', {}),
])
def test_rejects_invalid_header_even_with_valid_digest(field, value):
    payload = rewrite(ready().transfer('decode'), {field: value})
    with pytest.raises(HandoffError):
        HandoffReceiver(make_model(), 'decode').claim(payload, request_id='r', source='prefill')


@pytest.mark.parametrize('mutation', [
    lambda p: b'garbage' + p[7:],
    lambda p: p[:-1],
    lambda p: p + b'\0',
    lambda p: p[:100] + bytes([p[100] ^ 1]) + p[101:],
    lambda p: p[:len(MAGIC)] + struct.pack('>I', 2**32 - 1) + p[len(MAGIC) + 4:],
    lambda p: rewrite(p, canonical=False),
    lambda p: bytearray(p),
])
def test_rejects_corruption_trailing_data_noncanonical_and_mutable_buffers(mutation):
    payload = ready().transfer('decode')
    with pytest.raises(HandoffError):
        HandoffReceiver(make_model(), 'decode').claim(mutation(payload), request_id='r', source='prefill')


@pytest.mark.parametrize('which', ['token', 'logits', 'latent', 'selection', 'duplicate', 'shared'])
def test_rejects_invalid_array_values_with_valid_digest(which):
    model = make_model()
    def mutate(body, header):
        pos, vocab = header['position'], model.config['vocab_size']
        # Offsets follow fixed wire layout (all scalar widths are four bytes).
        selection = pos + vocab + pos * (model.rank + model.rot + model.index_dim)
        if which == 'token':
            struct.pack_into('<I', body, 0, vocab)
        elif which == 'logits':
            struct.pack_into('<f', body, pos * 4, float('nan'))
        elif which == 'latent':
            struct.pack_into('<f', body, (pos + vocab) * 4, float('inf'))
        elif which == 'selection':
            struct.pack_into('<I', body, selection * 4, pos)
        elif which == 'duplicate':
            body[(selection + 1) * 4:(selection + 2) * 4] = body[selection * 4:(selection + 1) * 4]
        else:
            shared = selection + min(pos, model.index_topk) + pos * (model.rank + model.rot)
            # Different order remains individually valid but violates sharing.
            first, second = body[shared * 4:(shared + 1) * 4], body[(shared + 1) * 4:(shared + 2) * 4]
            body[shared * 4:(shared + 1) * 4], body[(shared + 1) * 4:(shared + 2) * 4] = second, first
    payload = rewrite(ready(model).transfer('decode'), mutate_body=mutate)
    with pytest.raises(HandoffError):
        HandoffReceiver(model, 'decode').claim(payload, request_id='r', source='prefill')


@pytest.mark.parametrize('change', ['weight', 'config', 'architecture', 'backend'])
def test_identity_rejects_model_config_weight_or_backend_mismatch(change):
    source = ready()
    target = make_model('vx' if change == 'backend' else 'numpy')
    if change == 'weight':
        target.weights['lm_head.weight'][0, 0] += np.float32(0.01)
    elif change == 'config':
        target.config['rope_parameters'] = {'rope_theta': 10000, 'rope_type': 'default'}
    elif change == 'architecture':
        target.indexer_types = ['full'] * target.n_layers
    with pytest.raises(HandoffError, match='identity mismatch'):
        HandoffReceiver(target, 'decode').claim(source.transfer('decode'), request_id='r', source='prefill')


def test_identity_normalizes_mapping_order_and_weight_endianness():
    a, b = make_model(), make_model()
    b.weights = {key: value.astype('>f4') for key, value in reversed(list(b.weights.items()))}
    assert model_identity(a) == model_identity(b)
    b.weights = b.weights.__getitem__
    with pytest.raises(HandoffError, match='mapping'):
        model_identity(b)


def test_mutation_after_start_fails_closed_and_does_not_relinquish():
    source = ready()
    source.model.weights['lm_head.weight'][0, 0] += 1
    with pytest.raises(HandoffError, match='changed'):
        source.transfer('decode')
    assert source.active
    receiver = HandoffReceiver(make_model(), 'decode')
    receiver.model.config['something_new'] = 3
    with pytest.raises(HandoffError, match='changed'):
        receiver.claim(ready().transfer('decode'), request_id='r', source='prefill')


def test_wrong_ownership_replay_and_bounded_claim_ledger():
    receiver = HandoffReceiver(make_model(), 'decode', limits=Limits(max_claims=1))
    payload = ready().transfer('decode', transfer_id='one')
    for request, source in [('other', 'prefill'), ('r', 'wrong')]:
        with pytest.raises(HandoffError, match='ownership'):
            receiver.claim(payload, request_id=request, source=source)
    with pytest.raises(HandoffError, match='ownership'):
        HandoffReceiver(make_model(), 'wrong').claim(payload, request_id='r', source='prefill')
    actual = receiver.claim(payload, request_id='r', source='prefill')
    for p in [payload, rewrite(payload, {'transfer_id': 'different'})]:
        with pytest.raises(HandoffError, match='duplicate'):
            receiver.claim(p, request_id='r', source='prefill')
    with pytest.raises(HandoffError, match='replayed'):
        receiver.claim(rewrite(payload, {'request_id': 'new'}), request_id='new', source='prefill')
    with pytest.raises(HandoffError, match='full'):
        receiver.claim(ready(request_id='new').transfer('decode'), request_id='new', source='prefill')
    assert actual.tokens == [1, 7, 2, 11, 4]


def test_concurrent_duplicate_claims_have_one_winner():
    receiver = HandoffReceiver(make_model(), 'decode')
    payload = ready().transfer('decode')
    def claim(_):
        try:
            return receiver.claim(payload, request_id='r', source='prefill')
        except HandoffError:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(claim, range(4)))
    assert sum(result is not None for result in results) == 1


def test_limits_and_bad_source_state_do_not_relinquish():
    with pytest.raises(HandoffError):
        Limits(max_tokens=True)
    with pytest.raises(HandoffError, match='weight'):
        model_identity(make_model(), Limits(max_weight_bytes=32))
    with pytest.raises(HandoffError, match='layer'):
        model_identity(make_model(), Limits(max_layers=1))
    request = ready(limits=Limits(max_tokens=5))
    with pytest.raises(HandoffError, match='position'):
        request.greedy()
    assert request.tokens == [1, 7, 2, 11, 4]
    payload = request.transfer('decode')
    for limits in [Limits(max_tokens=4), Limits(max_payload_bytes=len(payload) - 1), Limits(max_header_bytes=32)]:
        with pytest.raises(HandoffError):
            HandoffReceiver(make_model(), 'decode', limits=limits).claim(payload, request_id='r', source='prefill')
    bad = ready()
    bad.cache.layers[1].index_keys.append(np.zeros(bad.model.index_dim, np.float32))
    with pytest.raises(HandoffError, match='shape'):
        bad.transfer('decode')
    assert bad.active
    bad = ready()
    bad.logits = bad.logits.astype(np.float64)
    with pytest.raises(HandoffError, match='float32'):
        bad.transfer('decode')
    assert bad.active


def test_empty_invalid_tokens_sealed_prompt_and_chunk_rollback():
    model = make_model()
    source = OwnedRequest(model, 'r', 'prefill')
    for action in (source.seal_prompt, lambda: source.transfer('decode'),
                   lambda: source.append_prompt([]), lambda: source.append_prompt([True]),
                   lambda: source.append_prompt([1, -1])):
        with pytest.raises(HandoffError):
            action()
    assert source.active and source.cache.position == 0
    source.append_prompt([1, 2])
    expected = model.new_cache()
    model.prefill([1, 2], expected)
    before_logits = source.logits.copy()
    forward = model.forward
    def fail(*args):
        raise RuntimeError('injected last token failure')
    model.forward = fail
    with pytest.raises(RuntimeError):
        source.append_prompt([3, 4])
    assert source.tokens == [1, 2]
    assert_cache_equal(source.cache, expected)
    np.testing.assert_array_equal(source.logits, before_logits)
    model.forward = forward
    source.append_prompt([3, 4])
    source.seal_prompt()
    with pytest.raises(HandoffError, match='sealed'):
        source.append_prompt([5])
    assert source.greedy(0) == []


def test_decode_batch_failure_rolls_back_samples_and_can_retry_without_loss():
    source = ready()
    expected = ready()
    before_logits = source.logits.copy()
    original = source.model.forward
    calls = 0
    def fail_second(token, cache):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Fail within the model, after some layers have appended state.
            name = 'model.layers.2.self_attn.q_a_proj.weight'
            weight = source.model.weights.pop(name)
            try:
                return original(token, cache)
            finally:
                source.model.weights[name] = weight
        return original(token, cache)
    source.model.forward = fail_second
    with pytest.raises(KeyError):
        source.greedy(3)
    assert source.tokens == expected.tokens
    assert_cache_equal(source.cache, expected.cache)
    np.testing.assert_array_equal(source.logits, before_logits)
    source.model.forward = original
    assert source.greedy(3) == expected.greedy(3)
    assert_cache_equal(source.cache, expected.cache)


def test_multiple_requests_do_not_share_state_and_concurrent_transfer_has_one_winner():
    model = make_model()
    receiver = HandoffReceiver(model, 'decode')
    a = ready(model, request_id='a')
    b = ready(model, tokens=[2, 9, 8], request_id='b')
    decoded_a = receiver.claim(a.transfer('decode'), request_id='a', source='prefill')
    decoded_b = receiver.claim(b.transfer('decode'), request_id='b', source='prefill')
    before = decoded_b.logits.copy()
    decoded_a.greedy(2)
    assert decoded_b.tokens == [2, 9, 8] and decoded_b.cache.position == 3
    np.testing.assert_array_equal(decoded_b.logits, before)
    def transfer(_):
        try:
            return decoded_a.transfer('next')
        except HandoffError:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(transfer, range(4)))
    assert sum(result is not None for result in results) == 1


@pytest.mark.parametrize('attribute,value', [('eps', .1), ('theta', 123.0), ('index_topk', 3)])
def test_resolved_model_state_is_part_of_identity(attribute, value):
    model = make_model()
    setattr(model, attribute, value)
    payload = ready().transfer('decode')
    with pytest.raises(HandoffError, match='identity mismatch'):
        HandoffReceiver(model, 'decode').claim(payload, request_id='r', source='prefill')


def test_gguf_identity_hashes_actual_bytes_once_without_tensor_decode(tmp_path, monkeypatch):
    pytest.importorskip('gguf')
    from test_gguf_checkpoint import fixture_data, write_fixture, opened
    from glm_vx.handoff import GGUFIdentity
    config, weights = fixture_data()
    path = write_fixture(tmp_path / 'model.gguf', config, weights)
    with opened(path, config, cache_bytes=0) as checkpoint:
        model = GlmMoeDsaModel(config, checkpoint, NumpyBackend())
        def no_decode(*args, **kwargs):
            pytest.fail('identity must never dequantize/read a model tensor')
        monkeypatch.setattr(checkpoint, 'tensor', no_decode)
        captured = []
        capture = GGUFIdentity.capture.__func__
        def recording(cls, source):
            captured.append(source)
            return capture(cls, source)
        monkeypatch.setattr(GGUFIdentity, 'capture', classmethod(recording))
        first = model_identity(model, Limits(max_weight_bytes=1))
        assert model_identity(model) == first
        assert len(captured) == 1
        assert checkpoint.cache_bytes == 0
        # Decode size settings do not enter the identity: identical math/storage.
        checkpoint.store.decode_rows = 1
        assert model_identity(model) == first
        assert len(captured) == 1


def test_gguf_handoff_matches_unsplit_and_ignores_local_path(tmp_path):
    pytest.importorskip('gguf')
    from test_gguf_checkpoint import fixture_data, write_fixture, opened
    import shutil
    config, weights = fixture_data()
    first = write_fixture(tmp_path / 'first.gguf', config, weights)
    second = tmp_path / 'copy.gguf'
    shutil.copyfile(first, second)
    with opened(first, config, cache_bytes=0) as a, opened(second, config, cache_bytes=0) as b:
        source_model = GlmMoeDsaModel(config, a, NumpyBackend())
        target_model = GlmMoeDsaModel(config, b, NumpyBackend())
        source = ready(source_model)
        assert model_identity(source_model) == model_identity(target_model)
        decoded = HandoffReceiver(target_model, 'decode').claim(
            source.transfer('decode'), request_id='r', source='prefill')
        oracle = ready(GlmMoeDsaModel(config, weights, NumpyBackend()))
        assert decoded.greedy(4) == oracle.greedy(4)
        assert_cache_equal(decoded.cache, oracle.cache)
        np.testing.assert_array_equal(decoded.logits, oracle.logits)


@pytest.mark.parametrize('mutation', ['mtime', 'replace', 'bytes', 'close'])
def test_gguf_file_changes_fail_closed_without_rehash(tmp_path, mutation):
    pytest.importorskip('gguf')
    from test_gguf_checkpoint import fixture_data, write_fixture, opened
    import os
    import shutil
    config, weights = fixture_data()
    path = write_fixture(tmp_path / 'model.gguf', config, weights)
    with opened(path, config, cache_bytes=0) as checkpoint:
        model = GlmMoeDsaModel(config, checkpoint, NumpyBackend())
        source = ready(model)
        if mutation == 'mtime':
            stamp = path.stat()
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
        elif mutation == 'replace':
            copy = tmp_path / 'replacement.gguf'
            shutil.copyfile(path, copy)
            copy.replace(path)
        elif mutation == 'bytes':
            with path.open('r+b') as file:
                file.seek(-1, 2)
                byte = file.read(1)
                file.seek(-1, 2)
                file.write(bytes([byte[0] ^ 1]))
        else:
            checkpoint.close()
        with pytest.raises(HandoffError, match='changed'):
            source.transfer('decode')
        assert source.active


def test_gguf_different_weight_bytes_rejected(tmp_path):
    pytest.importorskip('gguf')
    from test_gguf_checkpoint import fixture_data, write_fixture, opened
    config, weights = fixture_data()
    first = write_fixture(tmp_path / 'first.gguf', config, weights)
    weights['lm_head.weight'][0, 0] += 1
    second = write_fixture(tmp_path / 'other.gguf', config, weights)
    with opened(first, config) as a, opened(second, config) as b:
        source = ready(GlmMoeDsaModel(config, a, NumpyBackend()))
        receiver = HandoffReceiver(GlmMoeDsaModel(config, b, NumpyBackend()), 'decode')
        with pytest.raises(HandoffError, match='identity mismatch'):
            receiver.claim(source.transfer('decode'), request_id='r', source='prefill')


def test_split_gguf_identity_binds_every_shard_and_detects_later_shard_edit(tmp_path):
    gguf = pytest.importorskip('gguf')
    from test_gguf_checkpoint import fixture_data, packed_weights, fixture_metadata, opened
    config, weights = fixture_data()
    tensors = list(packed_weights(config, weights).items())
    paths = []
    for shard in range(2):
        path = tmp_path / f'model-{shard + 1:05d}-of-00002.gguf'
        writer = gguf.GGUFWriter(path, 'glm-dsa')
        writer.add_uint16('split.no', shard)
        writer.add_uint16('split.count', 2)
        writer.add_uint64('split.tensors.count', len(tensors))
        for name, value in fixture_metadata(config).items():
            add = writer.add_bool if isinstance(value, bool) else writer.add_float32 if isinstance(value, float) else writer.add_uint32
            add('glm-dsa.' + name, value)
        for name, value in tensors[shard::2]:
            writer.add_tensor(name, np.ascontiguousarray(value))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(path)
    with opened(paths[0], config) as checkpoint:
        model = GlmMoeDsaModel(config, checkpoint, NumpyBackend())
        identity = model_identity(model)
        assert len(model._handoff_gguf_identity.paths) == 2
        request = ready(model)
        with paths[1].open('r+b') as file:
            file.seek(-1, 2)
            byte = file.read(1)
            file.seek(-1, 2)
            file.write(bytes([byte[0] ^ 1]))
        with pytest.raises(HandoffError, match='changed'):
            request.transfer('decode')
    with opened(paths[1], config) as checkpoint:
        assert model_identity(GlmMoeDsaModel(config, checkpoint, NumpyBackend())) != identity
