"""Untrained full-architecture CPU fixtures: no trained-model parity claim."""
import threading

import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.model import Model, tiny_weights
from glm_vx.prefix import PrefixCache, _size
from glm_vx.scheduler import Scheduler
from glm_vx.tiny import tiny_config


def model(seed=19, backend="numpy"):
    cfg = tiny_config()
    if backend == 'vx':
        from kernels.backend import VxBackend
        runner = VxBackend()
    else: runner = NumpyBackend()
    return Model(cfg, tiny_weights(cfg, seed=seed), runner)


def collect(req):
    result = []
    while True:
        event = req.events.get(timeout=10)
        if event['type'] == 'done': return result, event
        result.append(event['token_id'])


def snapshot(m, tokens):
    cache = m.new_cache()
    for token in tokens: logits = m.forward(token, cache)
    return cache, logits


def assert_cache(actual, expected):
    assert actual.position == expected.position
    for a, b in zip(actual.layers, expected.layers):
        for name in ('latents', 'rope_keys', 'index_keys', 'selected_indices'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_exact_snapshot_preserves_first_logits_and_all_dsa_state_without_aliasing():
    m = model(); prefix = PrefixCache(m, 1_000_000); tokens = [1, 4, 3, 7, 12, 9]
    cache, logits = snapshot(m, tokens)
    assert prefix.store(tokens, cache, logits)
    a, b = prefix.lookup(tokens), prefix.lookup(tokens)
    np.testing.assert_array_equal(a.logits, logits)
    assert_cache(a.cache, cache)
    expected = m.forward(16, cache)
    np.testing.assert_array_equal(m.forward(16, a.cache), expected)
    assert b.cache.position == len(tokens)
    b.cache.layers[0].latents[0][:] = -100
    b.logits[:] = -100
    fresh = prefix.lookup(tokens)
    np.testing.assert_array_equal(m.forward(16, fresh.cache), expected)
    assert not np.all(fresh.logits == -100)
    assert prefix.lookup([2] + tokens) is None
    assert prefix.lookup(tokens[:-1]) is None


@pytest.mark.parametrize('backend', ['numpy', 'vx'])
@pytest.mark.parametrize('temperature,seed', [(0, 0), (.7, 42), (1.2, 923)])
def test_cached_uncached_same_extension_miss_and_interleaved_requests(temperature, seed, backend):
    cached_model, uncached_model = model(backend=backend), model(backend=backend)
    cached = Scheduler(cached_model, prefix_cache_bytes=1_000_000, prefill_chunk=1)
    uncached = Scheduler(uncached_model, prefill_chunk=1)
    prompt = [1, 4, 3, 7, 12, 9]
    try:
        baseline = collect(uncached.submit(prompt, 6, temperature, seed))
        warm = collect(cached.submit(prompt, 6, temperature, seed))
        assert warm[0] == baseline[0]
        assert warm[1]['cached_prompt_tokens'] == 0
        prompts = [prompt, prompt+[3,8], [8,5,2,1], prompt, prompt+[2]]
        with cached.lock, uncached.lock:
            requests = [cached.submit(p, 6, temperature, seed) for p in prompts]
            references = [uncached.submit(p, 6, temperature, seed) for p in prompts]
        for p, a, b in zip(prompts, requests, references):
            actual, event = collect(a); expected, other = collect(b)
            assert actual == expected
            assert event['reason'] == other['reason'] == 'length'
            assert event['cached_prompt_tokens'] == (len(prompt) if p[:len(prompt)] == prompt else 0)
        assert cached.status()['reserved_tokens'] == 0
    finally: cached.close(); uncached.close()


def test_frozen_input_and_no_generated_token_prefix_leak():
    m = model(); scheduler = Scheduler(m, prefix_cache_bytes=1_000_000)
    prompt = [1, 3, 5]
    try:
        with scheduler.lock:
            req = scheduler.submit(prompt, 4)
            prompt[:] = [7]
        output, event = collect(req)
        assert req.tokens == (1,3,5)
        hit = scheduler.prefix_cache.lookup([1,3,5] + output)
        assert hit.tokens == 3 and hit.cache.position == 3
        assert event['prompt_tokens'] == 3
    finally: scheduler.close()


def test_lru_eviction_exact_resident_budget_and_oversize_rejection():
    m = model(); tokens = [1,2,3]; cache, logits = snapshot(m, tokens)
    probe = PrefixCache(m, 1_000_000); probe.store(tokens, cache, logits)
    size = probe.retained_bytes
    prefix = PrefixCache(m, size)
    assert prefix.store(tokens, cache, logits)
    assert prefix.retained_bytes == _size(prefix.entries) <= size
    assert prefix.store([2,3,4], cache, logits)
    assert prefix.lookup(tokens) is None
    assert prefix.lookup([2,3,4]) is not None
    assert prefix.status()['evictions'] == 1
    too_small = PrefixCache(m, size-1)
    assert not too_small.store(tokens, cache, logits)
    assert too_small.status()['retained_bytes'] == 0
    assert PrefixCache(m).lookup(tokens) is None


def test_model_instance_revision_backend_config_identity_invalidates():
    m = model(); tokens = [1,2,3]; cache, logits = snapshot(m, tokens)
    prefix = PrefixCache(m, 1_000_000)
    for change in (lambda: setattr(m, 'prefix_cache_revision', 1),
                   lambda: setattr(m, 'backend', NumpyBackend()),
                   lambda: m.config.update(rms_norm_eps=2e-5),
                   lambda: setattr(prefix, 'model', model(seed=20))):
        prefix.store(tokens, cache, logits)
        change()
        assert prefix.lookup(tokens) is None
        assert prefix.retained_bytes == 0


def test_exact_hit_eos_uses_saved_logits_without_any_forward():
    m = model(); prompt = [1,4,3]
    _, logits = snapshot(m, prompt)
    eos = int(np.argmax(logits)); m.config['eos_token_id'] = [eos]
    scheduler = Scheduler(m, prefix_cache_bytes=1_000_000)
    try:
        assert collect(scheduler.submit(prompt, 8))[1]['reason'] == 'stop'
        def forbidden(*args): raise AssertionError('exact prompt must not run forward')
        m.forward = forbidden; m.prefill_token = forbidden
        output, event = collect(scheduler.submit(prompt, 8))
        assert output == [eos]
        assert event['reason'] == 'stop'
        assert event['cached_prompt_tokens'] == len(prompt)
    finally: scheduler.close()


def test_cancelled_partial_prompt_is_not_published_and_future_requests_survive():
    m = model(); scheduler = Scheduler(m, prefix_cache_bytes=1_000_000, prefill_chunk=1)
    entered, release = threading.Event(), threading.Event(); original = m.prefill_token
    def blocked(token, cache):
        entered.set(); assert release.wait(5); return original(token, cache)
    m.prefill_token = blocked
    try:
        req = scheduler.submit([1,2,3,4], 3)
        assert entered.wait(2)
        scheduler.cancel(req); release.set()
        assert collect(req)[1]['reason'] == 'cancelled'
        assert scheduler.prefix_cache.status()['entries'] == 0
        m.prefill_token = original
        assert collect(scheduler.submit([1,2,3,4], 3))[1]['reason'] == 'length'
    finally: release.set(); scheduler.close()


def test_longest_completed_prompt_prefix_wins_and_lru_touches_preserve_bound():
    m = model(); prefix = PrefixCache(m, 1000000)
    short = [1,3,5]; longer = short + [7,9]
    for tokens in (short,longer,[2,4,6]):
        cache, logits = snapshot(m,tokens)
        assert prefix.store(tokens,cache,logits)
        assert prefix.retained_bytes == _size(prefix.entries) <= prefix.budget
    hit = prefix.lookup(longer+[11])
    assert hit.tokens == len(longer)
    expected, logits = snapshot(m,longer)
    assert_cache(hit.cache,expected)
    np.testing.assert_array_equal(hit.logits,logits)
    assert prefix.entries[-1][0] == tuple(longer)
    assert prefix.lookup(short+[12]).tokens == len(short)


def test_eviction_recomputes_same_seeded_outputs_without_reusing_wrong_prompt():
    m = model(); prompt = [1,2,3]
    probe=PrefixCache(m,1000000); cache,logits=snapshot(m,prompt)
    probe.store(prompt,cache,logits)
    scheduler=Scheduler(m,prefix_cache_bytes=probe.retained_bytes)
    reference=Scheduler(model())
    try:
        for tokens in (prompt,[4,5,6],prompt,prompt):
            actual,event=collect(scheduler.submit(tokens,4,.8,31))
            expected,_=collect(reference.submit(tokens,4,.8,31))
            assert actual==expected
            assert scheduler.status()['prefix_cache']['retained_bytes']<=probe.retained_bytes
        assert event['cached_prompt_tokens']==3
        assert scheduler.prefix_cache.status()['evictions']==2
    finally: scheduler.close();reference.close()


@pytest.mark.parametrize('attribute,value', [
    ('packed_weights', True), ('eps', .1), ('theta', 123.), ('index_topk', 3),
    ('indexer_types', ['full', 'shared', 'full', 'shared']),
    ('mlp_types', ['dense', 'dense', 'sparse', 'sparse']),
])
def test_resolved_execution_changes_invalidate_snapshots(attribute, value):
    m = model()
    tokens = [1, 4, 3]
    cache, logits = snapshot(m, tokens)
    prefix = PrefixCache(m, 1_000_000)
    assert prefix.store(tokens, cache, logits)
    assert getattr(m, attribute) != value
    setattr(m, attribute, value)
    assert prefix.lookup(tokens) is None
    assert prefix.retained_bytes == 0
