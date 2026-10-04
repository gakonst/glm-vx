"""Checkpoint omissions must use the pinned official DSA config defaults.

Transformers 469230357aab0f2b303b0d638c1f8d06edb14184,
configuration_glm_moe_dsa.py:136-148 uses frequency=1, offset=2.
The shipped GLM-5.3 config explicitly specifies frequency=4, offset=3.
"""
from pathlib import Path

import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.config import GLMConfig
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend


def omitted_schedule():
    raw = tiny_config()
    for name in ('indexer_types', 'index_topk_freq', 'index_skip_topk_offset'):
        raw.pop(name)
    return raw


@pytest.mark.parametrize('patch,expected', [
    ({}, ('full',) * 4),
    ({'index_skip_topk_offset': 3}, ('full',) * 4),
    ({'index_topk_freq': 4}, ('full', 'full', 'shared', 'shared')),
    ({'index_topk_freq': 4, 'index_skip_topk_offset': 3},
     ('full', 'full', 'full', 'shared')),
    ({'indexer_types': None}, ('full',) * 4),
    ({'index_topk_pattern': 'FSSF'}, ('full', 'shared', 'shared', 'full')),
    ({'indexer_types': ['full', 'shared', 'full', 'shared'],
      'index_topk_pattern': 'FFFF'}, ('full', 'shared', 'full', 'shared')),
])
def test_checkpoint_dsa_schedule_defaults_and_precedence(patch, expected):
    raw = dict(omitted_schedule(), **patch)
    config = GLMConfig.from_dict(raw)
    assert config.indexer_types == expected
    # The direct runtime already uses the official defaults. Both entry points
    # must resolve the same topology, including partial scheduling metadata.
    assert tuple(GlmMoeDsaModel(raw, {}, NumpyBackend()).indexer_types) == expected


@pytest.mark.parametrize('backend', [NumpyBackend, VxBackend])
@pytest.mark.parametrize('batched', [False, True])
def test_omitted_schedule_executes_every_indexer(batched, backend):
    raw = omitted_schedule()
    # Exceed top-k so accidental sharing changes the actual sparse attention.
    raw['index_topk'] = 2
    weights = tiny_weights(raw, seed=31)
    expected = GlmMoeDsaModel(dict(raw, indexer_types=['full'] * 4), weights, backend())
    actual = GlmMoeDsaModel(GLMConfig.from_dict(raw).to_dict(), weights, backend())
    actual_cache, expected_cache = actual.new_cache(), expected.new_cache()
    tokens = [1, 7, 2, 11, 4, 9]
    if batched:
        wanted = expected.prefill_chunk(tokens, expected_cache)
        got = actual.prefill_chunk(tokens, actual_cache)
    else:
        wanted = expected.prefill(tokens, expected_cache)
        got = actual.prefill(tokens, actual_cache)
    # No numerical tolerance: these must execute the exact same operations.
    np.testing.assert_array_equal(got, wanted)
    for cache in (actual_cache, expected_cache):
        assert all(len(layer.index_keys) == len(tokens) for layer in cache.layers)
    np.testing.assert_array_equal(actual.forward(13, actual_cache), expected.forward(13, expected_cache))


def test_shipped_checkpoint_keeps_its_explicit_shared_schedule():
    config = GLMConfig.from_file(Path(__file__).resolve().parents[1] / 'metadata/config.json')
    assert config.index_topk_freq == 4
    assert config.index_skip_topk_offset == 3
    assert config.indexer_types == tuple(
        'full' if max(i - 2, 0) % 4 == 0 else 'shared'
        for i in range(config.num_hidden_layers))
