"""A real prefill interpreter transfers bytes to a separate decode interpreter."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.disaggregated import run_demo
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.tiny import tiny_config


@pytest.mark.parametrize('backend', ['numpy', 'vx'])
@pytest.mark.parametrize('partitions', [[5], [1, 1, 1, 1, 1], [2, 3]])
def test_process_separated_prefill_decode_matches_unsplit(backend, partitions):
    if backend == 'vx':
        library = Path(__file__).resolve().parents[1] / 'kernels/build/libglm_vx.so'
        if not library.exists():
            pytest.skip('compiled CPU Vx library is not available')
        from kernels.backend import VxBackend
        engine = VxBackend(library)
    else:
        engine = NumpyBackend()
    tokens = [1, 7, 2, 11, 4]
    result = run_demo(tokens, partitions=partitions, new_tokens=5, backend=backend, seed=17)
    config = tiny_config()
    model = GlmMoeDsaModel(config, tiny_weights(config, 17), engine)
    cache = model.new_cache()
    logits = model.prefill(tokens, cache)[-1]
    expected = []
    for _ in range(5):
        token = int(np.argmax(logits))
        expected.append(token)
        logits = model.forward(token, cache)
    assert result['generated_tokens'] == expected
    assert result['consumed_tokens'] == tokens + expected
    assert result['received_position'] == result['prompt_length'] == len(tokens)
    assert result['final_position'] == len(tokens) + len(expected)
    np.testing.assert_array_equal(result['next_logits'], logits)
    assert len({result['prefill_pid'], result['decode_pid'], os.getpid()}) == 3
    assert result['handoff_bytes'] > 0


def test_worker_rejects_partitions_without_retry():
    with pytest.raises(subprocess.CalledProcessError):
        run_demo([1, 2], partitions=[1], new_tokens=1)


@pytest.mark.parametrize('backend', ['numpy', 'vx'])
def test_real_gguf_reader_process_handoff_on_synthetic_checkpoint(tmp_path, backend):
    pytest.importorskip('gguf')
    from test_gguf_checkpoint import fixture_data, write_fixture
    import json
    library = None
    if backend == 'vx':
        library = Path(__file__).resolve().parents[1] / 'kernels/build/libglm_vx.so'
        if not library.exists():
            pytest.skip('compiled CPU Vx library is not available')
        from kernels.backend import VxBackend
        engine = VxBackend(library)
    else:
        engine = NumpyBackend()
    config, weights = fixture_data()
    checkpoint = write_fixture(tmp_path / 'model.gguf', config, weights)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    tokens = [1, 7, 2, 11, 4]
    result = run_demo(tokens, partitions=[2, 1, 2], new_tokens=3, backend=backend,
                      gguf=checkpoint, config=config_path, library=library, decoded_cache_mib=0)
    model = GlmMoeDsaModel(config, weights, engine)
    cache = model.new_cache()
    logits = model.prefill(tokens, cache)[-1]
    expected = []
    for _ in range(3):
        token = int(np.argmax(logits))
        expected.append(token)
        logits = model.forward(token, cache)
    assert result['generated_tokens'] == expected
    assert result['consumed_tokens'] == tokens + expected
    np.testing.assert_array_equal(result['next_logits'], logits)
    assert result['checkpoint_kind'] == 'gguf'
    assert len({result['prefill_pid'], result['decode_pid'], os.getpid()}) == 3
    assert result['prefill_timings']['serialize_seconds'] > 0
    assert result['decode_timings']['claim_seconds'] > 0
    assert result['prefill_timings']['identity_seconds'] > 0
    assert result['decode_timings']['identity_seconds'] > 0
    assert result['total_seconds'] > 0
