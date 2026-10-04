"""Precision/reduction fixtures independent of the frozen trained-model gate."""
import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel
from tests.test_model import tiny_fixture
from validation.expanded_reference import expanded_batch_oracle
from validation.stage_replay import compare_stages, replay

# Existing immutable contract evidence, not knobs selected from candidate output.
BUDGET = dict(atol=4e-6, rtol=4e-5, normalized_l2=4e-5,
              normalization_floor=1e-6, softmax_tv=4e-5,
              top_k=5, top_k_margin_atol=4e-5)


def capture_run(mode='f32-v2', observer=None):
    c, w = tiny_fixture()
    trace = {}
    def capture(pos, name, value):
        trace[f'p{pos}.{name}'] = value.copy()
        if observer is not None:
            observer(value)
    logits, selected = expanded_batch_oracle(c, w, [1, 7, 2, 11], capture, numerical_mode=mode)
    return logits, selected, trace


def test_reference_f32_is_f32_at_every_operation():
    _, _, trace = capture_run()
    assert len(trace) > 400
    assert {v.dtype for v in trace.values()} == {np.dtype('float32'), np.dtype('int64')}
    for pos in range(4):
        for layer in range(4):
            for head in range(2):
                prob = trace[f'p{pos}.layer.{layer}.attention.head.{head}.probabilities']
                np.testing.assert_allclose(prob.sum(), 1, atol=2e-7, rtol=0)


def test_legacy_mode_exposes_original_precision_promotion():
    _, _, trace = capture_run('legacy-v1')
    # Pinned environment has NumPy 2. Under NumPy 1 the historical mode follows
    # that version's scalar promotion instead; never relabel it as universal F32.
    if int(np.__version__.split('.')[0]) >= 2:
        for key in ('p1.layer.0.indexer.head_weights', 'p1.layer.0.indexer.scores',
                    'p1.layer.0.attention.head.0.probabilities'):
            assert trace[key].dtype == np.float64
    assert 'p1.layer.1.mlp.shared_first_sum' in trace
    assert 'p1.layer.1.mlp.routed_sum' not in trace


def test_observer_cannot_mutate_independent_reference():
    expected, _, _ = capture_run()
    actual, _, _ = capture_run(observer=lambda value: value.fill(0))
    np.testing.assert_array_equal(actual, expected)


def test_complete_common_stage_coverage_and_replay():
    c, w = tiny_fixture()
    trace = {}
    model = GlmMoeDsaModel(c, w, NumpyBackend(), trace=lambda p,n,v: trace.__setitem__(f'p{p}.{n}',v))
    model.prefill([1, 7, 2, 11])
    _, _, independent = capture_run()
    assert trace.keys() == independent.keys()
    for layer in range(4):
        inputs = np.stack([trace[f'p{p}.layer.{layer}.input'] for p in range(4)])
        selections = [trace[f'p{p}.layer.{layer}.selected'] for p in range(4)]
        ref, cand = replay(c, w, NumpyBackend(), layer, inputs, selections)
        report = compare_stages(ref, cand, BUDGET)
        assert not report['missing_reference'] and not report['missing_candidate']
        assert report['first_outside_inherited_budget'] is None
        for p in range(4):
            key = f'p{p}.layer.{layer}.output'
            np.testing.assert_array_equal(cand[key], trace[key])


def test_moe_cancellation_matches_official_expert_id_then_shared_order():
    c, w = tiny_fixture()
    class RoutedBackend(NumpyBackend):
        def route(self, *args):
            return np.array([2, 0, 1]), np.array([1, 1, 1], np.float32)
    class CancellationModel(GlmMoeDsaModel):
        def _expert(self, prefix, expert, x, position=0):
            return np.full(self.hidden, [1e8, -1e8, 3][expert], np.float32)
        def _mlp(self, prefix, x, position=0):
            return np.ones(self.hidden, np.float32)
    m = CancellationModel(c, w, RoutedBackend())
    # Router-order accumulation produces 1; shared-first accumulation produces
    # 3. Official expert-ID accumulation followed by shared produces 4.
    np.testing.assert_array_equal(m._feed_forward(1, np.zeros(m.hidden, np.float32)), np.full(m.hidden, 4, np.float32))


def test_reference_moe_reduction_and_bias_only_selection():
    _, _, trace = capture_run()
    c, w = tiny_fixture()
    for pos in range(4):
        stage = f'p{pos}.layer.1.mlp'
        ids, weights = trace[stage+'.route_ids'], trace[stage+'.route_weights']
        logits = trace[stage+'.router_logits']
        probabilities = 1 / (1 + np.exp(-logits))
        expected_ids = np.argsort(-(probabilities + w['model.layers.1.mlp.gate.e_score_correction_bias']), kind='stable')[:2]
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(weights, probabilities[ids] / probabilities[ids].sum() * c['routed_scaling_factor'], rtol=0, atol=0)
        total = np.zeros(c['hidden_size'], np.float32)
        for expert in sorted(ids):
            total += trace[stage+f'.expert.{expert}.weighted']
            np.testing.assert_array_equal(total, trace[stage+f'.expert.{expert}.partial_sum'])
        np.testing.assert_array_equal(total, trace[stage+'.routed_sum'])
        np.testing.assert_array_equal(total+trace[stage+'.shared_output'], trace[stage+'.output'])


def test_diagnostics_separate_rounding_budget_dtype_and_coverage():
    ref = {'first': np.array([1], np.float32), 'second': np.array([1], np.float32),
           'only_reference': np.array([0], np.float32)}
    cand = {'first': np.nextafter(ref['first'], np.float32(2)), 'second': np.array([2], np.float32),
            'only_candidate': np.array([0], np.float32)}
    report = compare_stages(ref, cand, BUDGET)
    assert report['first_nonidentical']['key'] == 'first'
    assert report['first_outside_inherited_budget']['key'] == 'second'
    assert report['missing_candidate'] == ['only_reference']
    assert report['missing_reference'] == ['only_candidate']
    report = compare_stages({'a':np.ones(1,np.float64)}, {'a':np.ones(1,np.float32)}, BUDGET)
    assert 'dtype' in report['first_outside_inherited_budget']['failure']


def test_shared_replay_without_selection_fails():
    c, w = tiny_fixture()
    with pytest.raises(ValueError, match='selection'):
        expanded_batch_oracle(c, w, [1], start_layer=1, stop_layer=2,
                              hidden_states=np.zeros((1, c['hidden_size']), np.float32), output_logits=False)


def test_linear_replay_keeps_f64_explicit_and_uses_identical_inputs():
    from validation.linear_replay import linear_variants
    weight = np.array([[1e8, 1, -1e8], [2, 3, 4]], np.float32)
    inputs = np.ones((2, 3), np.float32)
    result = linear_variants(weight, inputs, NumpyBackend())
    assert all(x.dtype == np.float32 for x in result.values())
    np.testing.assert_array_equal(result['f64_diagnostic_rounded_f32'], [[1, 9], [1, 9]])
    np.testing.assert_array_equal(inputs, np.ones((2, 3), np.float32))


def test_logit_diagnostics_apply_frozen_discrete_contract_too():
    ref = {'p0.logits': np.array([1, 1, 0, -1, -2, -3], np.float32)}
    cand = {'p0.logits': ref['p0.logits'].copy()}
    cand['p0.logits'][1] = np.nextafter(np.float32(1), np.float32(2))
    assert compare_stages(ref, cand, BUDGET)['first_outside_inherited_budget'] is None
    checked = compare_stages(ref, cand, BUDGET, logits_budget=BUDGET)
    assert 'top-k' in checked['first_outside_inherited_budget']['failure']


def test_legacy_oracle_preserves_original_equations_bitwise():
    from tests.test_model import expanded_batch_oracle as original_oracle
    c, w = tiny_fixture()
    expected, expected_selections = original_oracle(c, w, [1, 7, 2, 11])
    actual, actual_selections = expanded_batch_oracle(c, w, [1, 7, 2, 11], numerical_mode='legacy-v1')
    np.testing.assert_array_equal(actual, expected)
    for left_layer, right_layer in zip(expected_selections, actual_selections):
        for left, right in zip(left_layer, right_layer):
            np.testing.assert_array_equal(left, right)
