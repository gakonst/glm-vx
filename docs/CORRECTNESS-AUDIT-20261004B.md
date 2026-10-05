# DSA configuration correctness audit, 2026-10-04

Base: `3ae29e5`. CPU only. No trained model execution, model-weight download,
or trained layer reads. Frozen gates and numerical thresholds are unchanged.

## Confirmed bug and fix

`GLMConfig.from_dict()` inherited the dataclass's GLM-5.3-specific schedule
(frequency 4, offset 3) when a checkpoint omitted either field. The pinned
Transformers implementation defaults each omitted field independently to
frequency 1 and offset 2. The direct `GlmMoeDsaModel` constructor already used
the official defaults. Loading the same checkpoint through validation therefore
silently changed its layer topology and omitted required indexer computation.

The fix sets the two official defaults at the checkpoint-dictionary boundary.
Explicit `indexer_types`, explicit patterns, and partial scheduling metadata
retain their precedence. Direct dataclass construction retains its documented
GLM-5.3 convenience defaults. The shipped checkpoint explicitly supplies both
fields, so its resolved topology is unchanged.

This is a discrete correctness error, not floating-point drift. In a deterministic
six-token random fixture (seed 31, index_topk 2), the final layer selected `[4, 0]`
before the fix instead of `[0, 5]`. Its index-key cache contained zero entries
instead of six. Sequential execution changed 1,024/1,536 logits; the largest
absolute difference was 0.0497622 with Vx. Batched prefill changed all 256 final
logits. After the fix, each backend matches its explicit-all-full control
**exactly**, including subsequent decode. This does not establish trained-model
numerical parity or address the existing 112/308 frozen numerical failures.

Evidence:

- [Source hashes and before/after measurements](audit-evidence-20261004b/config-defaults.json)
- [Original failing regression: 6 failed, 4 passed](audit-evidence-20261004b/config-before.txt)
- [Final focused suite: 161 passed, 33 subtests passed](audit-evidence-20261004b/config-final-tests.txt)
- [Reproduction script](audit-evidence-20261004b/verify_config_defaults.py)
- [Zero and extreme finite numeric probes](audit-evidence-20261004b/numeric-probes.json)

## Primary-source comparison

The official configuration and model were downloaded again at the repository's
pinned Transformers revision. Their SHA-256 hashes exactly match
`metadata/sources.json`. Current vLLM and SGLang main heads were resolved once
and the inspected sources fetched by those immutable SHAs.

- [Official configuration at 4692303, lines 136–148](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py#L136): defaults 1/2; explicit types precede pattern, pattern precedes schedule.
- [vLLM DSA attention at 1388100, lines 167–177](https://github.com/vllm-project/vllm/blob/138810056093301f4881050fcf2b1786939da387/vllm/models/deepseek_v32/attention.py#L167): defaults 1/2 and the same offset formula.
- [SGLang configuration at 92d6035, lines 378–412](https://github.com/sgl-project/sglang/blob/92d60351e2eb0dc2a947ff4a18c79bc54b6c77ce/python/sglang/srt/configs/model_config.py#L378): default frequency 1; missing offset implements `max(layer_id - 1, 0)`; explicit per-layer types take precedence.

The reproduction script executes the pinned official `__post_init__` method's
unmodified AST for five schedule cases, with only its parent method stubbed.
This checks the actual method's schedule computation without importing optional
Transformers/PyTorch packages. It is **not** a full official runtime execution.
The numerical before/after measurements use actual tiny GLM-Vx models and a
freshly compiled Vx CPU library.

## Other inspected paths and limits

No additional substantive mismatch was established in these bounded checks:

- DSA score scaling, signed head weights after ReLU, causal eligible keys, and
  shared-layer propagation match the [official indexer and attention](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L236).
  Fresh full layers replace the carried selection; shared layers reuse it.
  [SGLang's shared-layer guard](https://github.com/sgl-project/sglang/blob/92d60351e2eb0dc2a947ff4a18c79bc54b6c77ce/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L135)
  confirms that a shared layer must not recompute with absent indexer weights.
  Top-k tie policies remain an explicitly documented difference; this audit
  does not claim identical tied membership across frameworks.
- MLA absorption uses the correct key transpose and applies the value projection
  to the attended latent. This agrees with the [official expanded definition](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L359)
  and [vLLM's absorbed query path](https://github.com/vllm-project/vllm/blob/138810056093301f4881050fcf2b1786939da387/vllm/models/deepseek_v32/attention.py#L423).
  The Q/KV RMSNorm epsilon of `1e-6` is intentional: the [pinned official constructors](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L333)
  omit `config.rms_norm_eps` and inherit that default. Current provider kernels
  need not have identical rounding or epsilon behavior; they are corroborating
  architecture sources, not substitutes for the pinned official oracle.
- Underflowed router probabilities produce zero weights without division by zero,
  using the [official denominator's `1e-20`](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L494).
  Zero RMSNorm/LayerNorm and extreme finite softmax inputs pass bounded native
  probes. This is not exhaustive overflow robustness coverage.
- Existing CPU tests verify request-local caches, causal pruning around
  2047/2048/2049/4096 tokens, batched/sequential cache equivalence, prefix snapshot
  ownership, interleaved requests and validated handoff. The inspected prefix
  cache namespaces model/backend/weights/config/revision and unpickles owning
  state on every hit. This numerical audit did not inspect mutation of resolved execution fields;
  the parallel [serving audit](SERVING-AUDIT-20261004B.md) subsequently reproduced
  and fixed namespace omissions for those fields and packed mode.

## Reproduce

From this checkout:

```sh
source ../vx-toolchain/env.sh
kernels/build.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  ../glm-vx/.venv/bin/python -m pytest -q \
  tests/test_config_dsa_defaults.py tests/test_checkpoint.py \
  tests/test_checkpoint_model.py tests/test_model.py tests/test_model_trace.py \
  tests/test_batched_prefill.py tests/test_prefix_cache.py tests/test_concurrency.py \
  tests/test_backend_oracle.py tests/test_dsa_boundaries.py tests/test_handoff.py
```

For the separate evidence script, populate `build/upstream-audit/<file>` from each
immutable URL in `config-defaults.json` (its `sources` array), then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  ../glm-vx/.venv/bin/python docs/audit-evidence-20261004b/verify_config_defaults.py
```

The script reads the old configuration with `git show 3ae29e5:glm_vx/config.py`
and executes it in an isolated Python module. It does not edit main or the
current configuration. The JSON receipt records the exact Vx library hash.
