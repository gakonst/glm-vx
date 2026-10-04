"""Matched layer-wise chunk versus sequential prefill, synthetic weights only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.prefill import prefill
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    if a.output.exists(): p.error('output exists')
    if a.repeats < 3: p.error('at least three repeats required')
    c = tiny_config()
    # Still synthetic, but large enough to exercise matrix weight reuse.
    c.update(hidden_size=256, q_lora_rank=128, kv_lora_rank=64,
             num_attention_heads=4, qk_nope_head_dim=32, qk_rope_head_dim=16,
             v_head_dim=32, index_n_heads=4, index_head_dim=32,
             intermediate_size=512, moe_intermediate_size=128, index_topk=16)
    weights = tiny_weights(c, seed=487)
    model = GlmMoeDsaModel(c, weights, VxBackend())
    rng = np.random.default_rng(148)
    cases = []
    for count in (8, 32, 64):
        tokens = rng.integers(0, c['vocab_size'], size=count).tolist()
        def run(kind):
            cache = model.new_cache()
            if kind == 'batch':
                return prefill(model, tokens, cache)
            for token in tokens[:-1]: model.prefill_token(token, cache)
            return model.forward(tokens[-1], cache)
        reference = run('sequential')
        np.testing.assert_array_equal(run('batch'), reference)
        samples = {'sequential': [], 'batch': []}
        for repeat in range(a.repeats):
            for kind in ('sequential', 'batch') if repeat % 2 else ('batch', 'sequential'):
                start = time.perf_counter()
                output = run(kind)
                samples[kind].append(time.perf_counter() - start)
                np.testing.assert_array_equal(output, reference)
        medians = {key: statistics.median(values) for key, values in samples.items()}
        cases.append({'prompt_tokens': count, 'seconds': samples, 'median_seconds': medians,
                      'speedup': medians['sequential']/medians['batch'],
                      'logits': 'bitwise identical in measured runs'})
    report = {'status': 'complete', 'scope': 'synthetic resident-weight CPU prefill; not trained GLM throughput or HTTP latency',
              'library_sha256': hashlib.sha256(model.backend.library_path.read_bytes()).hexdigest(),
              'config': c, 'seed': 487, 'cases': cases}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k:v for k,v in item.items() if k!='seconds'} for item in cases], indent=2))


if __name__ == '__main__': main()
