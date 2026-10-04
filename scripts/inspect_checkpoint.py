#!/usr/bin/env python3
"""Validate checkpoint metadata and print explicit compressed-cache accounting."""
import argparse
import json
from glm_vx.config import GLMConfig
from glm_vx.checkpoint import Checkpoint
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory');p.add_argument('--inspect-shards',action='store_true')
p.add_argument('--tokens',type=int,default=8192);p.add_argument('--sequences',type=int,default=1)
a=p.parse_args()
if min(a.tokens,a.sequences)<1:p.error('tokens and sequences must be positive')
c=GLMConfig.from_file(a.directory)
with Checkpoint(a.directory,block_size=c.weight_block_size) as checkpoint:
    manifest=checkpoint.validate_manifest(c.tensor_shapes(),inspect_shards=a.inspect_shards)
    # Current runtime stores float32 latents and index keys; Python object and
    # workspace overhead omitted deliberately and reported separately.
    floats=c.num_hidden_layers*(c.kv_lora_rank+c.qk_rope_head_dim)+sum(t=='full' for t in c.indexer_types)*c.index_head_dim
    result={'manifest':manifest,'checkpoint_declared_bytes':checkpoint.metadata.get('total_size'),
        'cache_float32_bytes_per_token':floats*4,'cache_payload_bytes':floats*4*a.tokens*a.sequences,
        'cache_note':'payload only; excludes Python objects, selected-index arrays, workspace, weights, and allocator overhead',
        'full_model_execution_verified':False}
    print(json.dumps(result,indent=2))
