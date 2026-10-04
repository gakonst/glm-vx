"""Synthetic on-disk checkpoint -> loader -> model -> compiled Vx parity."""
import json
import struct
import numpy as np
from glm_vx.checkpoint import Checkpoint
from glm_vx.config import GLMConfig
from glm_vx.model import Model,tiny_weights
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend

def test_disk_checkpoint_matches_in_memory(tmp_path):
    config=tiny_config();weights=tiny_weights(config,seed=123)
    # Write a complete tiny safetensors fixture independently of loader.
    header={};chunks=[];offset=0
    for name,w in weights.items():
        raw=np.asarray(w,dtype='<f4').tobytes()
        header[name]={'dtype':'F32','shape':list(w.shape),'data_offsets':[offset,offset+len(raw)]}
        offset+=len(raw);chunks.append(raw)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(chunks))
    (tmp_path/'config.json').write_text(json.dumps(config))
    validated=GLMConfig.from_file(tmp_path)
    with Checkpoint(tmp_path) as checkpoint:
        checkpoint.validate_manifest(validated.tensor_shapes(),inspect_shards=True)
        # Callable works independently of mapping sugar in the loader.
        disk=Model(config,checkpoint.tensor,VxBackend())
        memory=Model(config,weights,VxBackend())
        np.testing.assert_array_equal(disk.prefill([1,3,5,7,9,11]),memory.prefill([1,3,5,7,9,11]))
