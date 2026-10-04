"""Bounded trained GGUF trace through pinned official eager CPU layer forwards.

Storage changes only: one meta-constructed decoder layer is materialized at a
time; expert bank __getitem__ lazily materializes one selected expert. Official
numerical methods are neither replaced nor copied. This is an explicit F32,
expanded-attention, batch teacher-forced mode, not native GGML/FP8 equivalence.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import time
import warnings
import zipfile

import numpy as np


NUMERICAL_MODE = 'official-eager-cpu-f32-streamed-storage-v1'
MAX_TOKENS = 32


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def official_modules():
    """Reject a different installed revision before allocating model storage."""
    import torch
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as modeling
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa as configuration
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / 'metadata/sources.json').read_text())
    for module in (modeling, configuration):
        path = Path(inspect.getfile(module))
        if sha_file(path) != pins['sources'][path.name]['sha256']:
            raise ValueError('installed official source differs from pinned revision: ' + path.name)
    return torch, modeling, configuration, pins


def prepare_config(config, configuration):
    """A deliberately narrow supported checkpoint family; never silent fallback."""
    c = dict(config)
    checks = {
        'attention_bias': False, 'mlp_bias': False, 'attention_dropout': 0.0,
        'hidden_act': 'silu', 'norm_topk_prob': True, 'scoring_func': 'sigmoid',
        'n_group': 1, 'topk_group': 1, 'ep_size': 1,
        'rope_interleave': True, 'indexer_rope_interleave': True,
    }
    for key, expected in checks.items():
        if c.get(key, expected) != expected:
            raise ValueError('unsupported official streamed configuration: ' + key)
    rope = c.get('rope_parameters') or {'rope_type': 'default', 'rope_theta': c.get('rope_theta', 10000)}
    if rope.get('rope_type', 'default') != 'default':
        raise ValueError('only default RoPE is supported by this validated stream mode')
    c['rope_parameters'] = dict(rope, rope_type='default')
    if c.get('num_key_value_heads', c['num_attention_heads']) != c['num_attention_heads']:
        raise ValueError('MLA key/value heads must match attention heads')
    for key in ('hidden_size', 'vocab_size', 'num_hidden_layers', 'n_routed_experts',
                'num_experts_per_tok', 'n_shared_experts', 'moe_intermediate_size',
                'intermediate_size', 'num_attention_heads', 'q_lora_rank',
                'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim',
                'v_head_dim', 'index_n_heads', 'index_head_dim', 'index_topk'):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError('positive integer required: ' + key)
    if c['num_experts_per_tok'] > c['n_routed_experts'] or c['n_routed_experts'] < 2:
        raise ValueError('unsupported expert count/top-k')
    if c['qk_rope_head_dim'] % 2 or c['qk_rope_head_dim'] > c['index_head_dim']:
        raise ValueError('invalid rotary dimensions')
    # Direct modules never invoke from_pretrained quantization. Preserve original
    # config in provenance; execution is explicitly F32 on decoded GGUF weights.
    c['dtype'] = 'float32'
    cfg = configuration.GlmMoeDsaConfig(**c)
    cfg._attn_implementation = 'eager'
    cfg._experts_implementation = 'eager'
    if len(cfg.indexer_types) != cfg.num_hidden_layers or cfg.indexer_types[0] != 'full' or any(x not in ('full', 'shared') for x in cfg.indexer_types):
        raise ValueError('unsupported indexer pattern')
    if len(cfg.mlp_layer_types) != cfg.num_hidden_layers or any(x not in ('dense', 'sparse') for x in cfg.mlp_layer_types):
        raise ValueError('unsupported MLP pattern')
    if cfg.layer_types != ['indexed_attention'] * cfg.num_hidden_layers:
        raise ValueError('only full GLM indexed-attention layers are supported')
    return cfg


class WeightStorage:
    """Inference-only F32 views; no retained decoded-weight cache in this wrapper."""
    def __init__(self, reader, torch, *, max_bytes):
        self.reader, self.torch, self.max_bytes = reader, torch, max_bytes
        self.loads = 0
        self.max_tensor_bytes = 0
        self.lazy_requests = []
        self.max_layer_bytes = 0

    def read(self, name, shape, *, rows=None):
        if len(shape) > 2:
            raise ValueError('refusing to materialize an expert bank or other rank>2 weight')
        expected_bytes = int(np.prod(shape)) * 4
        if expected_bytes > self.max_bytes:
            raise ValueError('weight exceeds materialization budget: ' + name)
        value = self.reader(name, rows=rows)
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or tuple(value.shape) != tuple(shape):
            raise ValueError('decoded weight must have exact F32 shape: ' + name)
        if not value.flags.c_contiguous:
            raise ValueError('decoded weight must be contiguous: ' + name)
        # GGUF readers intentionally return read-only NumPy storage. The official
        # eval/no-grad forwards only read Parameters; using a view avoids copying
        # the ~3.8GB final head. No exposed tensor is ever mutated or optimized.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given NumPy array is not writable')
            result = self.torch.from_numpy(value)
        self.loads += 1
        self.max_tensor_bytes = max(self.max_tensor_bytes, expected_bytes)
        return result


class LazyExpertBank:
    """Supports exactly the scalar accesses made by unmodified Experts.forward."""
    def __init__(self, storage, prefix, config, kind):
        self.storage, self.prefix, self.config, self.kind = storage, prefix, config, kind

    def __getitem__(self, expert):
        torch = self.storage.torch
        if isinstance(expert, torch.Tensor):
            if expert.ndim != 0 or expert.dtype not in (torch.int32, torch.int64) or expert.device.type != 'cpu':
                raise ValueError('lazy expert lookup requires one CPU integer')
            expert = int(expert.item())
        if type(expert) is not int or not 0 <= expert < self.config.n_routed_experts:
            raise ValueError('lazy expert lookup requires one valid integer ID')
        self.storage.lazy_requests.append((self.prefix, self.kind, expert))
        h, width = self.config.hidden_size, self.config.moe_intermediate_size
        prefix = self.prefix + f'.experts.{expert}'
        if self.kind == 'gate_up':
            # Peak temporary storage: two [I,H] tensors plus one [2I,H] concat.
            # Never construct [E,2I,H] or [E,H,I] on CPU.
            gate = self.storage.read(prefix + '.gate_proj.weight', (width, h))
            up = self.storage.read(prefix + '.up_proj.weight', (width, h))
            return torch.cat((gate, up), dim=0)
        if self.kind == 'down':
            return self.storage.read(prefix + '.down_proj.weight', (h, width))
        raise ValueError('unknown expert storage kind')


def materialize_layer(cfg, index, modeling, storage):
    torch = storage.torch
    with torch.device('meta'):
        layer = modeling.GlmMoeDsaDecoderLayer(cfg, index)
    if any(t.device.type != 'meta' for t in list(layer.parameters()) + list(layer.buffers())):
        raise RuntimeError('official decoder constructor allocated real weight storage')
    excluded = {'mlp.experts.gate_up_proj', 'mlp.experts.down_proj'}
    tensors = list(layer.named_parameters()) + list(layer.named_buffers())
    resident_bytes = sum(t.numel()*4 for name,t in tensors if name not in excluded)
    # Conservative weight allocation bound includes lazy concat transient; hidden
    # states/attention intermediates are accounted separately by streamed_forward.
    expert_bytes = 16 * cfg.hidden_size * cfg.moe_intermediate_size if cfg.mlp_layer_types[index] == 'sparse' else 0
    if resident_bytes + expert_bytes > storage.max_bytes:
        raise ValueError('layer plus lazy expert exceeds materialization budget')
    storage.max_layer_bytes = max(storage.max_layer_bytes, resident_bytes + expert_bytes)
    for name, meta in tensors:
        if name in excluded:
            continue
        owner_name, field = name.rsplit('.', 1)
        owner = layer.get_submodule(owner_name)
        value = storage.read(f'model.layers.{index}.{name}', tuple(meta.shape))
        if field in owner._parameters:
            setattr(owner, field, torch.nn.Parameter(value, requires_grad=False))
        else:
            setattr(owner, field, value)
    if cfg.mlp_layer_types[index] == 'sparse':
        experts = layer.mlp.experts
        del experts.gate_up_proj
        del experts.down_proj
        experts.gate_up_proj = LazyExpertBank(storage, f'model.layers.{index}.mlp', cfg, 'gate_up')
        experts.down_proj = LazyExpertBank(storage, f'model.layers.{index}.mlp', cfg, 'down')
    if any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in list(layer.parameters()) + list(layer.buffers())):
        raise RuntimeError('materialization left non-CPU/F32 storage')
    return layer.eval()


def streamed_forward(config, reader, tokens, trace=None, *, max_weight_bytes=8*1024**3):
    """Official layer forwards with bounded storage, no generation or KV history.

    reader(name, rows=None) returns a contiguous F32 ndarray. No caller may
    modify returned weights during execution. trace receives owning copies.
    """
    if not tokens or len(tokens) > MAX_TOKENS or any(type(t) is not int for t in tokens):
        raise ValueError('teacher forcing requires 1..32 integer token IDs')
    if type(max_weight_bytes) is not int or max_weight_bytes <= 0:
        raise ValueError('max_weight_bytes must be a positive integer')
    torch, modeling, configuration, _ = official_modules()
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('official streamed mode requires default torch.float32')
    cfg = prepare_config(config, configuration)
    if any(t < 0 or t >= cfg.vocab_size for t in tokens) or len(tokens) > cfg.max_position_embeddings:
        raise ValueError('tokens exceed vocabulary or context')
    storage = WeightStorage(reader, torch, max_bytes=max_weight_bytes)
    n = len(tokens)
    # This bound protects pathological configs before expanded activation storage.
    activation_bound = n * cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim) * 4 * 8
    activation_bound += n*n*(cfg.num_attention_heads+cfg.index_n_heads)*4*4
    if activation_bound > max_weight_bytes:
        raise ValueError('expanded attention activation estimate exceeds budget')
    def emit(name, tensor):
        if trace is not None:
            array = tensor.detach().cpu().numpy()
            for pos in range(n):
                trace(pos, name, np.array(array[0, pos], copy=True))
    with torch.no_grad():
        # Equivalent embedding row gather; no vocabulary table allocation.
        hidden = torch.cat([storage.read('model.embed_tokens.weight', (1,cfg.hidden_size), rows=slice(t,t+1)) for t in tokens], dim=0).unsqueeze(0)
        positions = torch.arange(n, dtype=torch.long).unsqueeze(0)
        mask = modeling.create_causal_mask(config=cfg, inputs_embeds=hidden,
                    attention_mask=None, past_key_values=None, position_ids=positions,
                    allow_is_causal_skip=False)
        rope = modeling.GlmMoeDsaRotaryEmbedding(cfg).eval()
        position_embeddings = rope(hidden, position_ids=positions)
        selected = None
        for index in range(cfg.num_hidden_layers):
            layer = materialize_layer(cfg, index, modeling, storage)
            handles = []
            if cfg.mlp_layer_types[index] == 'sparse':
                def routing_hook(module, inputs, output, index=index):
                    logits, weights, ids = output
                    emit(f'layer.{index}.mlp.router_logits', logits.unsqueeze(0))
                    emit(f'layer.{index}.mlp.route_weights', weights.unsqueeze(0))
                    emit(f'layer.{index}.mlp.route_ids', ids.unsqueeze(0))
                handles.append(layer.mlp.gate.register_forward_hook(routing_hook))
            try:
                hidden, selected = layer(hidden, attention_mask=mask,
                    position_ids=positions, position_embeddings=position_embeddings,
                    past_key_values=None, use_cache=False, prev_topk_indices=selected)
                emit(f'layer.{index}.output', hidden)
                emit(f'layer.{index}.official_selected', selected)
                if trace is not None:
                    for pos in range(n):
                        ids = selected[0,pos].detach().cpu().numpy()
                        # Official eager turns top-k into a mask. Preserve raw
                        # IDs above and expose canonical causal membership here.
                        trace(pos, f'layer.{index}.selected', np.sort(ids[ids <= pos]).astype(np.int64))
            finally:
                for handle in handles:
                    handle.remove()
                del layer
                gc.collect()
        with torch.device('meta'):
            norm = modeling.GlmMoeDsaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        norm.weight = torch.nn.Parameter(storage.read('model.norm.weight', (cfg.hidden_size,)), requires_grad=False)
        hidden = norm(hidden)
        del norm
        head_name = 'model.embed_tokens.weight' if cfg.tie_word_embeddings else 'lm_head.weight'
        head.weight = torch.nn.Parameter(storage.read(head_name, (cfg.vocab_size,cfg.hidden_size)), requires_grad=False)
        logits = head(hidden)
        emit('logits', logits)
        result = logits[0].detach().cpu().numpy().copy()
        del head
    return result, {'weight_loads':storage.loads, 'max_single_decoded_tensor_bytes':storage.max_tensor_bytes,
                    'max_layer_with_lazy_expert_weight_bytes':storage.max_layer_bytes,
                    'expanded_activation_estimate_bytes':activation_bound,
                    'lazy_expert_requests':storage.lazy_requests,
                    'storage_mode':'meta decoder layers, CPU F32 views, scalar lazy expert-bank indexing, full final head'}


class TraceArchive:
    """Write NPZ tensors immediately rather than retaining every layer in RAM."""
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True)
        self.keys = set()

    def capture(self, pos, name, value):
        key = f'p{pos}.{name}'
        if key in self.keys:
            raise ValueError('duplicate trace tensor: ' + key)
        self.keys.add(key)
        with self.archive.open(key+'.npy', 'w', force_zip64=True) as stream:
            np.save(stream, value, allow_pickle=False)

    def close(self):
        self.archive.close()


def dependency_pins(torch, modeling, configuration):
    root = Path(__file__).resolve().parents[1]
    import gguf.quants
    import numpy._core._multiarray_umath as numpy_native
    from transformers import masking_utils, modeling_rope_utils, activations
    from transformers.integrations import moe
    codec_evidence = root/'docs/parity-evidence/ggml-codec-parity.json'
    codec_pin = json.loads(codec_evidence.read_text())
    if sha_file(gguf.quants.__file__) != codec_pin['candidate_decoder_source']['sha256']:
        raise ValueError('installed GGUF codec differs from independently validated source; install gguf==0.17.1')
    paths = [Path(inspect.getfile(m)) for m in (modeling, configuration, masking_utils,
             modeling_rope_utils, activations, moe, gguf.quants)]
    paths += [Path(torch._C.__file__), Path(numpy_native.__file__)]
    paths += sorted((Path(torch.__file__).parent/'lib').glob('*.so*'))
    paths += [Path(inspect.getfile(torch.nn.Linear)), Path(inspect.getfile(torch.nn.functional))]
    # Wheel metadata/archive origin pins complement directly executed sources
    # and native numerical libraries. Hashes are streamed, never loaded whole.
    for name in ('torch','transformers','numpy','gguf','huggingface-hub'):
        distribution = importlib.metadata.distribution(name)
        paths += [Path(distribution.locate_file(file)) for file in distribution.files or []
                  if Path(file).name in ('RECORD','direct_url.json','METADATA')]
    paths += sorted(Path(gguf.quants.__file__).parent.glob('*.py'))
    paths += [Path(__file__).resolve(), root/'metadata/sources.json', codec_evidence]
    paths += [Path(importlib.import_module(name).__file__).resolve() for name in
              ('validation.trace_identity','validation.trace_gate','glm_vx.gguf_checkpoint',
               'glm_vx.gguf_reader','glm_vx.config','glm_vx.checkpoint')]
    hashes = {str(path):sha_file(path) for path in paths}
    versions = {name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','gguf','huggingface-hub')}
    return {'files_sha256':hashes,'versions':versions}


def main(argv=None):
    from validation.trace_identity import bind_model, bind_loaded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True, help='existing frozen provenance; budgets are not modified or used as a promotion gate')
    parser.add_argument('--tokens', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--decode-threads', type=int, default=4)
    parser.add_argument('--max-weight-gib', type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    if not 1 <= len(args.tokens) <= MAX_TOKENS or not 1 <= args.threads <= 64 or not 1 <= args.decode_threads <= 32 or not 1 <= args.max_weight_gib <= 64:
        parser.error('token count/thread count/memory budget out of bounds')
    config_path = args.gguf/'config.json'
    config = json.loads(config_path.read_text())
    contract = json.loads(args.contract.read_text())
    if sha_file(config_path) != contract['pins']['config_sha256']:
        parser.error('checkpoint config does not match frozen provenance')
    torch, modeling, configuration, sources = official_modules()
    prepare_config(config, configuration)
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    pins = dependency_pins(torch, modeling, configuration)
    identity = bind_model(args.gguf, args.contract)
    checkpoint_files = identity['checkpoint_files']
    from glm_vx.gguf_checkpoint import GGUFCheckpoint
    args.output.mkdir(parents=True)
    receipt = {'status':'running', 'eligible':False, 'numerical_mode':NUMERICAL_MODE,
               'scope':'actual official eager decoder/expert/attention/norm forwards, changed streamed weight storage; shared independently validated GGUF decoder; no promotion assertion',
               'transformers_revision':sources['transformers_revision'], 'token_ids':args.tokens,
               'config_sha256':sha_file(config_path), 'contract_sha256':sha_file(args.contract),
               'checkpoint_files':checkpoint_files,
               'checkpoint_hash_verification':'declared pins reused from frozen contract; checkpoint bytes NOT rehashed',
               'contract_reuse_scope':'config/checkpoint identity only; supplied token IDs define this new trace',
               'dependencies':pins, 'threads':args.threads, 'decode_threads':args.decode_threads,
               'max_weight_bytes':args.max_weight_gib*1024**3,
               'environment':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    def write_receipt():
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    write_receipt()
    started = time.monotonic()
    checkpoint = None
    archive = None
    try:
        checkpoint = GGUFCheckpoint(args.gguf, config, cache_bytes=0, decode_threads=args.decode_threads)
        receipt['loaded_checkpoint_paths'] = bind_loaded(identity, args.gguf, checkpoint.store.paths)
        archive = TraceArchive(args.output/'arrays.npz')
        def capture(pos, name, value):
            archive.capture(pos,name,value)
            if pos == len(args.tokens)-1 and name.endswith('.output'):
                print(json.dumps({'tensor':name,'seconds':time.monotonic()-started}),flush=True)
        _, stats = streamed_forward(config, checkpoint.tensor, args.tokens, capture,
                                     max_weight_bytes=args.max_weight_gib*1024**3)
        archive.close()
        if pins != dependency_pins(torch, modeling, configuration):
            raise RuntimeError('source or dependency changed during execution')
        if receipt['config_sha256'] != sha_file(config_path) or receipt['contract_sha256'] != sha_file(args.contract):
            raise RuntimeError('config or frozen provenance changed during execution')
        if identity != bind_model(args.gguf,args.contract) or receipt['loaded_checkpoint_paths'] != bind_loaded(identity,args.gguf,checkpoint.store.paths):
            raise RuntimeError('checkpoint binding changed during execution')
        for declared, original in checkpoint_files.items():
            stat = (args.gguf/Path(declared).name).stat()
            if (stat.st_size,stat.st_mtime_ns) != (original['bytes'],original['mtime_ns']):
                raise RuntimeError('checkpoint file metadata changed during execution')
        receipt.update(status='complete', tensor_count=len(archive.keys), storage=stats,
                       arrays_sha256=sha_file(args.output/'arrays.npz'))
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if archive is not None:
            archive.close()
        if checkpoint is not None:
            checkpoint.close()
        receipt.update(elapsed_seconds=time.monotonic()-started,
                       max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_receipt()
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('storage','dependencies','checkpoint_files')},indent=2))


if __name__ == '__main__':
    main()
