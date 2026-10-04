"""Packed codecs: independent raw-byte native GGML oracle and f32 dot oracle.

No candidate gguf-py decoder is used to generate expected packed results.
GLM_VX_LIBRARY selects the exact O0/O3 candidate library under test.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import struct
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
import pytest
from kernels.backend import VxBackend
from kernels.packed import LAYOUTS, lookup_tables, unpack_for_validation
from glm_vx.gguf_reader import GGUFStore
from glm_vx.checkpoint import CheckpointError
from validation.ggml_oracle import NativeGGML, DEFAULT_REFERENCE, DEFAULT_CHECKPOINT
from test_gguf_reader import write_gguf, q8_bytes


@pytest.fixture(scope='module')
def backend():
    return VxBackend()


@pytest.fixture(scope='module')
def native():
    ref=Path(os.environ.get('GLM_VX_GGML_REFERENCE',DEFAULT_REFERENCE))
    if not (ref/'build/bin/libggml-base.so').exists():
        pytest.skip('pinned native GGML codec oracle is not installed')
    return NativeGGML(ref)


def payload(kind, blocks=12):
    block,size,name=LAYOUTS[kind]
    rng=np.random.default_rng(934+kind)
    raw=rng.integers(0,256,size=(blocks,size),dtype=np.uint8)
    positions={'q2_k':[80,82],'q3_k':[108],'q6_k':[208],
               'q4_k':[0,2],'q5_k':[0,2]}.get(name,[0])
    # Scale signs, zero, normal extremes, and IEEE half subnormals.
    scales=np.resize(np.array([.125,-.25,.03125,.75,0,-0.,2**-24,-2**-24,.001,12],np.float16),blocks)
    for p in positions:raw[:,p:p+2]=scales.view(np.uint8).reshape(-1,2)
    return raw


def ordered_dot(weight,x):
    # Independent NumPy f32 multiply then ordered f32 accumulation, no BLAS and
    # no gguf-py decode. Equivalent rounding order to existing Vx matvec.
    return np.cumsum(weight*x,dtype=np.float32,axis=1)[:,-1]


@pytest.mark.parametrize('kind',LAYOUTS)
def test_synthetic_raw_bytes_codec_and_dot_exact(backend,native,kind,monkeypatch):
    import gguf
    monkeypatch.setattr(gguf.quants,'dequantize',lambda *a:pytest.fail('candidate decoder called'))
    block,size,name=LAYOUTS[kind];raw=payload(kind)
    if kind==1:
        expected=np.array([struct.unpack('<e',r.tobytes())[0] for r in raw],np.float32)
    else:expected=native.decode(raw.tobytes(),name.upper(),kind,12*block)
    expected=expected.reshape(3,4*block)
    actual=unpack_for_validation(backend.lib,raw,kind,expected.shape)
    np.testing.assert_array_equal(actual,expected)
    assert actual.tobytes()==expected.tobytes()
    for x in [np.ones(4*block,np.float32),np.random.default_rng(kind).normal(size=4*block).astype(np.float32)]:
        np.testing.assert_array_equal(backend.packed_matvec(raw,kind,expected.shape,x),ordered_dot(expected,x))


def test_iq1_all_2048_grid_indices_and_scale_bits(backend,native):
    raw=np.zeros((64,50),np.uint8)
    raw[:,:2]=np.frombuffer(struct.pack('<e',.5),np.uint8)
    for block in range(64):
        for g in range(8):
            high=((block+g)%8)<<12 | ((block+g)%2)<<15
            for l in range(4):
                index=block*32+g*4+l
                raw[block,2+g*4+l]=index&255
                high|=(index>>8)<<(3*l)
            raw[block,34+2*g:36+2*g]=np.frombuffer(struct.pack('<H',high),np.uint8)
    expected=native.decode(raw.tobytes(),'IQ1_S',19,64*256).reshape(64,256)
    np.testing.assert_array_equal(unpack_for_validation(backend.lib,raw,19,expected.shape),expected)


def test_half_lookup_all_finite_bit_patterns(backend):
    bits=np.arange(65536,dtype=np.uint16)
    mask=(bits&0x7c00)!=0x7c00
    bits=bits[mask]
    raw=bits.view(np.uint8)
    actual=unpack_for_validation(backend.lib,raw,1,(1,len(bits))).ravel()
    expected=bits.view(np.float16).astype(np.float32)
    assert actual.tobytes()==expected.tobytes()


@pytest.fixture(scope='module')
def real_store():
    checkpoint=Path(os.environ.get('GLM_VX_GGUF_CHECKPOINT',DEFAULT_CHECKPOINT))
    if not checkpoint.exists():pytest.skip('optional real checkpoint absent')
    store=GGUFStore(checkpoint)
    yield store
    store.close()


@pytest.mark.parametrize('kind', [0,8,10,11,12,13,14,16,18,19,23])
def test_trained_rows_experts_and_independent_positional_bytes(backend,native,kind,monkeypatch,real_store):
    import gguf
    store=real_store
    tensors=[t for t in store.tensors.values() if int(t.tensor_type)==kind and len(store.shape(t.name))>=2]
    assert tensors
    tensor=tensors[len(tensors)//2];shape=store.shape(tensor.name)
    block,size=(1,4) if kind==0 else LAYOUTS[kind][:2]
    path=next(p for p,r in zip(store.paths,store.readers) if any(t is tensor for t in r.tensors))
    experts=[0,shape[0]//2,shape[0]-1] if len(shape)==3 else [None]
    x=np.random.default_rng(kind).normal(size=shape[-1]).astype(np.float32)
    monkeypatch.setattr(gguf.quants,'dequantize',lambda *a:pytest.fail('candidate decoder called'))
    with path.open('rb') as f:
        for expert in experts:
            for row in sorted({0,shape[-2]//2,shape[-2]-1}):
                nbytes=shape[-1]//block*size
                offset=int(tensor.data_offset)+((expert or 0)*shape[-2]+row)*nbytes
                raw=os.pread(f.fileno(),nbytes,offset)
                expected=native.decode(raw,tensor.tensor_type.name,kind,shape[-1]).reshape(1,-1)
                with store.packed_rows(tensor.name,expert=expert,rows=slice(row,row+1)) as (view,dims,fmt):
                    assert view.tobytes()==raw
                    assert not view.flags.owndata
                    if kind:
                        np.testing.assert_array_equal(unpack_for_validation(backend.lib,view,kind,dims),expected)
                    np.testing.assert_array_equal(backend.packed_matvec(view,fmt,dims,x),ordered_dot(expected,x))



@pytest.mark.parametrize('change,error',[
    ({'raw':np.zeros(49,np.uint8)},'byte count'),
    ({'raw':np.zeros(50,np.float32)},'uint8'),
    ({'raw':np.zeros(100,np.uint8)[::2]},'contiguous'),
    ({'shape':(1,255)},'block-aligned'),({'shape':(-1,256)},'nonnegative'),
    ({'shape':(2**31,256)},'int32'),({'shape':(1,0)},'positive'),
    ({'shape':(True,256)},'integer'),({'kind':999},'unsupported'),
    ({'x':np.ones(255,np.float32)},'columns'),
    ({'x':np.full(256,np.nan,np.float32)},'finite')])
def test_bounds_before_kernel(backend,change,error):
    args=dict(raw=np.zeros(50,np.uint8),kind=19,shape=(1,256),x=np.ones(256,np.float32));args.update(change)
    with pytest.raises(ValueError,match=error):backend.packed_matvec(**args)


@pytest.mark.parametrize('selection',[
    {'expert':-1},{'expert':3},{'expert':True},{'expert':1.0},{},
    {'expert':0,'rows':slice(None,None,2)}, {'expert':0,'rows':[1,2]}])
def test_store_rejects_bad_expert_or_rows(tmp_path,selection):
    import gguf
    raw,_=q8_bytes((3,7,64));p=write_gguf(tmp_path/'q.gguf',[('w',raw,gguf.GGMLQuantizationType.Q8_0)])
    store=GGUFStore(p)
    try:
        with pytest.raises(CheckpointError):
            with store.packed_rows('w',**selection):pytest.fail('invalid selection accepted')
    finally:store.close()


def test_empty_rows_and_close_waits_for_borrow(tmp_path,backend):
    import gguf
    raw,_=q8_bytes((3,7,64));p=write_gguf(tmp_path/'q.gguf',[('w',raw,gguf.GGMLQuantizationType.Q8_0)])
    store=GGUFStore(p);started=Event()
    def close():started.set();store.close()
    with ThreadPoolExecutor(1) as pool:
        with store.packed_rows('w',expert=2,rows=slice(4,2)) as (view,shape,kind):
            assert backend.packed_matvec(view,kind,shape,np.ones(64,np.float32)).shape==(0,)
            future=pool.submit(close);assert started.wait(2)
            assert not store.closed
        future.result(timeout=2)
    with pytest.raises(RuntimeError,match='closed'):
        with store.packed_rows('w'):pass


def test_linear_uses_borrowed_bytes_and_explicit_fallback(tmp_path,backend,monkeypatch):
    from test_gguf_checkpoint import fixture_data,write_fixture
    from glm_vx.gguf_checkpoint import GGUFCheckpoint
    from glm_vx.backend import NumpyBackend
    c,w=fixture_data();p=write_fixture(tmp_path/'model.gguf',c,w)
    cp=GGUFCheckpoint(p,c)
    try:
        name='model.layers.1.mlp.experts.1.up_proj.weight';x=np.ones(w[name].shape[1],np.float32)
        expected=backend.matvec(w[name],x)
        original=cp.tensor
        monkeypatch.setattr(cp,'tensor',lambda *a:pytest.fail('packed linear expanded weights'))
        np.testing.assert_array_equal(cp.linear(name,x,backend),expected)
        assert cp.cache_bytes==0
        monkeypatch.setattr(cp,'tensor',original)
        np.testing.assert_allclose(cp.linear(name,x,NumpyBackend()),expected,atol=1e-6)
        # Reconstructed K-B/V-B still follows the explicit old path.
        name='model.layers.1.self_attn.kv_b_proj.weight';x=np.ones(w[name].shape[1],np.float32)
        np.testing.assert_array_equal(cp.linear(name,x,backend),backend.matvec(cp.tensor(name),x))
    finally:cp.close()


def test_corrupt_scale_fails_closed(backend):
    raw=np.zeros(50,np.uint8);raw[:2]=np.frombuffer(struct.pack('<e',float('inf')),np.uint8)
    with pytest.raises(ValueError,match='nonfinite'):backend.packed_matvec(raw,19,(1,256),np.ones(256,np.float32))


def test_table_extraction_is_reproducible(native,tmp_path):
    from kernels.extract_ggml_tables import extract
    extract(native.reference,tmp_path)
    for name in ['ggml_tables.bin','ggml_tables.json','GGML-LICENSE']:
        assert (tmp_path/name).read_bytes()==(Path(__file__).resolve().parents[1]/'kernels'/name).read_bytes()


def test_backend_without_packed_symbols_has_explicit_fallback():
    from kernels.packed import supports
    class OldLibrary:pass
    assert supports(OldLibrary(),0)
    assert not supports(OldLibrary(),19)
    assert not supports(OldLibrary(),999)


def test_f32_capacity_counts_elements_and_rejects_unaligned_bytes(backend):
    from kernels.packed import _buffers
    # No giant allocation is needed to distinguish a valid element count from
    # an invalid byte-size limit: this must reach byte-count validation.
    with pytest.raises(ValueError, match='byte count'):
        _buffers(np.empty(0, np.uint8), 0, (154880, 6144))
    with pytest.raises(ValueError, match='int32'):
        _buffers(np.empty(0, np.uint8), 0, (2**30, 2))
    # Non-F32 kernels still require byte addressing to fit their int32 ABI.
    with pytest.raises(ValueError, match='int32'):
        _buffers(np.empty(0, np.uint8), 1, (2**30, 1))
    raw = np.zeros(17, np.uint8)[1:]
    with pytest.raises(ValueError, match='aligned'):
        backend.packed_matvec(raw, 0, (1, 4), np.ones(4, np.float32))
