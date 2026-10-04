"""Integration tests for model-facing VxBackend, run with unittest or pytest."""
import unittest
import numpy as np
from kernels.backend import VxBackend
from glm_vx.backend import NumpyBackend


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vx = VxBackend()
        cls.ref = NumpyBackend()

    def test_model_operations(self):
        rng = np.random.default_rng(531)
        x = rng.normal(size=32).astype(np.float32)
        w = rng.normal(size=(13,32)).astype(np.float32)
        for name, args in [('matvec',(w,x)), ('rmsnorm',(x,x+1,1e-5)),
                           
                           ('softmax',(x.reshape(2,16),)), ('swiglu',(x,x+0.5))]:
            with self.subTest(name=name):
                np.testing.assert_allclose(getattr(self.vx,name)(*args),
                                           getattr(self.ref,name)(*args),rtol=3e-5,atol=3e-5)
        args=(x,x[::-1],8,2.5)
        actual, expected=self.vx.route(*args),self.ref.route(*args)
        np.testing.assert_array_equal(actual[0],expected[0])
        np.testing.assert_allclose(actual[1],expected[1],rtol=3e-5,atol=3e-5)

    def test_noncontiguous_and_batched(self):
        x=np.arange(64,dtype=np.float32).reshape(8,8)[:,::2]
        angles=20*8000000.0**(-np.arange(0,4,2,dtype=np.float32)/4)
        expected=np.concatenate((x[...,::2]*np.cos(angles)-x[...,1::2]*np.sin(angles),
                                 x[...,1::2]*np.cos(angles)+x[...,::2]*np.sin(angles)),axis=-1)
        np.testing.assert_allclose(self.vx.rope(x,20,8000000),expected,rtol=3e-5,atol=3e-5)
        out=self.vx.rmsnorm(x,np.ones(4,np.float32),1e-5)
        ref=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)
        np.testing.assert_allclose(out,ref,rtol=3e-5,atol=3e-5)

    def test_rejects_unsafe_shapes(self):
        calls=[lambda:self.vx.matvec(np.ones((2,3)),np.ones(2)),
               lambda:self.vx.rope(np.ones(3),0,10000),
               lambda:self.vx.softmax([-np.inf,-np.inf]),
               lambda:self.vx.route([1,2],[0],1,2.5),
               lambda:self.vx.route([1,2],[0,0],3,2.5),
               lambda:self.vx.swiglu([1],[1,2])]
        for operation in calls:
            with self.assertRaises(ValueError): operation()

    def test_tiny_glm_forward_against_independent_oracle(self):
        from tests.test_model import tiny_fixture, expanded_batch_oracle
        from glm_vx.model import GlmMoeDsaModel
        config, weights=tiny_fixture()
        tokens=[1,7,2,11,4,3,16]
        expected,_=expanded_batch_oracle(config,weights,tokens)
        model=GlmMoeDsaModel(config,weights,self.vx)
        actual=model.prefill(tokens)
        np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=4e-6)

    def test_optional_native_ops(self):
        rng=np.random.default_rng(532)
        x=rng.normal(size=(3,128)).astype(np.float32)
        w=rng.normal(size=128).astype(np.float32)
        bias=rng.normal(size=128).astype(np.float32)
        ref=(x.astype(float)-x.astype(float).mean(-1,keepdims=True))/np.sqrt(x.astype(float).var(-1,keepdims=True)+1e-6)*w+bias
        np.testing.assert_allclose(self.vx.layernorm(x,w,bias),ref,rtol=3e-5,atol=3e-5)
        scores=np.array([3,1,3,-2],np.float32)
        np.testing.assert_array_equal(self.vx.topk(scores,3),[0,2,1])
        self.assertEqual(self.vx.topk(scores,0).size,0)
        q=rng.normal(size=(4,128)).astype(np.float32)
        keys=rng.normal(size=(7,128)).astype(np.float32)
        weights=rng.normal(size=4).astype(np.float32)/2
        ref=weights.astype(float)@np.maximum(q.astype(float)@keys.astype(float).T/np.sqrt(128),0)
        np.testing.assert_allclose(self.vx.index_scores(q,keys,weights),ref,rtol=3e-5,atol=3e-5)
        with self.assertRaises(ValueError): self.vx.topk(scores,5)
        with self.assertRaises(ValueError): self.vx.index_scores(q,keys,weights[:2])

    def test_rope_cache_reuse_and_bounded_eviction(self):
        from kernels.backend import _cached_rope_coefficients
        _cached_rope_coefficients.cache_clear()
        x=np.arange(64,dtype=np.float32)
        expected=self.vx.rope(x,7,8000000)
        for _ in range(3):
            np.testing.assert_array_equal(self.vx.rope(x,7,8000000),expected)
        info=_cached_rope_coefficients.cache_info()
        self.assertEqual(info.misses,1)
        self.assertEqual(info.hits,3)
        for pos in range(140): self.vx.rope(x,pos,8000000)
        self.assertLessEqual(_cached_rope_coefficients.cache_info().currsize,128)
        np.testing.assert_array_equal(self.vx.rope(x,7,8000000),expected)
        before=_cached_rope_coefficients.cache_info()
        self.vx.rope(np.ones(4098,np.float32),7,8000000)
        self.assertEqual(_cached_rope_coefficients.cache_info(),before)
        cosine,sine=_cached_rope_coefficients(64,7.0,8000000.0)
        self.assertFalse(cosine.flags.writeable)
        self.assertFalse(sine.flags.writeable)

    def test_missing_library_is_failure(self):
        with self.assertRaises(RuntimeError):VxBackend('/nonexistent/libglm_vx.so')

if __name__=='__main__':unittest.main()
