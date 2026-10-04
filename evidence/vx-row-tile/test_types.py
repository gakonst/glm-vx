import ctypes as C
from pathlib import Path
import numpy as np
lib=C.CDLL(str(Path('evidence/vx-row-tile/types.so').resolve()))
for name,kind in [('half_to_float','f16'),('bf16_to_float','bf16')]:
 f=getattr(lib,name);f.argtypes=[C.c_void_p,C.c_void_p,C.c_int32];f.restype=C.c_int32
 bits=np.arange(65536,dtype='u2')
 expected=bits.view('f2').astype('f4') if kind=='f16' else (bits.astype('u4')<<16).view('f4')
 out=np.empty(65536,'f4');assert f(out.ctypes.data,bits.ctypes.data,len(bits))==0
 finite=np.isfinite(expected)
 np.testing.assert_array_equal(out[finite].view('u4'),expected[finite].view('u4'))
 assert np.array_equal(np.isnan(out),np.isnan(expected))
 assert np.array_equal(np.isinf(out),np.isinf(expected))
 print(name, int(finite.sum()), 'finite bit patterns exact; NaN/Inf classes matched')
