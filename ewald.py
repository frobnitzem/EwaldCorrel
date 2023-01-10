#!/usr/bin/env python3

import os
from functools import reduce
from math import pi, exp, erfc

from ctypes import CDLL, c_void_p, c_int, c_double, POINTER, \
                   Structure, PYFUNCTYPE, byref, addressof, cast

import numpy as np
import numpy.ctypeslib as ct

array = np.array

def describe_sfac():
    # Boilerplate declaration code:
    def decl_fn(a, *args):
        a.argtypes = args[:-1]
        a.restype = args[-1]
    def void_fn(a, *args):
        decl_fn(a, *(args+(None,)))
    def int_fn(a, *args):
        decl_fn(a, *(args+(c_int,)))
    def dbl_fn(a, *args):
        decl_fn(a, *(args+(c_double,)))
    def arr_t(*shape):
       return reduce(lambda x,y: x*y, reversed(shape), c_double)
    nparr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    dblarr = POINTER(c_double)
    vecarr = POINTER(c_double)
    vec = arr_t(3)

    def type_list(type, *names):
      return [(a,type) for a in names]

    def mkstruct(*args):
      def concat(y):
        return reduce(lambda x,y:x+y, y, [])
      def type_em(args):
        return type_list(*args)
      return concat(map(type_em, args))

    def nparr_t(*shape):
        return np.ctypeslib.ndpointer(dtype=np.float64, shape=shape, flags='C_CONTIGUOUS')
    def np_intarr_t(*shape):
        return np.ctypeslib.ndpointer(dtype=np.int32, shape=shape, flags='C_CONTIGUOUS')

    sfac_p = c_void_p
    rad_fn = PYFUNCTYPE(c_double, c_double, POINTER(c_double), c_void_p)
    
    # load library
    cwd = os.path.dirname(os.path.abspath(__file__))
    sfac = CDLL(os.path.join(cwd,"libsfac.so"))

    #int_fn(sfac.sfac_ctor, sfac_p, nparr_t(6), np_intarr_t(3), c_int)
    decl_fn(sfac.sfac_ctor, nparr_t(6), np_intarr_t(3), c_int, sfac_p)

    # functions in sfac's C interface (pass sfac_p as first arg.)
    void_fn(sfac.sfac_dtor, sfac_p)
    void_fn(sfac.sfac, sfac_p, c_int, nparr, nparr)
    decl_fn(sfac.get_S, sfac_p, dblarr)
    void_fn(sfac.set_L, sfac_p, nparr_t(6))
    int_fn(sfac.set_A, sfac_p, rad_fn, c_double, c_void_p)
    dbl_fn(sfac.en, sfac_p)
    dbl_fn(sfac.de1, sfac_p, nparr_t(6))
    void_fn(sfac.de2, sfac_p, c_int, nparr, nparr, nparr)
    void_fn(sfac.potl, sfac_p, c_int, nparr, nparr)
    return sfac_p, rad_fn, sfac

class Sfac:
    def __init__(self, L, K, order=4):
        sfac_p, rad_fn, libsfac = describe_sfac()
        self._libsfac = libsfac
        self.rad_fn = rad_fn
        self.K = tuple(K.tolist())
        self.ldim = K[2]//2+1

        self._data = libsfac.sfac_ctor(L.astype(np.float64),
                                       K.astype(np.int32), order)
        if not self._data:
            raise RuntimeError("Error intializing Sfac.")

    def __getattr__(self, name):
        """ Pass the class pointer as the first argument to the
            following functions.
        """
        names = ["sfac_dtor", "en", "de1", "de2", "potl"]
        if name not in names:
            raise AttributeError()
            #return None
        try:
            f = getattr(self._libsfac, name)
        except AttributeError:
            return None
        return lambda *args, **kws: f(self._data, *args, **kws)

    def get_S(self):
        Q = self._libsfac.get_S(self._data)
        shape = (self.K[0], self.K[1], self.ldim, 2)
        s = ct.as_array(Q, shape=shape)
        return s[:,:,:,0] + 1j*s[:,:,:,1]

    def sfac(self, w, x):
        assert len(w.shape) == 1
        assert x.shape == (len(w), 3)
        self._libsfac.sfac(self._data, len(w), w, x)

    def __del__(self):
        self.sfac_dtor()

    def set_A(self, f, max_m):
        self.f = f

        n = self._libsfac.set_A(self._data, self.rad_fn(f), max_m,
                                cast(0, c_void_p))
        if n != 0:
            raise RuntimeError("Error (%d) setting A-array."%n)

    def set_L(self, L):
        self._libsfac.set_L(self._data, L)
        if hasattr(self, "f"):
            self.set_A(self.f, 1.0)

eta = 0.5
_fac = -(pi/eta)**2
def ewald_f(r2, ptr, info):
    if r2 < 1e-10:
        ptr.contents.value = 0.0
        return 0.0
    f = exp(r2*_fac)/(r2*pi)
    ptr.contents.value = (_fac - 1./r2)*f
    return f

def LofS(L):
    return np.array([[L[0], 0.  , 0.  ],
                     [L[3], L[1], 0.  ],
                     [L[4], L[5], L[2]]
                    ])
def SofL(L):
    return np.array([L[0,0], L[1,1], L[2,2], L[1,0], L[2,0], L[2,1]])

