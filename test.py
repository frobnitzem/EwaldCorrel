#!/usr/bin/env python

import os
import numpy as np
from ctypes import CDLL, c_void_p, c_int, c_double, POINTER, \
                   Structure, PYFUNCTYPE, byref, addressof, cast
import numpy.ctypeslib as ct
from math import pi, exp, erfc
random = np.random.random
np.random.seed(100)
array = np.array

def load_sfac():
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

    class sfac_t(Structure):
          _fields_ = mkstruct(
              (c_int, "order"),
              (c_int*3, "K"),
              (c_int, "ldim"),
              (arr_t(3,3), "L", "iL"),
              (c_double, "iV", "iK"),
              (dblarr, "bspc"),
              (dblarr*3, "iB"),
              (c_void_p, "fft_neg", "fft_pos"),
              (dblarr, "A", "dA", "Q"))
    sfac_p = POINTER(sfac_t)
    rad_fn = PYFUNCTYPE(c_double, c_double, POINTER(c_double), c_void_p)
    
    # load library
    cwd = os.path.dirname(os.path.abspath(__file__))
    sfac = CDLL(os.path.join(cwd,"libsfac.so"))

    int_fn(sfac.sfac_ctor, sfac_p, nparr_t(6), np_intarr_t(3), c_int)
    int_fn(sfac.sfac_dtor, sfac_p)
    int_fn(sfac.set_A, sfac_p, rad_fn, c_void_p)
    void_fn(sfac.set_L, sfac_p, nparr_t(6))
    void_fn(sfac.sfac, sfac_p, c_int, nparr, nparr)
    dbl_fn(sfac.en, sfac_p)
    dbl_fn(sfac.de1, sfac_p, nparr_t(6))
    void_fn(sfac.de2, sfac_p, c_int, nparr, nparr, nparr)
    return sfac_t, rad_fn, sfac

class Sfac:
    def __init__(self, L, K, order=4):
        sfac_t, rad_fn, sfac = load_sfac()
        self._sfac = sfac
        self.rad_fn = rad_fn
        self.data = sfac_t()
        self.K = tuple(K.tolist())

        self._s = cast(addressof(self.data), POINTER(sfac_t))

        n = self.sfac_ctor(L, K.astype(np.int32), order)
        if n != 0:
            raise RuntimeError, "Error (%d) initializing Sfac."%n

    def __getattr__(self, name):
        names = ["sfac_ctor", "sfac_dtor",
                 "sfac", "en", "de1", "de2"]
        if name not in names:
            raise AttributeError
            #return None
        try:
            f = getattr(self._sfac, name)
        except AttributeError:
            return None
        return lambda *args, **kws: f(self._s, *args, **kws)

    def S(self):
        shape = (self.K[0], self.K[1], self.data.ldim, 2)
        s = ct.as_array(self.data.Q, shape=shape)
        return s[:,:,:,0] + 1j*s[:,:,:,1]

    def __del__(self):
        self.sfac_dtor()

    def set_A(self, f):
        self.f = f

        n = self._sfac.set_A(self._s, self.rad_fn(f), cast(0, c_void_p))
        if n != 0:
            raise RuntimeError, "Error (%d) setting A-array."%n

    def set_L(self, L):
        self._sfac.set_L(self._s, L)
        if hasattr(self, "f"):
            self.set_A(self.f)

eta = 0.5
_fac = -(pi/eta)**2
def ewald_f(r2, ptr, info):
    if r2 < 1e-10:
        ptr.contents.value = 0.0
        return 0.0
    f = exp(r2*_fac)/r2
    ptr.contents.value = (_fac - 1./r2)*f
    return f

def LofS(L):
    return np.array([[L[0], 0.  , 0.  ],
                     [L[3], L[1], 0.  ],
                     [L[4], L[5], L[2]]
                    ])
def SofL(L):
    return np.array([L[0,0], L[1,1], L[2,2], L[1,0], L[2,0], L[2,1]])

def test_lines(M, Q, crds):
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for q,x in zip(Q, crds):
        sx += q*np.exp(-2j*pi*x[0]*np.arange(M.shape[0]))
        sy += q*np.exp(-2j*pi*x[1]*np.arange(M.shape[1]))
        sz += q*np.exp(-2j*pi*x[2]*np.arange(M.shape[2]))
    #print M[0,0]
    #print sz
    #print "Line test:"
    print np.abs(M[:,0,0] - sx).max(), \
          np.abs(M[0,:,0] - sy).max(), \
          np.abs(M[0,0] - sz).max()

def test():
    L = np.array([10., 10., 10., 2., -0.1, 1.0])
    #L = np.array([9.,9.,9.,0.,0.,0.])
    V = L[0]*L[1]*L[2]
    #N = 4
    #crds = random((N,3))
    crds = np.array([[0.1, 0.5, 0.1],
                     [0.9, 0.4, 0.1],
                     [0.1, 0.4, 0.9],
                     [0.9, 0.5, 0.9]])
    #crds = np.array([[0.1, 0.5, 0.9],
    #                 [0.7, 0.0, 0.3]])
    N = len(crds)
    atoms = np.dot(crds, LofS(L))
    dx = np.zeros((N,3))
    vir = np.zeros(6)
    #q = random(N)-0.5
    q = np.ones(N)

    Ecorr = np.sum(q*q)*(-eta/np.sqrt(pi))

    s = Sfac(L, np.array([10,10,10], dtype=np.int), 4)
    s.set_A(ewald_f)
    s.sfac(N, q, atoms)
    test_lines(s.S(), q, crds)

    def Ex(x):
        s.sfac(N, q, x)
        return s.en()
    dx0 = num_diff(Ex, atoms)

    s.sfac(N, q, atoms)
    print s.en()
    print s.de1(vir)
    s.de2(N, q, atoms, dx)
    print dx
    print dx0

    print vir
    def E(L):
        s.set_L(L)
        #s.sfac(N, q, np.dot(crds, LofS(L)))
        return s.en()
    s.sfac(N, q, atoms)
    Pi = LofS(num_diff(E, L)).transpose()
    # Pi is properly symmetric, but we're only computing the upper-diagonal.
    Pi = SofL(np.dot(Pi, LofS(L)).transpose())/(-V)
    print Pi

    atoms *= 0.0
    #for zi in np.arange(200)*0.1 - 10.05:
    for zi in []:
        atoms[1,2] = zi
        s.sfac(N, q, atoms)
        en = s.de1(vir)
        s.de2(N, q, atoms, dx)

        r1 = np.sqrt(np.sum((atoms[1]-atoms[0])**2))
        r2 = np.sqrt(np.sum((atoms[1]-atoms[0]-array([0., 0., 10.]))**2))
        r3 = np.sqrt(np.sum((atoms[1]-atoms[0]+array([0., 0., 10.]))**2))

        #Ereal = 0.5*( erfc(eta*r1)/r1 + erfc(eta*r2)/r2 + erfc(eta*r3)/r3 )
        en += Ecorr
        print zi, en, dx[0,2], dx[1,2], dx[1,0], dx[1,1]

def num_diff(f, x, h = 1e-7):
    ih = 1./(2.0*h)
    h = 0.5/ih

    s = x.shape
    x = np.reshape(x, -1)
    df = np.zeros(len(x))
    for i in range(len(x)):
        x0 = x[i]
        x[i] = x0 + h
        df[i] = f(np.reshape(x, s))
        x[i] = x0 - h
        df[i] -= f(np.reshape(x, s))
        x[i] = x0

    f0 = f(x)

    return np.reshape(df*ih, s)

test()
