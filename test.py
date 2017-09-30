#!/usr/bin/env python

import os
from ewald import Sfac, LofS, SofL, ewald_f, eta
import numpy as np
from math import pi, exp, erfc

random = np.random.random
np.random.seed(100)
array = np.array

def test_lines(M, Q, crds):
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for q,x in zip(Q, crds):
        sx += q*np.exp(-2j*pi*x[0]*np.arange(M.shape[0]))
        sy += q*np.exp(-2j*pi*x[1]*np.arange(M.shape[1]))
        sz += q*np.exp(-2j*pi*x[2]*np.arange(M.shape[2]))
    print M[0,:,0]
    print sy
    #print "Line test:"
    print np.abs(M[:,0,0] - sx).max(), \
          np.abs(M[0,:,0] - sy).max(), \
          np.abs(M[0,0] - sz).max()

def test():
    L = np.array([9., 10., 11., 2., -0.1, 1.0])
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

    s = Sfac(L, np.array([12,13,14], dtype=np.int), 4)
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
