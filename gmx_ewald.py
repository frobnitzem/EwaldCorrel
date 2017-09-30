#!/usr/bin/env python2.7
#
# Works on systems with 'ions' number of Na+ followed by the same number
# of Cl-, followed by an arbitrary number of solvent molecules.
#
# sys.py should contain (e.g. for 20 ions in spc/e):
#   ions = 20
#   mass = array( [15.9994, 1.008, 1.008] )
#   chg  = array( [-0.8476, 0.4238, 0.4238] )
#  

import os, sys, imp
#sys.path.append("/Users/rogers/work/traj_an/src/gtool")

from ewald import Sfac, LofS, SofL, ewald_f, eta
from math import pi, exp, erfc
#from trj_proto import trr
from xdrfile import xdrfile
from numpy import *

Cfac = 4.184*332.0716 # 96.4853 * 14.4 eV-A

# Read molecule information and create weight-calculator.
def mk_calc(name):
    mols = 1
    vecs = 1

    mod = imp.load_source("mod", name)
    ions = 0
    if hasattr(mod, 'ions'):
        ions = mod.ions
        mols += 2 # Na+ followed by Cl-
    m = mod.mass/sum(mod.mass)
    q = mod.chg
    M = len(q)
    assert len(q) == len(m)

    D = mols + 3*vecs # number of matrix components
    w1 = ones(ions) # ion weights

    # calculate per-molecule weights
    def calc_dof(x, L):
        # first, wrap all molecules
        y = x[2*ions:].reshape((-1,M,3))
        for i in range(1, M):
            y[:,i] -= L[2]*floor((y[:,i,2]-y[:,0,2])/L[2,2]+0.5)[:,newaxis]
            y[:,i] -= L[1]*floor((y[:,i,1]-y[:,0,1])/L[1,1]+0.5)[:,newaxis]
            y[:,i] -= L[0]*floor((y[:,i,0]-y[:,0,0])/L[0,0]+0.5)[:,newaxis]
        
        # next, calculate centers of mass
        cm = [x[:ions], x[ions:2*ions], tensordot(y, m, axes=[1,0])]
        cm += [cm[2]]*3 # all vectors at same location
        y -= cm[2][:,newaxis,:] 

        # finally, calculate vectors
        P  = tensordot(transpose(y, (2,0,1)), q, axes=[2,0])
        w  = [ w1, w1, ones(len(y)), P[0], P[1], P[2] ]
        return w[-D:], cm[-D:]
    return calc_dof, D

def run(name):
    # Define the molecule.
    calc, D = mk_calc("sys.py")

    trj = xdrfile(name)
    #vir = zeros(6)
    #N = trj.natoms/M
    #dx = zeros((N, 3))

    g = trj.__iter__()
    frame = g.next()
    #trj.seek(0)

    print frame.box
    K = 128
    s = Sfac(SofL(frame.box), array([K, K, K], dtype=int), 4)
    s.set_A(ewald_f)
    Q = zeros((D, K, K, s.data.ldim), complex128)
    C = zeros((D*(D+1)/2, K, K, s.data.ldim), complex128)
    S = 0

    print >>sys.stderr, "Working",
    for frame in g:
        L = frame.box
        wts, cm = calc(frame.x, L)

        for i in range(D):
            s.sfac(len(cm[i]), wts[i], cm[i].astype(float64))
            Q[i] = s.S()
        k = 0
        for i in range(D):
            for j in range(i, D):
                C[k] += Q[i]*Q[j].conjugate()
                k += 1
        S += 1
        #s.de1(vir)
        #s.de2(N, q, x, dx)
        #vir *= Cfac
        #dx *= Cfac
        #print dx/q[:,newaxis]
        #print frame.f - dx
        if S % 100 == 0:
            print >>sys.stderr, '\b.',

    print >>sys.stderr, "\n"
    C /= float(S)
    save("correl.npy", C)

if __name__=="__main__":
    assert len(sys.argv) == 2, "Usage: %s <run.trr>"%sys.argv[0]
    run(sys.argv[1])

