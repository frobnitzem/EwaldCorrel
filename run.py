import os, sys
from ewald import Sfac, LofS, SofL, ewald_f, eta
from math import pi, exp, erfc
import numpy as np
from time import time

bfac = {
  'H': -3.7390,
  'C':  6.6460,
  'O':  5.803,
}

def read_mol(fname):
    names = []
    crd = []
    with open(fname, "r") as f:
        lines = f.read().split("\n")
        for line in lines[2:]:
            tok = line.split()
            if len(tok) >= 4:
                names.append(tok[0])
                crd.append(list(map(float, tok[1:4])))

    return names, np.array(crd)

def run(inp, out):
    # Define the molecule.
    names, crd = read_mol(inp)
    wts = np.array([bfac[name] for name in names])
    L = 11.680933*5
    #58.404665

    K = 128
    s = Sfac(np.array([L,L,L,0,0,0]), np.array([K, K, K], dtype=int), 4)
    #s.set_A(ewald_f)
    Q = np.zeros((K, K, s.ldim), np.complex128)
    S = 0

    for frame in [crd]:
        for i in range(3):
            t0 = time()
            s.sfac(wts, crd)
            t1 = time()
            print(f"{(t1-t0)*1000} ms")
        Q += np.abs(s.get_S())**2

    np.save(out, Q)

if __name__=="__main__":
    import sys
    assert len(sys.argv) == 3, "Usage: %s <inp.xyz> <out.npy>"%sys.argv[0]
    run(sys.argv[1], sys.argv[2])
