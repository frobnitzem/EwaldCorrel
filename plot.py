import numpy as np

def kvec(n, L):
    x = np.arange(n)
    half = n//2
    x[half:] -= n
    return x*np.pi*2/L

def main(argv):
    f = np.load(argv[1]).real

    L = 11.680933*5
    kmax = 0.25*len(f)*np.pi/L
    dk = kmax/5000

    x = kvec(f.shape[0], L)
    y = kvec(f.shape[1], L)
    z = kvec(f.shape[2], L)

    dist = (((x*x)[:,None,None] +  (y*y)[None,:,None] +  (z*z)[None,None,:]
            )**0.5 / dk
           ).clip(0, kmax/dk+1).astype(int)
    dist = dist.ravel()

    counts = np.bincount(dist)
    values = np.bincount(dist, weights=f.ravel())

    for i, n in enumerate(counts[:-1]):
        if i == 0 or n == 0:
            continue
        k = i*dk
        print(f"{k} {values[i]/n} {n}")

if __name__=="__main__":
    import sys
    main(sys.argv)
