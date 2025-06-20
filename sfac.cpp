/*  Structure Factor Computation Code
 *  
 *  ... from a more civilized era.
 */

//#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <math.h>

#include "sfac.hpp"
#include <stdexcept>

/*  Calculates the (one over the) DFT of the B-spline coefficients.
    The thing is done by the slow-but-simple matrix multiply technique.
 */
static void bspl_dft(int n, int order,
                     std::vector<double> &dft_c, const double *c) {
    int i,j;
    const int j0 = order/2;

    for(i=0; i<n; i++) {
        double om = -(2.0*M_PI*i)/n;
        double x = 0.0;

        for(j=1; j<order; j++) {
            x += cos(om*(j-j0))*c[order*j];
        }
        dft_c[i] = 1.0/x;
    }
}

// fully initializes pbc
// L is a_x, b_y, c_z, b_x, c_x, c_y
SFac::SFac(double L[6], int32_t K_[3], int order_)
        : order(order_), cell(L), bspc(BSpline(order_)) {
    if(order%2 != 0) {
        printf("Bad order: %d (must be even)\n", order);
        throw std::runtime_error("invalid order");
    }
    K[0] = K_[0];
    K[1] = K_[1];
    K[2] = K_[2];
    if(K[0] < 1 || K[1] < 1 || K[2] < 1) {
        printf("Bad K: %d %d %d\n", K[0], K[1], K[2]);
        throw std::runtime_error("invalid K");
    }
    ldim = K[2]/2+1; // Add 1 to odd K and 2 to even K
                     // to store the cplx part of freq=0 [odd]
                     // or the cplx part of freq=0 and K/2 [even].
    iK = 1.0/(K[0]*K[1]*K[2]);

        /* Precomputed B magnitudes. */
    iB.push_back(std::vector<double>(K[0]));
    iB.push_back(std::vector<double>(K[1]));
    iB.push_back(std::vector<double>(K[2]));
    int n,i,j;

    set_L(reinterpret_cast<void *>(this), L);

    for(j=0; j<3; j++) {
        // Last row of bspc are spline values at integer nodes.
        bspl_dft(K[j], order, iB[j], &bspc.M[order-1]);
    }

    n = 2*ldim*K[1]*K[0]; // Padded array size.
    Q = (double *)fftw_malloc(sizeof(double)*n);
    if(Q == NULL) {
        throw std::runtime_error("Q = fftw_malloc()");
    }

    A = dA = NULL;

    fft_neg = fftw_plan_dft_r2c_3d(K[0], K[1], K[2],
            Q, FQ, FFTW_PATIENT);
    fft_pos = fftw_plan_dft_c2r_3d(K[0], K[1], K[2],
            FQ, Q, FFTW_PATIENT);
}

/*  Iterate through these using:
 *
 *  std::pair<auto, auto> range = srt.equal_range(0);
 *  for (auto it = range.first; it != range.second; ++it) {
 *      // cid = it->first
 *      // u = it->second
 *  }
 */
std::multimap<int, Vec4>
SFac::sort(int n, const double *w, const double *x) const {
    typedef std::multimap<int, Vec4> MMap;
    MMap srt;

    for(int a=0; a<n; a++) {
        Vec4 u = cell.scale(x+3*a); // wrap x into the scaled unit cell
        int n[3];
        for(int i=0; i<3; i++) { // scale up to integer indexing
            u[i] = u[i]*K[i];
            n[i] = (int)u[i];
            u[i] -= n[i];
        }
        u[3] = w[a]; // associate coordinate weight
        int cid = (n[0]*K[1] + n[1])*K[2] + n[2];
        srt.insert(MMap::value_type(cid, u));
    }

    return srt;
}

void SFac::operator()(int n, const double *w, const double *x) {
    memset(Q, 0.0, sizeof(double)*K[0]*K[1]*ldim*2);
    std::multimap<int, Vec4> srt = sort(n, w, x);

#pragma omp parallel for collapse(3)
    for(int i=0; i<K[0]; i++) {
        for(int j=0; j<K[1]; j++) {
            for(int k=0; k<K[2]; k++) {
                accum_spline(i, j, k, srt);
            }
        }
    }
    fftw_execute(fft_neg); // FQ = F-(Q)
    state = 0;

    // Divide FQ by the FFT of the B-spline smoothing operation.
    // sets FQ to FQ/F(B)
#pragma omp parallel for
    for(int cid=0; cid < K[0]*K[1]*ldim; cid++) {
        double *Qi = Q + 2*cid;
        double t = iB[0][ cid / (ldim*K[1]) ]
                 * iB[1][ (cid/ldim) % K[1] ]
                 * iB[2][ cid % ldim ];

        Qi[0] *= t; Qi[1] *= t;
    }
}

void SFac::accum_spline(int i1, int j1, int k1,
                        const std::multimap<int, Vec4> &srt) {
    int NN = (i1*K[1] + j1)*ldim*2 + k1;
    int i0 = i1 + K[0] - order/2;
    int j0 = j1 + K[1] - order/2;
    int k0 = k1 + K[2] - order/2;
    double ans = 0.0;

    for(int i=0; i<order; i++) {
        int xid = ( (i0+i)%K[0] ) * K[1];
        for(int j=0; j<order; j++) {
            int yid = ( xid + (j0+j)%K[1] ) * K[2];
            for(int k=0; k<order; k++) {
                int cid = yid + (k0+k)%K[2];

                auto range = srt.equal_range(cid);
                for (auto it = range.first; it != range.second; ++it) {
                    auto u = it->second; // Vec4
                    ans += bspc.bspl_coef(u[0], i)
                         * bspc.bspl_coef(u[1], j)
                         * bspc.bspl_coef(u[2], k)
                         * u.w;
                }
            }
        }
    }
    Q[NN] += ans;
}

SFac::~SFac() {
    fftw_destroy_plan(fft_neg);
    fftw_destroy_plan(fft_pos);
    fftw_free(Q);
    if(A != NULL) free(A);
    fftw_cleanup();
}

extern "C" {

void *sfac_ctor(double L[6], int K[3], int order) {
    SFac *ret = new SFac(L, K, order);
    return reinterpret_cast<void *>(ret);
}

void sfac_dtor(void *sfac) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    delete pbc;
}

void sfac(void *sfac, int n, const double *w, const double *x) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    (*pbc)(n, w, x);
}
double *get_S(void *sfac) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    return pbc->Q;
}

void set_L(void *sfac, double L[6]) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    pbc->cell = Cell(L);
}

/* Set value of A[m] for all |m| < max_m
 *
 * Note: We must track all 6 virial indices because
 *       multiple m-points could contribute at the same lattice n.
 *
 * This could potentially be parallelized by
 * visiting every m, then
 * summing over all periodic images of m within max_m.
 */
int set_A(void *sfac, rad_fn f, double max_m, void *info) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    int n[3];
    const double R2 = SQR(max_m);
    const int i0 = (int)(max_m/pbc->cell.iL[0][0]);

    int i = pbc->K[0]*pbc->K[1]*pbc->ldim;
    // Factor of 7 accounts for potential and 6 virial components.
    if(pbc->A == NULL) {
        pbc->A = (double *)malloc(sizeof(double)*i*7);
        if(pbc->A == NULL) return 2;
        pbc->dA = pbc->A + i;
    }
    memset(pbc->A, 0.0, sizeof(double)*i*7);

    // L = [ L00,   0, 0
    //       L01, L11, 0
    //       L02, L12, L22
    //     ]
    // iL = [ iL00, 0, 0
    //        iL10, iL11, 0
    //        iL20, iL21, iL22
    //      ]
    // mx = iL00 * i
    // my = iL10 * i + iL11 * j
    // mz = iL20 * i + iL21 * j + iL22 * k

    for(i=-i0; i<=i0; i++) {
        const double mx = i*pbc->cell.iL[0][0];
        const double m20 = SQR(mx);
        const double dy = sqrt(R2 - m20);
        const double my0 = i*pbc->cell.iL[1][0];
        // |my0 + iL11*j| < dy
        const int j0 = ceil((-dy - my0)/pbc->cell.iL[1][1]);
        const int j1 = floor((dy - my0)/pbc->cell.iL[1][1]);
        int j;

        n[0] = MOD(i,pbc->K[0])*pbc->K[1];
        for(j=j0; j<=j1; j++) {
            const double my = my0 + j*pbc->cell.iL[1][1];
            const double m21 = m20 + SQR(my);
            const double dz = sqrt(R2 - m21);
            const double mz0 = i*pbc->cell.iL[2][0] + j*pbc->cell.iL[2][1];
            // |mz0 + iL22*k| < dz
            const int k0 = ceil((-dz - mz0)/pbc->cell.iL[2][2]);
            const int k1 = floor((dz - mz0)/pbc->cell.iL[2][2]);
            int k;

            n[1] = (MOD(j,pbc->K[1])+n[0])*pbc->ldim;
            for(k=k0; k<=k1; k++) {
                const double mz = mz0 + k*pbc->cell.iL[2][2];
                const double m2 = m21 + SQR(mz);
                double *dA, s;

                n[2] = MOD(k, pbc->K[2]);
                if(n[2] >= pbc->ldim) {
                    n[2] = pbc->K[2] - n[2];
                    n[2] += n[0] == 0 ? 0 :
                            (pbc->K[0]*pbc->K[1] - n[0])*pbc->ldim;
                    n[2] += j % pbc->K[1] == 0 ? 0 :
                            (pbc->K[1]+n[0])*pbc->ldim - n[1];
                } else {
                    n[2] += n[1];
                }
                pbc->A[n[2]] += 0.5*f(m2, &s, info);

                dA = pbc->dA + 6*n[2];

                dA[0] += s*mx*mx;
                dA[1] += s*my*my;
                dA[2] += s*mz*mz;
                dA[3] += s*my*mx;
                dA[4] += s*mz*mx;
                dA[5] += s*mz*my;
            }
        }
    }

    return 0;
}

/*
static inline void cplx_mul(double *res, const double *a, const double *b) {
    res[0] = a[0]*b[0] - a[1]*b[1];
    res[1] = a[0]*b[1] + a[1]*b[0];
}*/

/* Stuff you can do with the spatial FT */

// Calculates the convolution, sum_kj Q_k F-[A]_{k-j} Q_j
// using the Fourier-space A, FQ
double en(void *sfac) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    int i;
    const int n = pbc->K[0]*pbc->K[1]*pbc->ldim;
    double en = 0.0;

    if(pbc->state != 0) {
        printf("Error: You must call sfac before this routine.\n");
        return 0.0;
    }
    if(pbc->A == NULL) {
        printf("Error: No convolution kernel has been set.\n");
        return 0.0;
    }
    #pragma omp parallel for reduction(+:en)
    for(i=0; i<n; i++) {
        double t = SQR(pbc->FQ[i][0])+SQR(pbc->FQ[i][1]);
        en += pbc->A[i]*t;
    }
    
    return en*pbc->cell.iV;
}

/* Note: for all these, A[z = 0] and A[z = n/2] (when n is even)
 * must be doubled so that the single-loop will work.
 *
 * This is done by the set_A routine.
 *
 * Carry out a convolution with 'A' and return derivative of
 * sum_k A_k |Q|_k^2 wrt the lattice vectors.
 *    Input:    Reciprocal space multipliers, A and their radial derivatives, dA
 *              It is assumed these have mirror symmetry in x,y,z (hence real).
 *    Output:   Returns energy
 *              FQ is replaced with A FQ = F+[F-[A] * Q]
 */
double de1(void *sfac, double vir[6]) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    int i;
    const int n = pbc->K[0]*pbc->K[1]*pbc->ldim;
    const int even = (pbc->K[2]+1)%2;
    mm_complex *FQ = pbc->FQ;
    double en = 0.0;
    double vir00,vir11,vir22,vir10,vir20,vir21;

    if(pbc->state != 0) {
        printf("Error: You must call sfac before this routine.\n");
        return 0.0;
    }
    if(pbc->A == NULL) {
        printf("Error: No convolution kernel has been set.\n");
        return 0.0;
    }

    /* Overwrite FQ with A FQ and sum en. */
    vir00 = vir11 = vir22 = 0.0;
    vir10 = vir20 = vir21 = 0.0;
#pragma omp parallel for reduction(+:en) reduction(+:vir00) reduction(+:vir11) reduction(+:vir22) reduction(+:vir10) reduction(+:vir20) reduction(+:vir21)
    for(i=0; i<n; i++) {
        double s = SQR(FQ[i][0]) + SQR(FQ[i][1]);
        double *dA = pbc->dA + i*6;
        double t = pbc->A[i]
                 * pbc->iB[0][ i/(pbc->ldim*pbc->K[1]) ]
                 * pbc->iB[1][ (i/pbc->ldim) % pbc->K[1] ]
                 * pbc->iB[2][ i% pbc->ldim ];

        if(i % pbc->ldim == 0
            || even && (i+1)%pbc->ldim == 0) t *= 2.0;
        FQ[i][0] *= t;
        FQ[i][1] *= t;

        en += s*pbc->A[i];

        vir00 += s*dA[0];
        vir11 += s*dA[1];
        vir22 += s*dA[2];
        vir10 += s*dA[3];
        vir20 += s*dA[4];
        vir21 += s*dA[5];
    } // Now: FQ := A FQ

    double fac = SQR(pbc->cell.iV);
    vir[0] = (vir00 + en)*fac;
    vir[1] = (vir11 + en)*fac;
    vir[2] = (vir22 + en)*fac;
    vir[3] = vir10*fac;
    vir[4] = vir20*fac;
    vir[5] = vir21*fac;

    fftw_execute(pbc->fft_pos); // Q = V F+[A F-(Q)]
    pbc->state = 1;

    return en * pbc->cell.iV;
}

/* Calculate the change in en wrt. input positions, x.
 * For PME, this routine calculates the negative electric field times w.
 *
 * Note that w and x sent to this routine do not need to be the
 * same ones used to compute FQ.
 *
 *    Input:    Original reciprocal space multipliers, A and coordinates.
 *              It is assumed de1 has already been called
 *              in order to scale FQ!
 *    Output:    Energy derivatives, dE/dx.
 */
void de2(void *sfac, int n, const double *w, const double *x, double *dx0) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    //double en = 0.0;
    int a;

    if(pbc->state != 1) {
        printf("Error: You must call de1 before this routine.\n");
        return;
    }

// To enable energy re-calculation, uncomment the next line:
//#pragma omp parallel for reduction(+:en)
    for(a=0; a<n; a++) {
        int i, j, k;
        int k0[3];    // Start of relevant k values
        int k1[3];    // Start of relevant k values
        double dx[3] = {0., 0., 0.}; // sum_l C(l) dQ(l)/ds

        // Calculate scaled particle position from s = L^{-T} n
        Vec4 u = pbc->cell.scale(x+3*a);
        for(int ii=0; ii<3; ii++) {
            u[ii] *= pbc->K[ii];
        }
        k1[0] = (int)ceil(u[0]);
        k1[1] = (int)ceil(u[1]);
        k1[2] = (int)ceil(u[2]);

        // make sure to index positive values
        k0[0] = MOD(k1[0] - pbc->order/2, pbc->K[0]);
        k0[1] = MOD(k1[1] - pbc->order/2, pbc->K[1]);
        k0[2] = MOD(k1[2] - pbc->order/2, pbc->K[2]);

        //printf("dAtom %d: u = %.2f %.2f %.2f, k0 = %d %d %d\n", a+1,
        //        u[0],u[1],u[2], k0[0],k0[1],k0[2]);
        // Multiply values and get forces from a block of the array.
        for(i=0; i<pbc->order; i++) {
            int ni = ((k0[0]+i) % pbc->K[0]) * pbc->K[1]; // Dimensionality...
            double mpx, dmpx;
            pbc->bspc.dbspl_coef(&mpx, &dmpx, k1[0] - u[0], i);

            for(j=0; j<pbc->order; j++) {
                double mpy, dmpy;
                pbc->bspc.dbspl_coef(&mpy, &dmpy, k1[1] - u[1], j);
                double fj0 = dmpx *  mpy;
                double fj1 =  mpx * dmpy;
                double fj2 =  mpx *  mpy;
                int nj = (ni + (k0[1]+j) % pbc->K[1])*pbc->ldim*2; // summing...
                for(k=0; k<pbc->order; k++) {
                    double mpz, dmpz;
                    pbc->bspc.dbspl_coef(&mpz, &dmpz, k1[2] - u[2], k);
                    int nk = nj + (k0[2]+k) % pbc->K[2]; // to final n.
                    double fk = mpz*pbc->Q[nk];
                    dx[0] += fj0 * fk;
                    dx[1] += fj1 * fk;
                    dx[2] += fj2 * dmpz*pbc->Q[nk];
                    // To enable energy re-calculation, uncomment
                    // the next line:
                    //en += w[a]*fj2*fk;
                    // (i.e. = w[a] * mpx *  mpy * mpz * pbc->Q[nk] )
                }
            }
        }
        dx[0] *= (-w[a]*pbc->cell.iV)*pbc->K[0];
        dx[1] *= (-w[a]*pbc->cell.iV)*pbc->K[1];
        dx[2] *= (-w[a]*pbc->cell.iV)*pbc->K[2];
        // dE(r)/ dr_i = dE(s)/ds_j * ds_j/dr_i (s_j = iL_{ij} r_i)
        dx0[3*a+0] = pbc->cell.iL[0][0]*dx[0];
        dx0[3*a+1] = pbc->cell.iL[1][0]*dx[0] + pbc->cell.iL[1][1]*dx[1];
        dx0[3*a+2] = pbc->cell.iL[2][0]*dx[0] + pbc->cell.iL[2][1]*dx[1]
                                              + pbc->cell.iL[2][2]*dx[2];
    }
    // To enable energy re-calculation, uncomment the next line:
    //en *= 0.5*pbc->cell.iV;
    //printf("E = %f\n", en);
}

/* Calculate the change in en wrt. input weights, w.
 * In a PME calculation, the weights are charges, so this
 * derivative is the potential.
 *
 * Note that n and x-crds sent to this routine do not need to be the
 * same ones used to compute FQ.
 *
 *    Input:    Original reciprocal space multipliers,  and coordinates.
 *              It is assumed de1 has already been called
 *              in order to scale FQ!
 *    Output:    Energy derivatives, dE/dw.
 */
void potl(void *sfac, int n, const double *x, double *phi) {
    SFac *pbc = reinterpret_cast<SFac *>(sfac);
    double en = 0.0;
    int a;

    if(pbc->state != 1) {
        printf("Error: You must call de1 before this routine.\n");
        return;
    }

    for(a=0; a<n; a++) {
        int i, j, k;
        int k0[3];    // Start of relevant k values
        int k1[3];    // Start of relevant k values
        double pot_a = 0.0; // summation result

        // Calculate scaled particle position from s = L^{-T} n
        Vec4 u = pbc->cell.scale(x+3*a);
        for(int ii=0; ii<3; ii++) {
            u[ii] *= pbc->K[ii];
        }
        k1[0] = (int)ceil(u[0]);
        k1[1] = (int)ceil(u[1]);
        k1[2] = (int)ceil(u[2]);

        // make sure to index positive values
        k0[0] = MOD(k1[0] - pbc->order/2, pbc->K[0]);
        k0[1] = MOD(k1[1] - pbc->order/2, pbc->K[1]);
        k0[2] = MOD(k1[2] - pbc->order/2, pbc->K[2]);

        // Multiply values and get potential from a block of the array.
        for(i=0; i<pbc->order; i++) {
            int ni = ((k0[0]+i) % pbc->K[0]) * pbc->K[1];
            double s1 = 0.0; // smaller piece of summation
            for(j=0; j<pbc->order; j++) {
                double mpy = pbc->bspc.bspl_coef(k1[1] - u[1], j);
                int nj = (ni + (k0[1]+j) % pbc->K[1])*pbc->ldim*2;
                for(k=0; k<pbc->order; k++) {
                    double mpz = pbc->bspc.bspl_coef(k1[2] - u[2], k);
                    int nk = nj + (k0[2]+k) % pbc->K[2];
                    s1 += mpy*mpz*pbc->Q[nk];
                }
            }
            double mpx = pbc->bspc.bspl_coef(k1[0] - u[0], i);
            pot_a += mpx*s1;
        }
        phi[a] = pot_a * pbc->cell.iV;
    }
}
}
