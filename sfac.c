/*  Structure Factor Computation Code */

//#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <math.h>

#define MAX_SPL_ORDER (8)

#define SQR(x) ((x)*(x))
#define MOD(a,b) (((a)%(b) + (b)) % (b))
#define WRAP(a,b) (a > b/2 ? a-b : a)

typedef fftw_plan fft_plan_t;
typedef double mm_complex[2];

/* Structure factor info.
 * Z-axis is last in storage order (X-major).
 */
typedef struct {
	int order; // B-spline order
	int K[3]; // Size of grid.
	int ldim; // number of cplx numbers along last grid dimension
	double L[3][3]; // Box shape
	double iL[3][3]; // inverse box shape
        double iV;
	double iK; // inverse-K-volume
	double *bspc; // B-spline coefficients.
	double *iB[3]; // length 2K-arrays
	fft_plan_t fft_neg, fft_pos; // pbc->K[2]/2+1
        double *A; // K[0] x K[1] x ldim (cplx, 0 imag part)
        double *dA; // K[0] x K[1] x ldim x 6
        union {
            double *Q; // K[0] x K[1] x ldim*2 (K[2] used) (real)
            mm_complex *FQ; // K[0] x K[1] x ldim x 2 (cplx)
        };
} sfac_t;
// For the last dimension, there are K[2] real numbers,
// and K[2] frequencies, stored sequentially from 0.
// However only ldim of those frequencies are actually
// stored, since FQ[i,j,k] = FQ[-i,-j,-k]^*.

void sfac(sfac_t *pbc, int n, const double *w, const double *x);
double en(sfac_t *pbc);
double de1(sfac_t *pbc, double vir[6]);
void de2(sfac_t *pbc, int n, const double *w, const double *x, double *dx);

void bspl_coef(double *b, const double x, const int order, const double *M);
void dbspl_coef(double *b, double *db, const double x,
                const int order, const double *M);

/* pre-computations */
void mk_bspl_coeffs(double *M, int n);

// rad_fn must return (A, dA) as a function of r2 = R*R.
// 'info' is passed through.
typedef double (*rad_fn)(double r2, double *dA, void *info);

/*  Calculates the coefficients used in bspl_coef
	Input:	Order of interpolation
	Output:	Array of coefficients for bspl_coef,
                M(x-floor(x)+j) = sum_i M_ij (x-floor(x))**i
	
	Note: This function must be called before any spline thing is done.
		mk_bspl_coeffs(pbc->bspc, pbc->order)
 */
void mk_bspl_coeffs(double *M, const int n) {
    int i, j, k;
    double p, f, den;

    bzero((void *)M, n*n*sizeof(double));
    den = 1.0;
    for(k=2;k<n;k++) den *= k; // (n-1)!
    f = 1.0; // f/den = (-1)**k n / (k! (n-k)!)
    for(k=0; k<n; k++) {
        if(k > 0) {
            f *= k-n-1;
            den *= k;
        }
        for(j=k; j<n; j++) {
            p = f/den; // (j-k)**(n-1-i)
            for(i=0; i<n; i++) {
                M[i*n+j] += p;
                p *= j-k;
            }
        }
    }
    p = 1.0;
    for(i=1; i<n-1; i++) { // multiply by n-1 C i
        p *= (n-i)/ (double)i;
        for(j=0; j<n; j++)
            M[i*n+j] *= p;
    }
}

/*  Calculates the raw coefficient of the Euler Cardinal B-Spline.
	Input:	Point x
		Order of interpolation
		Array of coefficients for calculation
	Output:	Spline coefficients, M_n(x+j) for j=0,1,...,n-1 - assuming x in [0,1)
    Note: This routine fails for x<0, which should rightly return 0.
	It also produces innaccurate results for x>=order, which should also
	be 0.  However, since these circumstances are not found in this
	program, they are here ignored.
 */
void bspl_coef(double *b, const double x, const int n, const double *M) {
    int i, j;
    const double *m;

    for(j=0; j<n; j++)
        b[j] = M[j];
    for(i=1; i<n; i++) {
        m = M + i*n;
        for(j=0; j<n; j++)
            b[j] = b[j]*x + m[j];
    }
}

/*  Calculates the raw coefficient of the
    Euler Cardinal B-Spline and its derivative.
	Input:	Point x
		Order of interpolation
		Array of coefficients for calculation
	Output:	Spline coefficients, M_n(x+j) for j=0,1,...,n-1 - assuming x in [0,1)
		and derivatives, M'_n(x+j) for j=0,1,...,n-1 - assuming x in [0,1)
    Note: This routine fails for x<0, which should rightly return 0.
	It also produces innaccurate results for x>=order, which should also
	be 0.  However, since these circumstances are not found in this
	program, they are here ignored.
 */
/* Assumes n >= 1, as it should be for nonzero derivatives. */
void dbspl_coef(double *b, double *db,
        const double x, const int n, const double *M) {
    int i, j;
    const double *m;

    m = M + n;
    for(j=0; j<n; j++) {
        db[j] = M[j];
        b[j] = M[j]*x + m[j];
    }
    for(i=2; i<n; i++) {
        m = M + i*n;
        for(j=0; j<n; j++) {
            db[j] = db[j]*x + b[j];
            b[j] = b[j]*x + m[j];
        }
    }
}

/*  Calculates the (one over the) DFT of the B-spline coefficients.
    The thing is done by the slow-but-simple matrix multiply technique.
 */
void bspl_dft(int n, int order, double *dft_c, double *c) {
    int i,j;
    const int j0 = order/2;

    for(i=0; i<n; i++) {
        double om = -(2.0*M_PI*i)/n;
        double *f = dft_c + i;

        f[0] = 0.0;
        for(j=1; j<order; j++) {
            f[0] += cos(om*(j-j0))*c[j];
        }
        f[0] = 1.0/f[0];
    }
}

void set_L(sfac_t *pbc, double L[6]) {
    pbc->L[0][0] = L[0];
    pbc->L[1][1] = L[1];
    pbc->L[2][2] = L[2];
    pbc->L[1][0] = L[3];
    pbc->L[2][0] = L[4];
    pbc->L[2][1] = L[5];

    pbc->iV = 1./(L[0]*L[1]*L[2]);
    pbc->iL[0][0] = 1./L[0];
    pbc->iL[1][1] = 1./L[1];
    pbc->iL[2][2] = 1./L[2];
    pbc->iL[1][0] = -L[3]*L[2]*pbc->iV;
    pbc->iL[2][0] = (L[3]*L[5] - L[1]*L[4])*pbc->iV;
    pbc->iL[2][1] = -L[0]*L[5]*pbc->iV;
}

// fully initializes pbc
// L is a_x, b_y, c_z, b_x, c_x, c_y
int sfac_ctor(sfac_t *pbc, double L[6], int32_t K[3], int order) {
    int n,i,j;

    pbc->order = order;
    set_L(pbc, L);

    if(order%2 != 0) {
        printf("Bad order: %d (must be even)\n", order);
        return -1;
    }
    if(K[0] < 1 || K[1] < 1 || K[2] < 1) {
        printf("Bad K: %d %d %d\n", K[0], K[1], K[2]);
        return -1;
    }
    for(j=0;j<3;j++)
        pbc->K[j] = K[j];
    pbc->ldim = pbc->K[2]/2+1; // Add 1 to odd K and 2 to even K
                               // to store the cplx part of freq=0 [odd]
                               // or the cplx part of freq=0 and K/2 [even].
    pbc->iK = 1.0/(K[0]*K[1]*K[2]);

    if( (pbc->bspc = malloc(sizeof(double)*SQR(order))) == NULL)
        return -1;

    mk_bspl_coeffs(pbc->bspc, order);

    /* Precompute B magnitudes. */
    if( (pbc->iB[0] = malloc(sizeof(double)*(K[0]+K[1]+K[2]))) == NULL) {
        free(pbc->bspc);
        return -2;
    }
    pbc->iB[1] = pbc->iB[0] + K[0];
    pbc->iB[2] = pbc->iB[1] + K[1];
    for(j=0; j<3; j++) {
        // Last row of pbc->bspc are spline values at integer nodes.
        bspl_dft(K[j], order, pbc->iB[j], pbc->bspc+order*(order-1));
    }

    n = 2*pbc->ldim*pbc->K[1]*pbc->K[0]; // Padded array size.
    pbc->Q = fftw_malloc(sizeof(double)*n);
    if(pbc->Q == NULL) {
        free(pbc->bspc);
        free(pbc->iB[0]);
        return -5;
    }

    pbc->A = pbc->dA = NULL;

    pbc->fft_neg = fftw_plan_dft_r2c_3d(pbc->K[0], pbc->K[1], pbc->K[2],
            pbc->Q, pbc->FQ, FFTW_PATIENT);
    pbc->fft_pos = fftw_plan_dft_c2r_3d(pbc->K[0], pbc->K[1], pbc->K[2],
            pbc->FQ, pbc->Q, FFTW_PATIENT);

    return 0;
}

// Warning! Errors will result from cleaning up the same ctor-data twice.
void sfac_dtor(sfac_t *pbc) {
    free(pbc->bspc);
    free(pbc->iB[0]);
    fftw_destroy_plan(pbc->fft_neg);
    fftw_destroy_plan(pbc->fft_pos);
    fftw_free(pbc->Q);
    if(pbc->A != NULL) free(pbc->A);
    fftw_cleanup();
}

/* Precompute a radially symmetric array of convolution factors.
 */
int set_A_uni(sfac_t *pbc, rad_fn f, void *info) {
    const int even = (pbc->K[2]+1)%2; // Is last dimension even?
    int i = pbc->K[0]*pbc->K[1]*pbc->ldim;
    if(pbc->A == NULL) {
        pbc->A = malloc(sizeof(double)*i*7);
        if(pbc->A == NULL) return 2;
        pbc->dA = pbc->A + i;
    }

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

    for(i=0; i<pbc->K[0]; i++) {
        const double mx = WRAP(i, pbc->K[0])*pbc->iL[0][0];
        const double m20 = SQR(mx);
        int n0 = i*pbc->K[1];
        int j;

        for(j=0; j<pbc->K[1]; j++) {
            const double my = WRAP(i, pbc->K[0])*pbc->iL[1][0]
                            + WRAP(j, pbc->K[1])*pbc->iL[1][1];
            const double m21 = m20 + SQR(my);
            const double mz0 = WRAP(i, pbc->K[0])*pbc->iL[2][0]
                             + WRAP(j, pbc->K[1])*pbc->iL[2][1];
            const int n1 = (n0 + j)*pbc->ldim;
            double *A = pbc->A + n1;
            double s, *dA = pbc->dA + 6*n1;
            int k;

            A[0] = 0.5*f(m21, &s, info);
            dA[0] = s*mx*mx; dA[1] = s*my*my; dA[2] = s*mz0*mz0;
            dA[3] = s*my*mx; dA[4] = s*mz0*mx; dA[5] = s*mz0*my;
            A++; dA+=6;
            for(k=1; k<pbc->ldim-even; k++,A++,dA+=6) {
                const double mz = mz0 + k*pbc->iL[2][2];
                const double m2 = m21 + SQR(mz);
                double s;

                A[0] = f(m2, &s, info); s *= 2.0;
                dA[0] = s*mx*mx; dA[1] = s*my*my; dA[2] = s*mz*mz;
                dA[3] = s*my*mx; dA[4] = s*mz*mx; dA[5] = s*mz*my;
            }
            if(even) { // Count Nyquist freq. once.
                const double mz = mz0 + k*pbc->iL[2][2];
                const double m2 = m21 + SQR(mz);
                A[0] = 0.5*f(m2, &s, info);
                dA[0] = s*mx*mx; dA[1] = s*my*my; dA[2] = s*mz*mz;
                dA[3] = s*my*mx; dA[4] = s*mz*mx; dA[5] = s*mz*my;
            }
        }
    }

    return 0;
}

int set_A(sfac_t *pbc, rad_fn f, void *info) {
    int i, n[3];
    const int i0 = (int)(1.0/pbc->iL[0][0]);
    const double R2 = 1.0;

    i = pbc->K[0]*pbc->K[1]*pbc->ldim;
    if(pbc->A == NULL) {
        pbc->A = malloc(sizeof(double)*i*7);
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
        const double mx = i*pbc->iL[0][0];
        const double m20 = SQR(mx);
        const double dy = sqrt(R2 - m20);
        const double my0 = i*pbc->iL[1][0];
        // |my0 + iL11*j| < dy
        const int j0 = ceil((-dy - my0)/pbc->iL[1][1]);
        const int j1 = floor((dy - my0)/pbc->iL[1][1]);
        int j;

        n[0] = MOD(i,pbc->K[0])*pbc->K[1];
        for(j=j0; j<=j1; j++) {
            const double my = my0 + j*pbc->iL[1][1];
            const double m21 = m20 + SQR(my);
            const double dz = sqrt(R2 - m21);
            const double mz0 = i*pbc->iL[2][0] + j*pbc->iL[2][1];
            // |mz0 + iL22*k| < dz
            const int k0 = ceil((-dz - mz0)/pbc->iL[2][2]);
            const int k1 = floor((dz - mz0)/pbc->iL[2][2]);
            int k;

            n[1] = (MOD(j,pbc->K[1])+n[0])*pbc->ldim;
            for(k=k0; k<=k1; k++) {
                const double mz = mz0 + k*pbc->iL[2][2];
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


static inline void cplx_mul(double *res, const double *a, const double *b) {
    res[0] = a[0]*b[0] - a[1]*b[1];
    res[1] = a[0]*b[1] + a[1]*b[0];
}

void sfac(sfac_t *pbc, int n, const double *w, const double *x) {
    int i, a;

    memset(pbc->Q, 0.0, sizeof(double)*pbc->K[0]*pbc->K[1]*pbc->ldim*2);
    for(a=0; a<n; a++) {
        int i, j, k;
        int k0[3];	// Start of relevant k values
        int n[3];	// Cumulative index to Q array
        double u[3];
        double yp, xp;
        double mpc[3][MAX_SPL_ORDER];

        // Calculate scaled particle position from s = L^{-T} n
        u[0] = (x[3*a+0]*pbc->iL[0][0] + x[3*a+1]*pbc->iL[1][0]
                + x[3*a+2]*pbc->iL[2][0]) * pbc->K[0];
        u[1] = (x[3*a+1]*pbc->iL[1][1] + x[3*a+2]*pbc->iL[2][1]
                 ) * pbc->K[1];
        u[2] = x[3*a+2]*pbc->iL[2][2] * pbc->K[2];
        k0[0] = (int)ceil(u[0]);
        k0[1] = (int)ceil(u[1]);
        k0[2] = (int)ceil(u[2]);

        // Fill mesh point precomputation arrays.
        bspl_coef(mpc[0], k0[0] - u[0], pbc->order, pbc->bspc);
        bspl_coef(mpc[1], k0[1] - u[1], pbc->order, pbc->bspc);
        bspl_coef(mpc[2], k0[2] - u[2], pbc->order, pbc->bspc);

        // make sure to index positive values
        k0[0] = MOD(k0[0] - pbc->order/2, pbc->K[0]);
        k0[1] = MOD(k0[1] - pbc->order/2, pbc->K[1]);
        k0[2] = MOD(k0[2] - pbc->order/2, pbc->K[2]);

        //printf("Atom %d: u = %.2f %.2f %.2f, k0 = %d %d %d\n", a+1,
        //		u[0],u[1],u[2], k0[0],k0[1],k0[2]);
        // Multiply values and add block into array.
        for(i=0; i<pbc->order; i++) {
            n[0] = ((k0[0]+i) % pbc->K[0])*pbc->K[1];
            xp = w[a]*mpc[0][i];
            for(j=0; j<pbc->order; j++) {
                n[1] = (n[0] + (k0[1]+j) % pbc->K[1]) * pbc->ldim*2;
                yp = xp*mpc[1][j];
                for(k=0; k<pbc->order; k++) {
                    n[2] = n[1] + (k0[2]+k) % pbc->K[2];
                    pbc->Q[n[2]] += yp*mpc[2][k];
                }
            }
        }
    }

    fftw_execute(pbc->fft_neg); // FQ = F-(Q)

    // Divide FQ by the FFT of the B-spline smoothing operation.
    // sets FQ to FQ/F(B)
#pragma omp parallel for private(i)
    for(i=0; i < pbc->K[0]*pbc->K[1]*pbc->ldim; i++) {
        double *Q = pbc->Q + 2*i;
        double iB[3] = {
            pbc->iB[0][ i/(pbc->ldim*pbc->K[1]) ],
            pbc->iB[1][ (i/pbc->ldim) % pbc->K[1] ],
            pbc->iB[2][ i% pbc->ldim ] };
        double t = iB[0]*iB[1]*iB[2];

        Q[0] *= t; Q[1] *= t;
    }
}

/* Stuff you can do with the spatial FT */

// Calculates the convolution, sum_kj Q_k F-[A]_{k-j} Q_j
// using the Fourier-space A, FQ
double en(sfac_t *pbc) {
    int i;
    const int n = pbc->K[0]*pbc->K[1]*pbc->ldim;
    double en = 0.0;
    #pragma omp parallel for reduction(+:en)
    for(i=0; i<n; i++) {
        double t = SQR(pbc->FQ[i][0])+SQR(pbc->FQ[i][1]);
        en += pbc->A[i]*t;
    }
    
    return en*pbc->iV;
}

/* Note: for all these, A[z = 0] and A[z = n/2] (when n is even)
 * must be doubled so that the single-loop will work.
 *
 * This is done by the set_A routine.
 */

/* Carry out a convolution with 'A' and return derivative of
   sum_k A_k |Q|_k^2 wrt the lattice vectors.
      Input:	Reciprocal space multipliers, A and their radial derivatives, dA
                It is assumed these have radial symmetry.
      Output:	Returns energy
                FQ is replaced with A FQ = F+[F-[A] * Q]
 */
double de1(sfac_t *pbc, double vir[6]) {
    int i;
    const int n = pbc->K[0]*pbc->K[1]*pbc->ldim;
    const int even = (pbc->K[2]+1)%2;
    mm_complex *FQ = pbc->FQ;
    double en = 0.0;
    double vir00,vir11,vir22,vir10,vir20,vir21;

    /* Overwrite FQ with A FQ and sum en. */
    vir00 = vir11 = vir22 = 0.0;
    vir10 = vir20 = vir21 = 0.0;
#pragma omp parallel for reduction(+:en) reduction(+:vir00) reduction(+:vir11) reduction(+:vir22) reduction(+:vir10) reduction(+:vir20) reduction(+:vir21)
    for(i=0; i<n; i++) {
        double s = SQR(FQ[i][0]) + SQR(FQ[i][1]);
        double *dA = pbc->dA + i*6;
        double iB[3] = {
            pbc->iB[0][ i/(pbc->ldim*pbc->K[1]) ],
            pbc->iB[1][ (i/pbc->ldim) % pbc->K[1] ],
            pbc->iB[2][ i% pbc->ldim ] };
        double t = pbc->A[i]*iB[0]*iB[1]*iB[2];

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
    } // Now: FQ = A FQ

    double fac = SQR(pbc->iV);
    vir[0] = (vir00 + en)*fac;
    vir[1] = (vir11 + en)*fac;
    vir[2] = (vir22 + en)*fac;
    vir[3] = vir10*fac;
    vir[4] = vir20*fac;
    vir[5] = vir21*fac;

    return en * pbc->iV;
}

/* Calculate the change in en wrt. input positions, x.
      Input:	Original reciprocal space multipliers, A and coordinates.
                It is assumed de1 has already been called
                in order to scale FQ!
      Output:	Energy derivatives, dE/dx.
 */
void de2(sfac_t *pbc, int n, const double *w, const double *x, double *dx0) {
    double en = 0.0;
    int a;

    fftw_execute(pbc->fft_pos); // Q = V F+[A F-(Q)]

#pragma omp parallel for reduction(+:en)
    for(a=0; a<n; a++) {
        int i, j, k;
        int k0[3];	// Start of relevant k values
        double u[3];
        double dx[3] = {0., 0., 0.}; // sum_l C(l) dQ(l)/ds
        double mpc[3][MAX_SPL_ORDER];
        double dmpc[3][MAX_SPL_ORDER];

        // Calculate scaled particle position from s = L^{-T} n
        u[0] = (x[3*a+0]*pbc->iL[0][0] + x[3*a+1]*pbc->iL[1][0]
                + x[3*a+2]*pbc->iL[2][0]) * pbc->K[0];
        u[1] = (x[3*a+1]*pbc->iL[1][1] + x[3*a+2]*pbc->iL[2][1]
                 ) * pbc->K[1];
        u[2] = x[3*a+2]*pbc->iL[2][2] * pbc->K[2];
        k0[0] = (int)ceil(u[0]);
        k0[1] = (int)ceil(u[1]);
        k0[2] = (int)ceil(u[2]);

        // Fill mesh point precomputation arrays.
        dbspl_coef(mpc[0], dmpc[0], k0[0] - u[0], pbc->order, pbc->bspc);
        dbspl_coef(mpc[1], dmpc[1], k0[1] - u[1], pbc->order, pbc->bspc);
        dbspl_coef(mpc[2], dmpc[2], k0[2] - u[2], pbc->order, pbc->bspc);

        // make sure to index positive values
        k0[0] = MOD(k0[0] - pbc->order/2, pbc->K[0]);
        k0[1] = MOD(k0[1] - pbc->order/2, pbc->K[1]);
        k0[2] = MOD(k0[2] - pbc->order/2, pbc->K[2]);

        //printf("dAtom %d: u = %.2f %.2f %.2f, k0 = %d %d %d\n", a+1,
        //		u[0],u[1],u[2], k0[0],k0[1],k0[2]);
        // Multiply values and get forces from a block of the array.
        for(i=0; i<pbc->order; i++) {
            int ni = ((k0[0]+i) % pbc->K[0]) * pbc->K[1]; // Dimensionality...
            for(j=0; j<pbc->order; j++) {
                double fj0 = dmpc[0][i] *  mpc[1][j];
                double fj1 =  mpc[0][i] * dmpc[1][j];
                double fj2 =  mpc[0][i] *  mpc[1][j];
                int nj = (ni + (k0[1]+j) % pbc->K[1])*pbc->ldim*2; // summing...
                for(k=0; k<pbc->order; k++) {
                    int nk = nj + (k0[2]+k) % pbc->K[2]; // to final n.
                    double fk = mpc[2][k]*pbc->Q[nk];
                    dx[0] += fj0 * fk;
                    dx[1] += fj1 * fk;
                    dx[2] += fj2 * dmpc[2][k]*pbc->Q[nk];
                    //en += w[a]*fj2*fk;
                    en += w[a] * mpc[0][i] *  mpc[1][j] * mpc[2][k] * pbc->Q[nk];
                }
            }
        }
        dx[0] *= (-w[a]*pbc->iV)*pbc->K[0];
        dx[1] *= (-w[a]*pbc->iV)*pbc->K[1];
        dx[2] *= (-w[a]*pbc->iV)*pbc->K[2];
        // dE(r)/ dr_i = dE(s)/ds_j * ds_j/dr_i (s_j = iL_{ij} r_i)
        dx0[3*a+0] = pbc->iL[0][0]*dx[0];
        dx0[3*a+1] = pbc->iL[1][0]*dx[0] + pbc->iL[1][1]*dx[1];
        dx0[3*a+2] = pbc->iL[2][0]*dx[0] + pbc->iL[2][1]*dx[1]
                                           + pbc->iL[2][2]*dx[2];
    }
    en *= 0.5*pbc->iV;
    //printf("E = %f\n", en);
}
