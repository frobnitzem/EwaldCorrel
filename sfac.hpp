#include <vector>

#include <fftw3.h>

#define MAX_SPL_ORDER (8)

#define SQR(x) ((x)*(x))
#define MOD(a,b) (((a)%(b) + (b)) % (b))
#define WRAP(a,b) (a > b/2 ? a-b : a)

typedef fftw_plan fft_plan_t;
typedef double mm_complex[2];

struct BSpline {
    const int n;
    std::vector<double> M;

    BSpline(const int n_) : n(n_), M(n_*n_) {
        mk_bspl_coeffs();
    }

    /*  Calculates the coefficients used in bspl_coef
        Input:    Order of interpolation
        Output:    Array of coefficients for bspl_coef,
                    M(x-floor(x)+j) = sum_i M_ij (x-floor(x))**i
        
        Note: This function must be called before any spline thing is done.
            mk_bspl_coeffs(pbc->bspc, pbc->order)
     */
    void mk_bspl_coeffs() {
        int i, j, k;
        double p, f, den;

        for(int i=0; i<n*n;i++) M[i] = 0.0;

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
                    M[j*n+i] += p;
                    p *= j-k;
                }
            }
        }
        p = 1.0;
        for(i=1; i<n-1; i++) { // multiply by n-1 C i
            p *= (n-i)/ (double)i;
            for(j=0; j<n; j++)
                M[j*n+i] *= p;
        }
    }

    /*  Calculates the raw coefficient of the Euler Cardinal B-Spline.
     *
     * Input:    Point x
     *    Order of interpolation
     *       Array of coefficients for calculation
     *
     * Output:    Spline coefficients, M_n(x+j) for j=0,1,...,n-1 - assuming x in [0,1)
     * 
     *    Note: This routine fails for x<0, which should rightly return 0.
     * It also produces innaccurate results for x>=order, which should also
     * be 0.  However, since these circumstances are not found in this
     * program, they are here ignored.
     */
    double bspl_coef(const double x, const int j) {
        double b = M[n*j];

        for(int i=1; i<n; i++) {
            b = b*x + M[n*j + i];
        }
        return b;
    }

    /*  Calculates the raw coefficient of the
     *   Euler Cardinal B-Spline and its derivative.
     *   Input:    Point x
     *Order of interpolation
     *Array of coefficients for calculation
     *  Output:    Spline coefficients, M_n(x+j) for j=0,1,...,n-1 - assuming x in [0,1)
     *      and derivatives, M'_n(x+j) for j=0,1,...,n-1 - assuming x in [0,1)
     *  Note: This routine fails for x<0, which should rightly return 0.
     *  It also produces innaccurate results for x>=order, which should also
     *  be 0.  However, since these circumstances are not found in this
     *  program, they are here ignored.
     * Assumes n >= 1, as it should be for nonzero derivatives.
     */
    void dbspl_coef(double *bp, double *dbp,
                    const double x, const int j) {
        double db = M[j*n];
        double b  = M[j*n]*x + M[j*n+1];

        for(int i=2; i<n; i++) {
            db = db*x + b;
            b = b*x + M[j*n+i];
        }
        *bp = b;
        *dbp = db;
    }
};

/* Structure factor info.
 * Z-axis is last in storage order (row / X-major).
 */
struct SFac {
    int order; // B-spline order
    int K[3]; // Size of grid.
    int ldim; // number of cplx numbers along last grid dimension
    double L[3][3]; // Box shape
    double iL[3][3]; // inverse box shape
    double iV;
    double iK; // inverse-K-volume
    BSpline bspc; // B-spline coefficients.
    std::vector<std::vector<double> > iB; // sizes { K[0], K[1], K[2] }
    fft_plan_t fft_neg, fft_pos; // pbc->K[2]/2+1
    double *A; // K[0] x K[1] x ldim (cplx, 0 imag part)
    double *dA; // K[0] x K[1] x ldim x 6
    union {
        double *Q; // K[0] x K[1] x ldim*2 (K[2] used) (real)
        mm_complex *FQ; // K[0] x K[1] x ldim x 2 (cplx)
    };

    SFac(double L[6], int32_t K[3], int order);
    ~SFac();
    void operator()(int n, const double *w, const double *x);
};
// For the last dimension, there are K[2] real numbers,
// and K[2] frequencies, stored sequentially from 0.
// However only ldim of those frequencies are actually
// stored, since FQ[i,j,k] = FQ[-i,-j,-k]^*.

// SFac's C interface:
extern "C" {
    void *sfac_ctor(double L[6], int K[3], int order);
    void sfac_dtor(void *sfac);
    void sfac(void *sfac, int n, const double *w, const double *x);
    double *get_S(void *sfac);

    // rad_fn must return (A, dA) as a function of r2 = R*R.
    // 'info' is passed through.
    typedef double (*rad_fn)(double r2, double *dA, void *info);

    void set_L(void *sfac, double L[6]);
    int set_A(void *sfac, rad_fn f, void *info);
    double en(void *sfac);
    double de1(void *sfac, double vir[6]);
    void de2(void *sfac, int n, const double *w, const double *x, double *dx);
}
