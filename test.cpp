#include <stdio.h>

#include "sfac.hpp"

int main() {
    int order = 4;
    double dx = 0.1;
    BSpline B(order);

    int nmax = order / dx;

    for(int i=0; i<=nmax; i++) {
        double x = i*dx;
        int jj = (int)x;
        double c = B.bspl_coef(x-jj, jj);
        printf("%.2f %f\n", x, c);
    }
}
