#include <string>
#include <iostream>
#include <sstream>
#include <math.h>
#include "snapshot.h"
#include "sfac.hpp"

int wrap(int i, int K) {
    return i <= K/2 ? i : K-i;
}

int main(int argc, char *argv[]) {
    std::string filename("dump.lammpstrj");
    double L[6] = {1.0, 1.0, 1.0,
                   0.0, 0.0, 0.0};
    int K[3] = {128,128,128};
    int order = 6;
    int ncount = 1000;
    double kmin = 0.001;
    double kmax = 3.0;

    while(argc > 2) {
        if(std::string(argv[1]) == "-K") {
            std::stringstream ss(argv[2]);
            ss >> K[2];
            K[0] = K[1] = K[2];
        } else if(std::string(argv[1]) == "-n") {
            std::stringstream ss(argv[2]);
            ss >> ncount;
        } else if(std::string(argv[1]) == "-o") {
            std::stringstream ss(argv[2]);
            ss >> order;
        } else if(std::string(argv[1]) == "-k0") {
            std::stringstream ss(argv[2]);
            ss >> kmin;
        } else if(std::string(argv[1]) == "-k1") {
            std::stringstream ss(argv[2]);
            ss >> kmax;
        } else {
            break;
        }

        argv += 2;
        argc -= 2;
        continue;
    }
    if(argc > 1) {
        filename = std::string(argv[1]);
    }
    SFac S(L, K, order);

    std::vector<double> hist(ncount, 0.0);
    std::vector<int> counts(ncount, 0);
    int n = 0;

    for(const Snapshot &snapshot : Dumpfile(filename)) {
        std::vector<double> w(snapshot.num_atoms, 1.0);
        /*for(int i=0; i<snapshot.num_atoms; i++) {
            w[i] = bfac[snapshot.types[i]];
        }*/
        L[0] = snapshot.L[0];
        L[1] = snapshot.L[1];
        L[2] = snapshot.L[2];
        // set_L(&S, L); // don't set, since coords are scaled by 1/L
        S(snapshot.num_atoms, &w[0], &snapshot.coords[0]);

        n++;
        std::cerr << "Frame " << n << std::endl;
        // smallest of these 3
        double inner = K[0]/L[0];
        inner = K[1]/L[1] < inner ? K[1]/L[1] : inner;
        inner = K[2]/L[2] < inner ? K[2]/L[2] : inner;
        inner *= M_PI;
        for(int i=0; i < K[0]*K[1]*S.ldim; i++) {
            // Decode each index into its k-vector
            double *Qi = S.Q + 2*i;
            double kx = 2*M_PI*wrap( i%S.ldim, K[0]) / L[0];
            double ky = 2*M_PI*wrap( (i/S.ldim)%K[1], K[1]) / L[1];
            double kz = 2*M_PI*wrap( i/(S.ldim*K[1]), K[2]) / L[2];
            double k = sqrt(kx*kx + ky*ky + kz*kz);

            if(k > inner) continue;

            double Q2 = Qi[0]*Qi[0] +  Qi[1]*Qi[1];
            if(k >= kmin && k < kmax) {
                int j = (k-kmin)*ncount/(kmax-kmin);
                hist[j] += Q2;
                counts[j] += 1;
            }
        }
    }

    for(int j=0; j<ncount; j++) {
        double k = kmin + (kmax-kmin)*(j+0.5)/ncount;
        if(counts[j] > 0) {
            std::cout << k << " "
                      << hist[j]/counts[j] << " "
                      << counts[j] << std::endl;
        }
    }

    return 0;
}
