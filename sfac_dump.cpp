/* sfac_dump - compute scattering spectrum from a LAMMPS dump file
 * using the particle mesh algorithm (smooth B-splines and an FFT).
 *
 * written in 2025 by David M. Rogers (ORNL)
 *
 * Usage: sfac_dump [dump.lammpstrj] [bfac.txt]
 *
 * Program options:
 *  -K NK -- k-point grid size in each dimension
 *  -n histogram_bins -- number of q-bins to collect the 1D S(q) plot onto
 *  -q0 qmin -- lower limit for output plot
 *  -q1 qmax -- upper limit for output plot
 *
 * If provided, bfac.txt should be a 2-column text file
 * listing <int atom type number> <b-factor>
 * for each atom type.  If not provided, all b-factors are set to 1
 * regardless of atom types.
 *
 * Note the smallest q-value is set by the MD system volume:
 *   dq_j = 2\pi/L_j
 *
 * The largest q-value is set by the quality of numerical approximation,
 * which we truncate at:
 *   q_max = \pi*\min_j(NK_j/L_j)
 *
 * When using this code, please cite:
 *
 *   David M. Rogers, "Extension of Kirkwood-Buff theory to the canonical ensemble."
 *   J. Chem. Phys. 148, 054102 (2018). https://doi.org/10.1063/1.5011696.
 *
 */
#include <string>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdexcept>

#include <fstream>
#include <map>

#include "snapshot.h"
#include "sfac.hpp"


int wrap(int i, int K) {
    return i <= K/2 ? i : i-K;
}


std::map<int, float> read_bfac_map(std::string &fname) {
    std::map<int, float> dataMap;
    std::ifstream inputFile(fname);

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file " << fname << std::endl;
        throw std::runtime_error("Error opening bfactors file");
    }

    int key;
    float value;

    while (inputFile >> key >> value) {
        dataMap[key] = value;
    }
    inputFile.close();
    return dataMap;
}

int main(int argc, char *argv[]) {
    std::string filename("dump.lammpstrj");
    std::string bfac_map("");
    std::map<int, float> bfac;
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
        } else if(std::string(argv[1]) == "-q0") {
            std::stringstream ss(argv[2]);
            ss >> kmin;
        } else if(std::string(argv[1]) == "-q1") {
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
    if(argc > 2) {
        bfac_map = std::string(argv[2]);
    }
    SFac S(L, K, order);

    std::vector<double> hist(ncount, 0.0);
    std::vector<int> counts(ncount, 0);
    int n = 0;
    double b2_sum = 0.0;
    double atom_sum = 0.0;
    if(bfac_map.size() > 0) {
        // load bfac map
        bfac = read_bfac_map(bfac_map);
    }

    for(const Snapshot &snapshot : Dumpfile(filename)) {
        std::vector<double> w(snapshot.num_atoms, 1.0);
        if(bfac_map.size() > 0) {
            for(int i=0; i<snapshot.num_atoms; i++) {
                w[i] = bfac[snapshot.types[i]];
            }
        }
        // normalization sum
        double local_sum = 0.0;
        for(int i=0; i<snapshot.num_atoms; i++) {
            local_sum += w[i]*w[i];
        }
        n++;
        b2_sum += local_sum;
        atom_sum += snapshot.num_atoms;
        if(local_sum == 0.0) {
            std::cerr << "Warning: no atoms from this frame scatter! Check bfac.txt" << std::endl;
            continue;
        }

        // Note: these three numbers should be the length of each
        // unit cell dimension, not the diagonals of the box vectors.
        L[0] = snapshot.L[0];
        L[1] = snapshot.L[1];
        L[2] = snapshot.L[2];
        // set_L(&S, L); // don't set, since coords are scaled by 1/L
        S(snapshot.num_atoms, &w[0], &snapshot.coords[0]);

        std::cerr << "Frame " << n << std::endl;
        // smallest of these 3
        // inner = pi*min(Kj/Lj)
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

    double norm_fac = n/b2_sum; // divide by avg. sum(b*b)
    std::cout << "# q S(q) counts -- for " << n << " frames at " << atom_sum/n << " atoms/frame -- normalized by avg \\sum_i b_i**2 = " << (b2_sum/n) << std::endl;

    for(int j=0; j<ncount; j++) {
        double k = kmin + (kmax-kmin)*(j+0.5)/ncount;
        if(counts[j] > 0) {
            hist[j] *= norm_fac;
            std::cout << k << " "
                      << hist[j]/counts[j] << " "
                      << counts[j] << std::endl;
        }
    }

    return 0;
}
