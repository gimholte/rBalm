/*
 * utilities.cpp
 *
 *  Created on: Apr 18, 2016
 *      Author: gimholte
 */

#include "RcppEigen.h"
#include "RngStream.h"
#include "Rv.h"
#include "utilities.h"

using namespace Eigen;
using namespace Rcpp;
typedef SparseMatrix<double> SpMat;

void mvNormSim(RngStream rng, SimplicialLLT<SpMat> & solver,
        const SpMat & omega, const VectorXd & tau, VectorXd & sample)
{
    for (size_t i = 0, n = sample.rows(); i < n; i++)
        *(sample.data() + i) = RngStream_N01(rng);

    solver.factorize(omega);
    solver.matrixU().solveInPlace(sample);
    sample.noalias() += solver.solve(tau);
}

bool checkListNames(const std::vector<std::string> & names, const List & ll) {
    bool all_in(true);
    for (size_t i = 0, n = names.size(); i < n; i++) {
        all_in &= ll.containsElementNamed(names[i].c_str());
    }
    return all_in;
}

double rho2phi(double rho, int p) {
    const double one_over_p = 1.0 / p;
    const double rho_full = rho * 2.0 / (1.0 + one_over_p) + (one_over_p - 1.0) / (1.0 + one_over_p);
    return atanh(rho_full);
}

double phi2rho(double phi, int p) {
    const double one_over_p = 1.0 / p;
    return (tanh(phi) - (one_over_p - 1.0) / (1.0 + one_over_p)) * (1.0 + one_over_p) / 2.0;
}

double drho_dphi(double phi, int p) {
    return 4.0 * ((1.0 + 1.0 / p) / 2.0) / pow(exp(phi) + exp(-phi), 2);
}

double dphi_drho(double rho, int p) {
    const double one_over_p = 1.0 / p;
    const double rho_full = rho * 2.0 / (1.0 + one_over_p) + (one_over_p - 1.0) / (1.0 + one_over_p);
    return (1.0 / (1.0 - pow(rho_full, 2))) *  (2.0 / (1.0 + one_over_p));
}
