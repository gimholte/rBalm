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

double sigmaLik(const double sig, const double n, const double yss) {
    return - n * log(sig) -.5 * yss / (sig * sig);
}

double precisionLik(const double & nu, const double & mu, const double & sig2) {
    const double s = mu * mu / sig2;
    const double l = mu / sig2;

    return s * log(l) - R::lgammafn(s) + (s - 1) * log(nu) - nu * l;
}

double foldedTDensity(const double x, const double loc, const double scale, const double df) {
    const double pos_part = - ((df + 1.0) / 2.0) * log1p(pow2((x - loc) / scale) / df);
    const double neg_part = - ((df + 1.0) / 2.0) * log1p(pow2((x + loc) / scale) / df);
    return - log(scale) + pos_part + log(1 + exp(neg_part - pos_part));
}

double pow2(const double x) {
    return x * x;
}

double rejectionSamplerProb(const double x, const double A) {
    const double y = x / (A * A);
    return y / (y + 1.0);
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

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
double logDeterminant(const Eigen::MatrixXd & S) {
    double det = 0.0;
    for (int i = 0; i < S.rows(); ++i) {
        det += log(S(i, i));
    }
    return det;
}
