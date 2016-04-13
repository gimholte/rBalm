/*
 * unit_tests.cpp
 *
 *  Created on: Apr 13, 2016
 *      Author: gimholte
 */

#include <vector>
#include <RcppEigenForward.h>
#include "RngStream.h"
#include "Rv.h"
#include "AllContainersForward.h"
#include <Rcpp.h>
#include <RcppEigenWrap.h>
#include "AllContainers.h"
#include "rbalm.h"

using namespace Eigen;
using namespace Rcpp;

// [[Rcpp::export]]
SEXP rBalmTestNuPriors() {
    Hypers hypers;
    RngStream rng = RngStream_CreateStream("");
    VectorXd nu_g(10);
    nu_g << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    int n_iter = 1000;
    MatrixXd out(n_iter, 2);
    for (int i = 0; i < n_iter; i++) {
        hypers.update(rng, nu_g);
        out(i, 0) = hypers.mfi_nu_shape;
        out(i, 1) = hypers.mfi_nu_rate;
    }
    return wrap(out);
}

// [[Rcpp::export]]
SEXP beadListTest(SEXP x_bead_list, SEXP x_eval) {
    ModelData beads(x_bead_list);
    NumericVector x(x_eval);
    int n = x.size();
    NumericVector sad(n);
    for (int i = 0; i < n; i++) {
        sad(i) = beads.bead_vec[i].computeSumAbsDev(x(i));
    }
    return wrap(sad);
}

// [[Rcpp::export]]
SEXP beadListMcmcTest(SEXP r_bead_list, SEXP r_chain_ctl) {
    ModelData beads(r_bead_list);
    ChainParameters cpars(r_chain_ctl);
    int n_mfi = beads.n_mfi;

    RngStream rng = RngStream_CreateStream("");
    VectorXd prior_mean = VectorXd::Constant(n_mfi, 0.0);
    VectorXd prior_prec = VectorXd::Constant(n_mfi, .01);

    double prec_prior_rate(0.1);
    double prec_prior_shape(0.1);

    MatrixXd out(n_mfi, cpars.nIter());
    MatrixXd out_prec(n_mfi, cpars.nIter());

    for (int i = 0; i < cpars.nBurn(); i++) {
        beads.update(rng, prior_mean, prior_prec,
                prec_prior_rate, prec_prior_shape);
    }

    for (int i = 0; i < cpars.nIter(); i++) {
        beads.update(rng, prior_mean, prior_prec,
                prec_prior_rate, prec_prior_shape);
        out.col(i) = beads.y;
        for (int j = 0; j < n_mfi; j++) {
            out_prec(j, i) = beads.bead_vec[j].getPrecision();
        }
    }
    return List::create(_["mfi"] = wrap(out.transpose()),
            _["prec"] = wrap(out_prec.transpose()));
}

// [[Rcpp::export]]
SEXP testSolver(SEXP r_linear_terms) {
    Linear linear(r_linear_terms);
    linear.LtZt = linear.Lambdat * linear.Zt;
    linear.Omega = linear.LtZt * linear.LtZt.transpose() + linear.I_p;
    linear.helpers.solver.factorize(linear.Omega);
    VectorXd m(linear.Omega.rows());
    m.fill(1.0);

    VectorXd out =  linear.helpers.solver.matrixL().solve(m);
    return List::create(_["om_test"] = wrap(out),
            _["om"] = wrap(linear.Omega));
}

// [[Rcpp::export]]
SEXP testMvNormSim(SEXP omega_, SEXP tau_, SEXP n_) {
    const SparseMatrix<double> omega(as<SparseMatrix<double> >(omega_));
    const VectorXd tau(as<VectorXd>(tau_));
    int n(as<int>(n_));

    if (omega.cols() != omega.rows())
        return R_NilValue;

    if (omega.cols() != tau.size())
        return R_NilValue;

    MatrixXd output(omega.rows(), n);
    VectorXd sample(omega.rows());

    SimplicialLLT<SparseMatrix<double> > solver;
    solver.analyzePattern(omega);
    RngStream rng = RngStream_CreateStream("");
    for (int i = 0; i < n; ++i) {
        mvNormSim(rng, solver, omega, tau, sample);
        output.col(i) = sample;
    }
    return wrap(output);
}

// [[Rcpp::export]]
SEXP testCholeskyFill() {
    CovarianceTemplate t1(0, 5, .25);
    CovarianceTemplate t2(0, 1, .25);

    Eigen::VectorXd test_sigma;
    test_sigma.resize(5);
    test_sigma << 1, 2, 3, 4, 5;
    t1.setSigma(test_sigma);

    return wrap(List::create(_["t1_lower"] = t1.lower_tri,
            _["t1_DL"] = t1.getDL(),
            _["t2_lower"] = t2.lower_tri,
                        _["t2_DL"] = t2.getDL()));
}
