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
#include "utilities.h"
#include "AllContainers.h"
#include "rbalm.h"

using namespace Eigen;
using namespace Rcpp;

typedef MappedSparseMatrix<double> MSpMat;
typedef SparseMatrix<double> SpMat;

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Rcpp::List testLinearHelpersInit(SEXP retrms_, Eigen::VectorXd tau) {
    List retrms(retrms_);
    MSpMat Zt(retrms["Zt"]);
    SpMat Lt(retrms["Lambdat"]);
    SpMat Omega, Ip;
    Ip.resize(Zt.rows(), Zt.rows());
    Ip.setIdentity();
    Omega = Lt * Zt * Zt.transpose() * Lt.transpose() + Ip;
    LinearHelpers helpers;
    helpers.initialize(retrms_, Omega, Lt);

    // record the number of blocks;
    int nblocks = helpers.numBlocks();

    // record number of block nonzero values;
    VectorXi block_nonzeros(nblocks);
    for (int i = 0; i < nblocks; i++)
        block_nonzeros(i) = helpers.numBlockNonZeros(i);

    // record all lambdat blocks
    std::vector<SpMat> block_list;
    for (int i = 0; i < nblocks; i++)
        block_list.push_back(helpers.getLambdaBlock(i));

    // use solver to check it works;
    helpers.solver.factorize(Omega);
    return List::create(_["blocks"] = wrap(block_list),
            _["block_nonZeros"] = wrap(block_nonzeros),
            _["offset_lambda"] = wrap(helpers.getOffsetLambda()),
            _["helper_blocks"] = wrap(helpers.block_lambdat),
            _["tau_solve"] = helpers.solver.solve(tau));;
}

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Rcpp::List testLinearThetaFilling(SEXP retrms_) {
    Linear lin(retrms_, 100);
    int k = lin.helpers.numBlocks();
    if (k < 2) {
        return List::create(_[""] = R_NilValue);
    }
    VectorXd new_sig = lin.cov_templates[0].getSigma();
    new_sig.fill(2.0);
    lin.cov_templates[0].setSigma(new_sig);
    lin.cov_templates[0].fillTheta(lin.theta);
    lin.setLambdaBlock(lin.theta, 0);
    lin.setLambdaHelperBlock(lin.theta, 0);
    new_sig = lin.cov_templates[1].getSigma();
    new_sig.fill(3.0);
    double new_rho = .5;

    lin.cov_templates[1].setRho(new_rho);
    lin.cov_templates[1].setSigma(new_sig);
    lin.cov_templates[1].fillTheta(lin.theta);
    lin.setLambdaBlock(lin.theta, 1);
    lin.setLambdaHelperBlock(lin.theta, 1);

    return List::create(_["theta"] = lin.theta,
            _["template_L0"] = lin.cov_templates[0].getDL(),
            _["template_L1"] = lin.cov_templates[1].getDL(),
            _["Lambdat"] = lin.Lambdat,
            _["Lambdat_block0"] = lin.helpers.getLambdaBlock(0),
            _["Lambdat_block1"] = lin.helpers.getLambdaBlock(1));
}

// [[Rcpp::depends(RcppEigen)]]
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

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
NumericVector beadListTest(SEXP x_bead_list, NumericVector x) {
    ModelData beads(x_bead_list);
    int n = x.size();
    NumericVector sad(n);
    for (int i = 0; i < n; i++) {
        sad(i) = beads.bead_vec[i].computeSumAbsDev(x(i));
    }
    return sad;
}

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Rcpp::List beadListMcmcTest(SEXP r_bead_list, SEXP r_chain_ctl) {
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

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Rcpp::List testSolver(SEXP r_linear_terms) {
    Linear linear(r_linear_terms, 100);
    linear.LtZt = linear.Lambdat * linear.Zt;
    linear.Omega = linear.LtZt * linear.LtZt.transpose() + linear.I_p;
    linear.helpers.solver.factorize(linear.Omega);
    VectorXd m(linear.Omega.rows());
    m.fill(1.0);

    VectorXd out =  linear.helpers.solver.matrixL().solve(m);
    return List::create(_["om_test"] = wrap(out),
            _["om"] = wrap(linear.Omega));
}

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Eigen::MatrixXd testMvNormSim(const Eigen::SparseMatrix<double> omega,
        Eigen::VectorXd tau, int n) {
    if (omega.cols() != omega.rows())
        stop("");
    if (omega.cols() != tau.size())
        stop("");
    MatrixXd output(omega.rows(), n);
    VectorXd sample(omega.rows());

    SimplicialLLT<SparseMatrix<double> > solver;
    solver.analyzePattern(omega);
    RngStream rng = RngStream_CreateStream("");
    for (int i = 0; i < n; ++i) {
        mvNormSim(rng, solver, omega, tau, sample);
        output.col(i) = sample;
    }
    return output;
}

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
SEXP testCholeskyFill() {
    CovarianceTemplate t1(0, 5, .25, 1000);
    CovarianceTemplate t2(0, 1, .25, 1000);

    Eigen::VectorXd test_sigma;
    test_sigma.resize(5);
    test_sigma << 1, 2, 3, 4, 5;
    t1.setSigma(test_sigma);

    return wrap(List::create(_["t1_lower"] = t1.lower_tri,
            _["t1_DL"] = t1.getDL(),
            _["t2_lower"] = t2.lower_tri,
                        _["t2_DL"] = t2.getDL()));
}
