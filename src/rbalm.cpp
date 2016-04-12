#include <vector>
#include <RcppEigenForward.h>
#include "RngStream.h"
#include "Rv.h"
#include "AllContainersForward.h"
#include <Rcpp.h>
#include <RcppEigenWrap.h>
#include "AllContainers.h"

using namespace std;
using namespace Rcpp;
using namespace Eigen;

class SimpleBalm {
public:
    ModelData data;
    ChainParameters chain_pars;
    Latent latent;
    Linear linear;
    Hypers hypers;

    // helpers
    RngStream rng;
    VectorXd all_fitted;

    // trace
    MatrixXd trace_b;
    MatrixXd trace_mu_g;
    MatrixXd trace_nu_g;
    MatrixXd trace_hypers;
    MatrixXd trace_bead_mfi;
    MatrixXd trace_bead_prec;
    MatrixXd trace_theta;

    SimpleBalm(SEXP r_bead_list, SEXP r_chain_pars,
            SEXP r_latent_terms, SEXP r_linear_terms);

    void iterate();
    void gatherTrace(const int i);
};

SimpleBalm::SimpleBalm(SEXP r_bead_list, SEXP r_chain_pars,
        SEXP r_latent_terms, SEXP r_linear_terms) :
                data(r_bead_list),
                chain_pars(r_chain_pars),
                latent(r_latent_terms),
                linear(r_linear_terms),
                hypers(),
                rng(RngStream_CreateStream("")) {
    const int n_iter = chain_pars.nIter();

    all_fitted = VectorXd::Constant(data.n_mfi, 0.0);
    trace_b.resize(linear.Zt.rows(), n_iter);
    trace_mu_g.resize(latent.M.cols(), n_iter);
    trace_nu_g.resize(latent.M.cols(), n_iter);
    trace_hypers.resize(2, n_iter);
    trace_bead_mfi.resize(latent.M.rows(), n_iter);
    trace_bead_prec.resize(latent.M.rows(), n_iter);
    trace_theta.resize(linear.theta.rows(), n_iter);
}

void SimpleBalm::iterate() {
    all_fitted = linear.fitted + latent.fitted;
    data.update(rng, all_fitted, latent.weights, hypers.bead_precision_shape,
            hypers.bead_precision_rate);
    linear.update(rng, latent.fitted, latent.weights, data.y, hypers);
    latent.update(rng, data.y, linear.fitted, hypers);
    hypers.update(rng, latent.nu_g);

    return;
}

void SimpleBalm::gatherTrace(const int i) {
    trace_b.col(i) = linear.b;
    trace_mu_g.col(i) = latent.mu_g;
    trace_nu_g.col(i) = latent.nu_g;
    trace_hypers(0, i) = hypers.mfi_nu_shape;
    trace_hypers(0, i) = hypers.mfi_nu_rate;
    trace_bead_mfi.col(i) = data.y;
    for (int j = 0; j < trace_bead_mfi.rows(); j++)
        trace_bead_prec(j, i) = data.bead_vec[j].getPrecision();
    trace_theta.col(i) = linear.theta;
    return;
}

// [[Rcpp::depends(RcppEigen)]]
RcppExport SEXP rBalmMcmc(SEXP r_bead_list, SEXP r_chain_pars,
        SEXP r_latent_reTrms, SEXP r_linear_reTrms) {
    Rcout << "Initializing model... ";
    SimpleBalm model(r_bead_list,
            r_chain_pars,
            r_latent_reTrms,
            r_linear_reTrms);

    Rcout << "Beginning iterations. " << std::endl;
    int n_thin = model.chain_pars.nThin();
    int n_iter = model.chain_pars.nIter();
    int n_burn = model.chain_pars.nBurn();

    for (int i = 0; i < n_burn; i++) {
        model.iterate();
    }
    int k = 0;

    for (int j = 1; j <= n_thin * n_iter; j++) {
        model.iterate();
        if (j % n_thin == 0) {
            model.gatherTrace(k);
            k++;
        }
    }
    return List::create(_["b"] = wrap(model.trace_b.transpose()),
            _["mu_g"] = wrap(model.trace_mu_g.transpose()),
            _["nu_g"] = wrap(model.trace_nu_g.transpose()),
            _["hypers"] = wrap(model.trace_hypers.transpose()),
            _["bead_mu"] = wrap(model.trace_bead_mfi.transpose()),
            _["bead_precision"] = wrap(model.trace_bead_prec.transpose()),
            _["theta"] = wrap(model.trace_theta.transpose()),
            _["Lambdat"] = wrap(model.linear.Lambdat));
}

RcppExport SEXP rBalmTestNuPriors() {
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

/*
 * Evaluate the sum of absolute deviations a list of bead distributions
 * x_bead_list: R list of processed bead distributions
 * x_eval: R numeric vector of values at which to evaluate the SADs
 */
RcppExport SEXP beadListTest(SEXP x_bead_list, SEXP x_eval) {
    ModelData beads(x_bead_list);
    NumericVector x(x_eval);
    int n = x.size();
    NumericVector sad(n);
    for (int i = 0; i < n; i++) {
        sad(i) = beads.bead_vec[i].computeSumAbsDev(x(i));
    }
    return wrap(sad);
}

RcppExport SEXP beadListMcmcTest(SEXP r_bead_list, SEXP r_chain_ctl) {
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

RcppExport SEXP testSolver(SEXP r_linear_terms) {
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

RcppExport SEXP testMvNormSim(SEXP omega_, SEXP tau_, SEXP n_) {
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

RcppExport SEXP testCholeskyFill() {
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

RcppExport SEXP block(SEXP Z_) {
    SparseMatrix<double> Z(as<SparseMatrix<double> >(Z_)), tmp;
    tmp = Z.block(0, 0, 2, 2);
    tmp.coeffRef(0, 0) = 1.0;
    return wrap(List::create(_["tmp"] = tmp, _["Z"] = Z));
}
