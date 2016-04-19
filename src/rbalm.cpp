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

using namespace Rcpp;
using namespace Eigen;

SimpleBalm::SimpleBalm(SEXP r_bead_list, SEXP r_chain_pars,
        SEXP r_latent_terms, SEXP r_linear_terms) :
                data(r_bead_list),
                chain_pars(r_chain_pars),
                latent(r_latent_terms),
                linear(r_linear_terms, chain_pars.nBurn()),
                hypers(),
                rng(RngStream_CreateStream("")) {
    const int n_iter = chain_pars.nIter();

    all_fitted = VectorXd::Constant(data.n_mfi, 0.0);
    trace_b.resize(linear.Zt.rows(), n_iter);
    trace_mu_g.resize(latent.Mt.rows(), n_iter);
    trace_nu_g.resize(latent.Mt.rows(), n_iter);
    trace_hypers.resize(3, n_iter);
    trace_bead_mfi.resize(latent.Mt.cols(), n_iter);
    trace_bead_prec.resize(latent.Mt.cols(), n_iter);
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
    trace_hypers(1, i) = hypers.mfi_nu_rate;
    trace_hypers(2, i) = hypers.shape_sampler.getW();
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

    for (int j = 1, k = 0; j <= n_thin * n_iter; j++) {
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
            _["Lambdat"] = wrap(model.linear.Lambdat),
            _["tuner_cov"] = wrap(model.linear.cov_templates[0].theta_tuner.par_L));
}
