#include <vector>
#include <RcppEigenForward.h>
#include "RngStream.h"
#include "Rv.h"
#include "AllContainersForward.h"
#include <Rcpp.h>
#include <RcppEigenWrap.h>
#include "utilities.h"
#include "AllContainers.h"
#include "bamba.h"

using namespace Rcpp;
using namespace Eigen;

Bamba::Bamba(SEXP r_bead_list, SEXP r_chain_pars,
        SEXP r_latent_terms, SEXP r_linear_terms,
        SEXP r_hyper_list, SEXP r_at_matrix, SEXP r_update_beads) :
                data(r_bead_list),
                chain_pars(r_chain_pars),
                latent(r_latent_terms),
                hypers(r_hyper_list, r_at_matrix),
                linear(r_linear_terms, chain_pars.nBurn(), hypers),
                rng(RngStream_CreateStream("")) {
    const int n_iter = chain_pars.nIter();
    update_beads = as<bool>(r_update_beads);
    all_fitted = VectorXd::Constant(data.n_mfi, 0.0);
    y_tilde = VectorXd::Constant(data.n_mfi, 0.0);

    if (!linear.isNull()) {
        trace_b.resize(linear.Zt.rows(), n_iter);
        trace_u.resizeLike(trace_b);
        trace_theta.resize(linear.theta.rows(), n_iter);
    }

    trace_p.resize(hypers.At.rows(), n_iter);
    trace_mu_g.resize(latent.Mt.rows(), n_iter);
    trace_nu_g.resize(latent.Mt.rows(), n_iter);
    trace_hypers.resize(5, n_iter);
    trace_bead_mfi.resize(latent.Mt.cols(), n_iter);
    trace_bead_prec.resize(latent.Mt.cols(), n_iter);
    ppr.resizeLike(latent.gamma);
    ppr.fill(0.0);
}

void Bamba::iterate() {
    if (linear.isNull()) {
        y_tilde = data.y;
    } else {
        y_tilde =  data.y - linear.fitted;
    }
    latent.updateGamma(rng, y_tilde, hypers);
    latent.updateMuMarginal(rng, linear, data.y, hypers);

    if (linear.isNull()) {
        latent.updateNu(rng, data.y, hypers);
        all_fitted = latent.fitted;
    } else {
        linear.updateTheta(rng, latent.fitted, latent.weights, data.y, hypers);
        linear.updateU(rng, latent.fitted, latent.weights, data.y, hypers);
        latent.updateNu(rng, data.y - linear.fitted, hypers);
        all_fitted = linear.fitted + latent.fitted;
    }

    if (update_beads) {
        data.update(rng, all_fitted, latent.weights, hypers.bead_precision_shape,
                hypers.bead_precision_rate);
    }

    hypers.update(rng, latent.nu_g, latent.gamma, latent.mu_g, latent.var_prior);
    return;
}

void Bamba::gatherTrace(const int i) {
    // linear RE component parameters
    if (!linear.isNull()) {
        trace_b.col(i) = linear.b;
        trace_u.col(i) = linear.u;
        trace_theta.col(i) = linear.theta;
    }

    // latent component parameters
    double frac_old = (double) i / (double) (i + 1);
    double frac_new = (double) 1.0 / (double) (i + 1);
    ppr = ppr * frac_old + latent.gamma * frac_new;
    trace_mu_g.col(i) = latent.mu_g;
    trace_nu_g.col(i) = latent.nu_g;

    // hyperparameters
    trace_p.col(i) = hypers.p;
    if ((latent.var_prior == "gamma") | (latent.var_prior == "gamma_mean_var")) {
        trace_hypers(0, i) = hypers.mfi_nu_shape;
        trace_hypers(1, i) = hypers.mfi_nu_rate;
    }
    if (latent.var_prior == "folded_t") {
        trace_hypers(0, i) =
                hypers.folded_t_location;
        trace_hypers(1, i) = hypers.folded_t_scale;
    }
    trace_hypers(2, i) = hypers.tau;
    trace_hypers(3, i) = hypers.mu_overall;
    trace_hypers(4, i) = hypers.sig_delta;

    // bead level parameters
    trace_bead_mfi.col(i) = data.y;
    for (int j = 0; j < trace_bead_mfi.rows(); j++)
        trace_bead_prec(j, i) = data.bead_vec[j].getPrecision();

    return;
}

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
SEXP bambaMcmc(SEXP r_bead_list, SEXP r_chain_pars,
        SEXP r_latent_reTrms, SEXP r_linear_reTrms,
        SEXP r_hyper_list, SEXP r_at_matrix,
        SEXP r_update_beads) {
    Rcout << "Initializing model... " << std::endl;
    Bamba model(r_bead_list,
            r_chain_pars,
            r_latent_reTrms,
            r_linear_reTrms,
            r_hyper_list,
            r_at_matrix,
            r_update_beads);

    Rcout << "Beginning iterations. " << std::endl;
    int n_thin = model.chain_pars.nThin();
    int n_iter = model.chain_pars.nIter();
    int n_burn = model.chain_pars.nBurn();

    //model.latent.mu_g = (model.latent.Mt * model.data.y).cwiseQuotient(model.latent.counts);
    //model.latent.fitted = model.latent.Mt.transpose() * model.latent.mu_g;

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

    if(model.linear.isNull()) {
        return List::create(_["p"] = wrap(model.trace_p.transpose()),
                _["mu_g"] = wrap(model.trace_mu_g.transpose()),
                _["nu_g"] = wrap(model.trace_nu_g.transpose()),
                _["hypers"] = wrap(model.trace_hypers.transpose()),
                _["bead_mu"] = wrap(model.trace_bead_mfi.transpose()),
                _["bead_precision"] = wrap(model.trace_bead_prec.transpose()),
                _["Rt"] = wrap(model.latent.Rt),
                _["V"] = wrap(model.latent.V),
                _["data_y"] = wrap(model.data.y),
                _["ppr"] = wrap(model.ppr));
    } else {
        return List::create(_["b"] = wrap(model.trace_b.transpose()),
                _["u"] = wrap(model.trace_u.transpose()),
                _["p"] = wrap(model.trace_p.transpose()),
                _["mu_g"] = wrap(model.trace_mu_g.transpose()),
                _["nu_g"] = wrap(model.trace_nu_g.transpose()),
                _["hypers"] = wrap(model.trace_hypers.transpose()),
                _["bead_mu"] = wrap(model.trace_bead_mfi.transpose()),
                _["bead_precision"] = wrap(model.trace_bead_prec.transpose()),
                _["theta"] = wrap(model.trace_theta.transpose()),
                _["Lambdat"] = wrap(model.linear.Lambdat),
                _["tuner_cov"] = wrap(model.linear.cov_templates[0].theta_tuner.par_L),
                _["Omega_bar"] = wrap(model.latent.Omega_bar),
                _["Rt"] = wrap(model.latent.Rt),
                _["V"] = wrap(model.latent.V),
                _["data_y"] = wrap(model.data.y),
                _["ppr"] = wrap(model.ppr));
    }
}
