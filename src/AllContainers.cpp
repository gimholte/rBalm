//#include <vector>
//#include <map>
//#include <string>
#include <RcppEigenForward.h>
#include "RngStream.h"
#include "Rv.h"
#include "AllContainersForward.h"
#include <Rcpp.h>
#include <RcppEigenWrap.h>
#include "utilities.h"
#include "AllContainers.h"

namespace Rcpp {
    /* Define the conversion from SEXP to BeadDist object
     * This is kind of a wrapper for what already happens in the BeadDist constructor*/
    template<> BeadDist as(SEXP bead_dist_sexp) {
        Rcpp::List bead_dist(bead_dist_sexp);
        BeadDist ell(bead_dist);
        return ell;
    };
};

using namespace Rcpp;
using namespace Eigen;

typedef SparseMatrix<double> SpMat;
typedef MappedSparseMatrix<double> MSpMat;

ModelData::ModelData(SEXP r_bead_dist_list) {
    initializeListNames();
    List bead_dist_list(r_bead_dist_list);
    List::iterator in_iter, in_end = bead_dist_list.end();
    for (in_iter = bead_dist_list.begin(); in_iter != in_end; ++in_iter) {
        if (!checkListNames(r_names, *in_iter))
            stop("failure initializing bead distributions");
        // rely on implicit conversion to List.
        bead_vec.push_back(as<BeadDist>(*in_iter));
    }
    n_mfi = bead_vec.size();
    y.resize(n_mfi);
    fillY();
};

void ModelData::update(RngStream rng,
        VectorXd & mu_prior_mean,
        VectorXd & mu_prior_precision,
        const double precision_prior_shape,
        const double precision_prior_rate) {
    if (mu_prior_mean.rows() != n_mfi)
        stop("invalid length of mu_prior_mean in BeadDist update");

    if (mu_prior_precision.rows() != n_mfi)
        stop("invalid length of mu_prior_precision in BeadDist update");

    double m, nu;
    int i;
    std::vector<BeadDist>::iterator it_bd, bd_end = bead_vec.end();
    for (i = 0, it_bd = bead_vec.begin(); it_bd != bd_end; ++it_bd, ++i) {
        m = mu_prior_mean(i);
        nu = mu_prior_precision(i);
        it_bd->updateLaplacePrecision(rng, precision_prior_shape,
                precision_prior_rate);
        it_bd->updateLaplaceMu(rng, m, nu);
        y(i) = it_bd->getMu();
    }
    return;
};

ChainParameters::ChainParameters(SEXP r_chain_pars) {
    initializeListNames();
    List chain_pars(r_chain_pars);
    bool is_ok = checkListNames(r_names, chain_pars);
    if (!is_ok)
        stop("Missing required list elements in Chain component initialization");
    std::vector<std::string>::iterator it;
    for (it = r_names.begin(); it != r_names.end(); ++it) {
        iter_pars[*it] = as<int>(chain_pars[it->c_str()]);
    }
};

Latent::Latent(SEXP r_latent_term) {
    initializeListNames();
    /* check that provided list has required named elements */
    List latent_term(r_latent_term);
    if (!checkListNames(r_names, latent_term))
        stop("Missing required list elements in Latent component initialization");

    /* check that we have one term in latent model formula */
    List cnms = latent_term["cnms"]; // implicit conversion
    if (cnms.size() != 1)
        stop("Invalid number of terms in latent model formula");

    /* check that number of term levels for each group level is two */
    CharacterVector cnms_tmp = cnms[0]; // implicit conversion
    if (cnms_tmp.size() != 2)
        stop("Invalid number of latent response levels in latent model (must be 2)");

    var_prior = as<std::string>(latent_term["var_prior"]);
    Mt = latent_term["Zt"]; // implicit conversion
    const int nmu = Mt.rows();
    const int nobs = Mt.cols();
    Rt.resize(nmu, nobs);
    Rt.reserve(VectorXi::Constant(Mt.cols(), 1));

    Omega_bar.resize(nobs, nobs);
    G_diag.resize(nmu);
    tau.resize(nmu);
    mu_compressed.resize(nmu);

    G_diag.fill(1.0);
    G.resize(nmu, nmu);
    G.setIdentity();
    V.resize(nobs, nobs);
    V.setIdentity();

    tau.fill(1.0);
    mu_compressed.fill(1.0);

    mu_g = VectorXd::Constant(nmu, 0.0);
    nu_g = VectorXd::Constant(nmu, 148.3589);
    gamma = VectorXd::Constant(nmu / 2, 1.0);
    counts = Mt * VectorXd::Constant(nobs, 1.0);

    yss_w = VectorXd::Constant(nmu, 1.0);
    sum_y_w = VectorXd::Constant(nmu, 0.0);
    sum_yy_w = VectorXd::Constant(nmu, 0.0);
    resid = VectorXd::Constant(nobs, 1.0);
    fitted = Mt.transpose() * mu_g;
    weights = Mt.transpose() * nu_g;
    V = V * weights.asDiagonal();
};

void Latent::updateNu(RngStream rng, const VectorXd & y_tilde,
        const Hypers & hypers) {
    resid.noalias() = y_tilde - fitted;
    yss_w.noalias() = Mt * resid.cwiseProduct(resid);
    int j;
    double prop_val, cur_val;
    MHValues mhv;
    for (j = 0; j < nu_g.size(); j++) {
        if ((var_prior == "gamma") | (var_prior == "gamma_mean_var")) {
            double joint_rate, joint_shape;
            joint_shape = hypers.mfi_nu_shape + counts(j) / 2.0;
            joint_rate = hypers.mfi_nu_rate + yss_w(j) / 2.0;
            nu_g(j) = RngStream_GA1(joint_shape, rng) / joint_rate;
        }
        if (var_prior == "half_cauchy") {
            cur_val = 1.0 / sqrt(nu_g(j));
            prop_val = cur_val * exp(.5* RngStream_UnifAB(-1.0, 1.0, rng));

            mhv.setCurLik(sigmaLik(cur_val, counts(j), yss_w(j), hypers.cauchy_sd_scale));
            mhv.setPropLik(sigmaLik(prop_val, counts(j), yss_w(j), hypers.cauchy_sd_scale));
            mhv.setCurPrior(0.0);
            mhv.setPropPrior(0.0);
            mhv.setPropToCur(-log(cur_val));
            mhv.setCurToProp(-log(prop_val));

            if (log(RngStream_RandU01(rng)) <= mhv.logA()) {
                nu_g(j) = 1.0 / (prop_val * prop_val);
            }
        }
    }
    weights.noalias() = Mt.transpose() * nu_g;
    V.setIdentity();
    V = V * weights.asDiagonal();
    return;
}

void Latent::updateMu(RngStream rng, const VectorXd & y_tilde,
        const Hypers & hypers) {
    double c0, c1, c01;
    double l0, l1, l01;
    double mu0, mu1, mu01;
    double null_prob;
    double p_local;
    VectorXd p_fitted = hypers.At.transpose() * hypers.p;
    sum_y_w.noalias() = Mt * (y_tilde).cwiseProduct(weights);

    size_t i, j, jp1, n_group = mu_g.size() / 2;
    for (i = 0, j = 0, jp1 = 1;
            i < n_group;
            ++i, j += 2, jp1 += 2) {
        l0 = hypers.tau + counts(j) * nu_g(j);
        l1 = hypers.tau + counts(jp1) * nu_g(jp1);
        l01 = hypers.tau + counts(j) * nu_g(j) + counts(jp1) * nu_g(jp1);

        mu0 = (sum_y_w(j) + hypers.tau * hypers.mu_overall) / l0;
        mu1 = (sum_y_w(jp1) + hypers.tau * hypers.mu_overall) / l1;
        mu01 = (sum_y_w(j) + sum_y_w(jp1) + hypers.tau * hypers.mu_overall) / l01;

        c0 = .5 * pow(mu0, 2) * l0 + .5 * log(hypers.tau) - .5 * log(l0);
        c1 = .5 * pow(mu1, 2) * l1 + .5 * log(hypers.tau) - .5 * log(l1);
        c01 = .5 * pow(mu01, 2) * l01 + .5 * log(hypers.tau) - .5 * log(l01);

        p_local = p_fitted[i];
        null_prob = (1 - p_local) / (p_local * exp(c0 + c1 - c01) + (1 - p_local));

        if (RngStream_RandU01(rng) < null_prob) {
            mu01 = RngStream_N01(rng) / sqrt(l01) + mu01;
            mu_g(j) = mu01;
            mu_g(jp1) = mu01;
            gamma(i) = 0.0;
        } else {
            mu_g(j) = RngStream_N01(rng) / sqrt(l0) + mu0;
            mu_g(jp1) = RngStream_N01(rng) / sqrt(l1) + mu1;
            gamma(i) = 1.0;
        }
    }
    fitted = Mt.transpose() * mu_g;
    return;
}

void Latent::constructRG(double g01, double g0, double g1,
        double m_c, double m_0, double m_1) {
    int n_terms = gamma.size() + (int) gamma.sum();
    Rt.resize(n_terms, Mt.cols());
    Rt.reserve(VectorXi::Constant(Mt.cols(), 1));

    G_diag.resize(n_terms);
    G.resize(n_terms, n_terms);
    G.setIdentity();
    prior_tau.resize(n_terms);
    VectorXi rowMap(Mt.rows());
    int new_row = 0;

    for (int k = 0, j = 0; k < gamma.size(); ++k) {
        rowMap[2 * k] = new_row;
        G_diag[new_row] = g01;
        prior_tau[j] = m_c;
        if (gamma[k] > .5) {
            G_diag[new_row] = g0;
            prior_tau[j] = m_0;
            ++new_row;
            ++j;
            G_diag[new_row] = g1;
            prior_tau[j] = m_1;
        }
        rowMap[2 * k + 1] = new_row;
        ++new_row;
        ++j;
    }

    for (int k = 0; k < Mt.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(Mt,k); it; ++it)
        {
            new_row = rowMap[it.row()];   // row index remapped based on gamma
            Rt.insert(new_row, it.col()) = it.value();
        }

    G = G * G_diag.asDiagonal();
    prior_tau = G * prior_tau;
    return;
}

void Latent::updateMuMarginal(RngStream rng,
        Linear & linear,
        const Eigen::VectorXd & data_y,
        const Hypers & hyp) {

    constructRG(hyp.tau, hyp.tau, hyp.tau,
            hyp.mu_overall, hyp.mu_overall, hyp.mu_overall);
    mu_compressed.resize(Rt.rows());
    if (linear.isNull()) {
        Omega_bar = V;
    } else {
        linear.LtZt = linear.Lambdat * linear.Zt;
        linear.Omega = linear.LtZt * V * linear.LtZt.transpose() + linear.I_p;
        linear.helpers.solver.factorize(linear.Omega);

        Omega_bar = linear.helpers.solver.solve(linear.LtZt);
        Omega_bar = V * linear.LtZt.transpose() * Omega_bar * V;
        Omega_bar = V - Omega_bar;
    }
    tau.noalias() = Rt * Omega_bar * data_y + prior_tau;
    Omega_bar = Rt * Omega_bar * Rt.transpose() + G;
    Omega_bar_solver.analyzePattern(Omega_bar);
    Omega_bar_solver.factorize(Omega_bar);

    mvNormSim(rng, Omega_bar_solver, Omega_bar, tau, mu_compressed);
    expandMu();
    fitted = Mt.transpose() * mu_g;
}

void Latent::expandMu() {
    int n_groups = gamma.size();
    int k = 0;
    for (int i = 0; i < n_groups; i++) {
        if (gamma[i] < .5) {
            mu_g[2 * i] = mu_compressed[k];
            mu_g[2 * i + 1] = mu_compressed[k];
        } else {
            mu_g[2 * i] = mu_compressed[k];
            k++;
            mu_g[2 * i + 1] = mu_compressed[k];
        }
        k++;
    }
}

Hypers::Hypers(SEXP r_fixed_hypers, SEXP r_at_matrix) :
        names_initialized(false),
        r_names(0),
        fixed_hypers(initializeHypersList(r_fixed_hypers)),
        lambda_a_prior(as<double>(fixed_hypers["lambda_a_prior"])),
        lambda_b_prior(as<double>(fixed_hypers["lambda_b_prior"])),
        shape_sampler(2.0, .25, lambda_a_prior, lambda_b_prior) {
    bead_precision_rate = as<double>(fixed_hypers["bead_precision_rate"]);
    bead_precision_shape = as<double>(fixed_hypers["bead_precision_shape"]);

    p_alpha = as<double>(fixed_hypers["p_alpha"]);
    p_beta = as<double>(fixed_hypers["p_beta"]);

    tau_prior_shape = as<double>(fixed_hypers["tau_prior_shape"]);
    tau_prior_rate = as<double>(fixed_hypers["tau_prior_rate"]);
    mu_overall_bar = as<double>(fixed_hypers["mu_overall_bar"]);

    n_0 = as<double>(fixed_hypers["n_0"]);

    theta_shape = as<double>(fixed_hypers["theta_shape"]);
    theta_rate = as<double>(fixed_hypers["theta_rate"]);
    tau = as<double>(fixed_hypers["tau"]);

    mfi_nu_shape = 5;
    mfi_nu_rate = .05;
    cauchy_sd_scale = as<double>(fixed_hypers["cauchy_sd_scale"]);
    prec_mean_prior_mean = as<double>(fixed_hypers["prec_mean_prior_mean"]);
    prec_mean_prior_sd = as<double>(fixed_hypers["prec_mean_prior_sd"]);
    prec_var_prior_mean = as<double>(fixed_hypers["prec_var_prior_mean"]);
    prec_var_prior_sd = as<double>(fixed_hypers["prec_var_prior_sd"]);

    mu_overall = 0.0;

    At = as<SparseMatrix<double> >(r_at_matrix);
    p.resize(At.rows());
    p.fill(0.05);
    a_counts.resize(At.rows());
    VectorXd tmp_ones = VectorXd::Constant(At.cols(), 1.0);
    total_counts = At * tmp_ones;
};

Hypers::Hypers(SEXP r_at_matrix) :
        names_initialized(true),
        r_names(0),
        fixed_hypers(0),
        lambda_a_prior(.05),
        lambda_b_prior(.05),
        shape_sampler(2.0, .25, lambda_a_prior, lambda_b_prior) {
    bead_precision_rate = 1.0,
    bead_precision_shape = 1.0,

    p_alpha = .05;
    p_beta = .5;

    tau_prior_shape = .5;
    tau_prior_rate = 5.;
    mu_overall_bar = 0.0;
    n_0 = .05;

    theta_shape = .5;
    theta_rate = .5;

    tau = .05;
    mu_overall = 0.0;

    mfi_nu_shape = .0005;
    mfi_nu_rate = .0005;


    cauchy_sd_scale = .1;

    prec_mean_prior_mean = 50.0;
    prec_mean_prior_sd = 50.0;
    prec_var_prior_mean = 50.0 * 50.0;
    prec_var_prior_sd = 250.0;

    At = as<SparseMatrix<double> >(r_at_matrix);
    p.resize(At.rows());
    p.fill(0.05);
    a_counts.resize(At.rows());
    VectorXd tmp_ones = VectorXd::Constant(At.cols(), 1.0);
    total_counts = At * tmp_ones;
};

void Hypers::update(RngStream rng, const VectorXd & mfi_precision, const VectorXd & gamma,
        const VectorXd & mu_vec, const std::string var_prior) {
    if (var_prior == "gamma") {
        shape_sampler.setSummaryStatistics(mfi_precision);
        double tmp_w = shape_sampler.getW();
        double new_mfi_nu_shape = unimodalSliceSampler(rng, mfi_nu_shape,
                shape_sampler.getUpper(), shape_sampler.getLower(), tmp_w,
                shape_sampler);
        shape_sampler.updateW(tmp_w);
        mfi_nu_shape = new_mfi_nu_shape;
        const int n = shape_sampler.getN();
        const double sum_nu = shape_sampler.getSumNu();
        mfi_nu_rate = RngStream_GA1(1.0 + n * mfi_nu_shape, rng) / (sum_nu + lambda_b_prior);
    }
    if (var_prior == "half_cauchy"){
        updateCauchySdScale(rng, mfi_precision);
    }
    if (var_prior == "gamma_mean_var") {
        updateGammaMeanVarPrior(rng, mfi_precision);
    }
    updateMuPrior(rng, gamma, mu_vec);
    updateP(rng, gamma);
}

void Hypers::updateGammaMeanVarPrior(RngStream rng, const VectorXd & mfi_precision) {
    double mu_cur = mfi_nu_shape / mfi_nu_rate;
    double sig_cur = mfi_nu_shape / pow(mfi_nu_rate, 2);
    double cur_lik(0.0), prop_lik(0.0), nu;
    MHValues mhv;

    const double mu_prop = mu_cur * exp(.2 * RngStream_UnifAB(-1, 1, rng));
    for (int i = 0; i < mfi_precision.size(); ++i) {
        nu = mfi_precision(i);
        cur_lik += precisionLik(nu, mu_cur, sig_cur);
        prop_lik += precisionLik(nu, mu_prop, sig_cur);
    }

    mhv.setCurLik(cur_lik);
    mhv.setPropLik(prop_lik);
    mhv.setCurPrior(precisionLik(mu_cur, prec_mean_prior_mean, pow(prec_mean_prior_sd, 2)));
    mhv.setPropPrior(precisionLik(mu_prop, prec_mean_prior_mean, pow(prec_mean_prior_sd, 2)));
    mhv.setCurToProp(-log(mu_prop));
    mhv.setPropToCur(-log(mu_cur));

    if (log(RngStream_RandU01(rng)) <= mhv.logA()) {
        mu_cur = mu_prop;
    }

    const double sig_prop = sig_cur * exp(.3 * RngStream_UnifAB(-1, 1, rng));
    prop_lik = 0.0;
    cur_lik = 0.0;
    for (int i = 0; i < mfi_precision.size(); ++i) {
        nu = mfi_precision(i);
        cur_lik += precisionLik(nu, mu_cur, sig_cur);
        prop_lik += precisionLik(nu, mu_cur, sig_prop);
    }

    mhv.setCurLik(cur_lik);
    mhv.setPropLik(prop_lik);
    mhv.setCurPrior(precisionLik(sig_cur, prec_var_prior_mean, pow(prec_var_prior_sd, 2)));
    mhv.setPropPrior(precisionLik(sig_prop, prec_var_prior_mean, pow(prec_var_prior_sd, 2)));
    mhv.setCurToProp(-log(sig_prop));
    mhv.setPropToCur(-log(sig_cur));

    if (log(RngStream_RandU01(rng)) <= mhv.logA()) {
        sig_cur = sig_prop;
    }

    mfi_nu_shape = mu_cur * mu_cur / sig_cur;
    mfi_nu_rate = mu_cur / sig_cur;
}

void Hypers::updateCauchySdScale(RngStream rng, const VectorXd & mfi_precision) {
    MHValues mhv;
    double nu, cur_lik(0.0), prop_lik(0.0);
    const double cur_val = cauchy_sd_scale;
    const double prop_val = cur_val * exp(.5 * RngStream_UnifAB(-1.0, 1.0, rng));

    int j;
    for (j = 0; j < mfi_precision.size(); ++j) {
        nu = mfi_precision(j);
        cur_lik += -log1p(1.0 / (cur_val * cur_val * nu));
        prop_lik += -log1p(1.0 / (prop_val * prop_val * nu));
    }
    double n_nu = (double) mfi_precision.size();
    cur_lik += -n_nu * log(cur_val);
    prop_lik += -n_nu * log(prop_val);

    mhv.setCurPrior(0.0);
    mhv.setPropPrior(0.0);
    if (prop_val > 10.0) {
        mhv.setPropPrior(-INFINITY);
    }
    mhv.setCurLik(cur_lik);
    mhv.setPropLik(prop_lik);
    mhv.setPropToCur(-log(cur_val));
    mhv.setCurToProp(-log(prop_val));

    if (log(RngStream_RandU01(rng)) < mhv.logA()) {
        cauchy_sd_scale = prop_val;
    }
    return;
}

void Hypers::updateMuPrior(RngStream rng, const Eigen::VectorXd & gamma,
        const Eigen::VectorXd & mu_vec) {
    double s1 = gamma.sum();
    double s0 = gamma.size() - s1;

    double mu_mean, mu_prec;
    mu_mean = mu_overall_bar * n_0;
    mu_prec = (n_0 + s0 + s1) * tau;

    int i, j, jp1;
    for (i = 0, j = 0, jp1 = 1; i < gamma.size(); ++i, j += 2, jp1 += 2) {
        if (gamma(i) > .5) {
            mu_mean += mu_vec(j);
            mu_mean += mu_vec(jp1);
        } else {
            mu_mean += mu_vec(j);
        }
    }

    mu_mean = tau * mu_mean / mu_prec;
    mu_overall = RngStream_N01(rng) / sqrt(mu_prec) + mu_mean;

    double tau_r = tau_prior_rate + .5 * n_0 * pow(mu_overall - mu_overall_bar, 2);
    for (int i = 0, j = 0, jp1 = 1; i < gamma.size(); ++i, j += 2, jp1 += 2) {
        if (gamma(i) > .5) {
            tau_r += .5 * pow(mu_vec(j) - mu_overall, 2);
            tau_r += .5 * pow(mu_vec(jp1) - mu_overall, 2);
        } else {
            tau_r += .5 * pow(mu_vec(j) - mu_overall, 2);
        }
    }

    tau = RngStream_GA1(tau_prior_shape + .5 * (1 + s0 + s1), rng) / tau_r;
}


void Hypers::updateP(RngStream rng, const VectorXd & gamma) {
    a_counts = At * gamma;
    double s, a, b;
    for (int j = 0; j < p.size(); ++j) {
        s  = a_counts[j];
        a = p_alpha + s;
        b = p_beta + total_counts[j] - s;
        p[j] = RngStream_Beta(a, b, rng);
    }
    return;
}

Linear::Linear(SEXP r_linear_terms, int n_burn, const Hypers & hyp) {
    if (Rf_isNull(r_linear_terms)) {
        is_null = true;
        return;
    }

    // if we made it here, r_linear_terms is not an empty list
    is_null = false;

    // check that required named components are present
    initializeListNames();
    List linear_terms(r_linear_terms);
    bool is_ok = checkListNames(r_names, linear_terms);
    if (!is_ok)
        stop("Missing required list elements in Linear component initialization");

    // acquire named components for linear model fitting
    theta = as<VectorXd>(linear_terms["theta"]);
    Lambdat = as<SpMat>(linear_terms["Lambdat"]);
    Zt = as<SpMat>(linear_terms["Zt"]);
    List r_Ztlist = as<List>(linear_terms["Ztlist"]);
    VectorXi component_p = as<VectorXi>(linear_terms["component_p"]);


    // initialize helper vectors/matrices
    b = VectorXd::Constant(Zt.rows(), 0.0);
    u = VectorXd::Constant(Zt.rows(), 0.0);
    fitted = Zt.transpose() * b;

    I_p.resize(Zt.rows(), Zt.rows());
    I_p.setIdentity();
    LtZt = Lambdat * Zt;
    Omega = LtZt * LtZt.transpose() +  I_p;

    helpers.initialize(r_linear_terms, Omega, Lambdat);
    // initialize block stuff for each sub-component (i.e, term)
    int n_component = r_Ztlist.size();
    for (int k = 0; k < n_component; k++) {
        cov_templates.push_back(CovarianceTemplate(helpers.getOffsetTheta(k),
                component_p(k), 0.0, n_burn));

        SEXP ztk = r_Ztlist[k];
        Ztlist.push_back(as<SpMat>(ztk));
    };

    if (!checkBlocks(helpers.block_lambdat, Lambdat)) {
        stop("Bad lambdat_blocks transfer in Linear::Linear(SEXP) constructor");
    };
};


Linear::Linear(SEXP r_linear_terms, int n_burn) {
    if (Rf_isNull(r_linear_terms)) {
        is_null = true;
        return;
    }

    // if we made it here, r_linear_terms is not an empty list
    is_null = false;

    // check that required named components are present
    initializeListNames();
    List linear_terms(r_linear_terms);
    bool is_ok = checkListNames(r_names, linear_terms);
    if (!is_ok)
        stop("Missing required list elements in Linear component initialization");

    // acquire named components for linear model fitting
    theta = as<VectorXd>(linear_terms["theta"]);
    Lambdat = as<SpMat>(linear_terms["Lambdat"]);
    Zt = as<SpMat>(linear_terms["Zt"]);
    List r_Ztlist = as<List>(linear_terms["Ztlist"]);
    VectorXi component_p = as<VectorXi>(linear_terms["component_p"]);


    // initialize helper vectors/matrices
    b = VectorXd::Constant(Zt.rows(), 0.0);
    u = VectorXd::Constant(Zt.rows(), 0.0);
    fitted = Zt.transpose() * b;

    I_p.resize(Zt.rows(), Zt.rows());
    I_p.setIdentity();
    LtZt = Lambdat * Zt;
    Omega = LtZt * LtZt.transpose() +  I_p;

    helpers.initialize(r_linear_terms, Omega, Lambdat);
    // initialize block stuff for each sub-component (i.e, term)
    int n_component = r_Ztlist.size();
    for (int k = 0; k < n_component; k++) {
        cov_templates.push_back(CovarianceTemplate(helpers.getOffsetTheta(k),
                component_p(k), 0.0, n_burn));

        SEXP ztk = r_Ztlist[k];
        Ztlist.push_back(as<SpMat>(ztk));
    };

    if (!checkBlocks(helpers.block_lambdat, Lambdat)) {
        stop("Bad lambdat_blocks transfer in Linear::Linear(SEXP) constructor");
    };
};


void Linear::updateU(RngStream rng, const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & mfi_obs_weights, const Eigen::VectorXd & data_y,
            const Hypers & hypers) {
    if (is_null) {
        return;
    }
    Omega = LtZt * mfi_obs_weights.asDiagonal() * LtZt.transpose() + I_p;
    work_y_vec = LtZt *
            mfi_obs_weights.asDiagonal() * (data_y - latent_fitted);
    /* sample will be stored in u */
    mvNormSim(rng, helpers.solver, Omega, work_y_vec, u);
    /* update b and fitted */
    setFitted();
    return;
}

void Linear::updateTheta(RngStream rng, const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & mfi_obs_weights, const Eigen::VectorXd & data_y,
            const Hypers & hypers) {
    if (is_null) {
        return;
    }
    for (size_t i = 0; i < cov_templates.size(); ++i) {
        updateComponent(rng, latent_fitted, mfi_obs_weights, data_y,
                (int) i, Ztlist[i], helpers.block_lambdat[i], cov_templates[i], hypers);
    };
    setFitted();
    return;
}

void Linear::updateComponent(RngStream rng, const VectorXd & latent_fitted,
        const VectorXd & weights, const VectorXd & data_y,
        const int k, const SparseMatrix<double> & zt_block,
        SparseMatrix<double> & lambdat_block, CovarianceTemplate & cvt,
        const Hypers & hyp) {
    MHValues mhv;
    LtZt = Lambdat * Zt;
    work_y_vec.noalias() = LtZt * weights.asDiagonal() * (data_y - latent_fitted);
    Omega = LtZt * weights.asDiagonal() * LtZt.transpose() + I_p;
    helpers.solver.factorize(Omega);

    const double cur_lik = thetaLikelihood(weights, Omega, helpers.solver, work_y_vec, u);
    work_theta_vec = theta * exp(.5 * (RngStream_RandU01(rng) - .5));

    //    cvt.proposeTheta(rng, work_theta_vec, mhv);
    setLambdaBlock(work_theta_vec, k);
    LtZt = Lambdat * Zt;
    work_y_vec.noalias() = LtZt * weights.asDiagonal() * (data_y - latent_fitted);
    Omega = LtZt * weights.asDiagonal() * LtZt.transpose() + I_p;
    helpers.solver.factorize(Omega);

    const double prop_lik = thetaLikelihood(weights, Omega, helpers.solver, work_y_vec, u);

//    const int nb = helpers.nB(k);
//    const int b_offset = helpers.getOffsetB(k);
//    MHValues mhv;
//
//    work_y_vec.noalias() = data_y - latent_fitted - fitted;
//    const double cur_lik = thetaLikelihood(work_y_vec, weights);
//
//    work_theta_vec = theta * exp(.1 * (RngStream_RandU01(rng) - .5));
//    work_y_vec.noalias() += zt_block.transpose() *
//            lambdat_block.transpose() * u.segment(b_offset, nb);
//
//    setLambdaHelperBlock(work_theta_vec, k);
//    work_y_vec.noalias() -= zt_block.transpose() *
//            lambdat_block.transpose() * u.segment(b_offset, nb);
//    const double prop_lik = thetaLikelihood(work_y_vec, weights);
//
    mhv.setCurLik(cur_lik);
    mhv.setPropLik(prop_lik);
    mhv.setCurPrior((hyp.theta_shape - 1.0) * log(theta(0)) -
            hyp.theta_rate * theta(0));
    mhv.setPropPrior((hyp.theta_shape - 1.0) * log(work_theta_vec(0)) -
            hyp.theta_rate * work_theta_vec(0));
    mhv.setCurToProp(-log(work_theta_vec(0)));
    mhv.setPropToCur(-log(theta(0)));

//    if (log(RngStream_RandU01(rng)) < mhv.logA()) {
//        // accept proposal
//        theta.noalias() = work_theta_vec;
//        setLambdaBlock(theta, k);
//        // update the internal state of the covariance template
//        //        cvt.acceptLastProposal(true);
//        // update fitted values with new
//        LtZt = Lambdat * Zt;
//        setFitted();
//    } else {
//        // reject proposal
//        // reset the lambda helper
//        setLambdaHelperBlock(theta, k);
//        //        cvt.acceptLastProposal(false);
//    }

    if (log(RngStream_RandU01(rng)) < mhv.logA()) {
        // accept proposal
        theta = work_theta_vec;
//        cvt.acceptLastProposal(true);
    } else {
        // reject proposal
        // reset the lambda block
//        cvt.acceptLastProposal(false);
        setLambdaBlock(theta, k);
        LtZt = Lambdat * Zt;
        work_y_vec.noalias() = LtZt * weights.asDiagonal() * (data_y - latent_fitted);
        Omega = LtZt * weights.asDiagonal() * LtZt.transpose() + I_p;
        helpers.solver.factorize(Omega);
    }
    return;
}

double thetaLikelihood(const VectorXd & delta, const VectorXd & weights) {
    return -.5 * delta.cwiseProduct(weights).dot(delta);
}

double thetaLikelihood(const VectorXd & weights,
        const SparseMatrix<double> Omega,
        const SimplicialLLT<SparseMatrix<double> > & solver,
        const VectorXd & tau,  VectorXd & u) {
    u = solver.solve(tau);
    const double lik = .5 * u.dot(tau) - logDeterminant(solver.matrixL());
    return lik;
}

void Linear::setLambdaHelperBlock(const VectorXd & new_theta, int block_idx) {
    // use Lind to fill the block of lambdat at block_idx with values in new_theta
    const int offset = helpers.getOffsetLambda(block_idx);
    const int n_theta = helpers.nLambda(block_idx);
    if (n_theta != helpers.numBlockNonZeros(block_idx)) {
        stop("nonzeros in lamdbat_block do not match replacement length");
    }
    double* block_ptr = helpers.getBlockValuePtr(block_idx);
    for (int j = 0, i = offset; j < n_theta; ++i, ++j) {
        *(block_ptr + j) = new_theta(helpers.getLind(i));
    }
    return;
}

void Linear::setLambdaBlock(const VectorXd & new_theta, int block_idx) {
    // use Lind to fill the block of lambdat starting at block_idx with values in new_theta
    const int offset = helpers.getOffsetLambda(block_idx);
    const int n_theta = helpers.nLambda(block_idx);
    for (int i = offset; i < offset + n_theta; ++i) {
        *(Lambdat.valuePtr() + i) = new_theta(helpers.getLind(i));
    }
    return;
}

BeadDist::BeadDist(const List & bead_dist) :
                        x_sorted(as<NumericVector>(bead_dist["x_sorted"])),
                        abs_dev_from_median(as<NumericVector>(bead_dist["abs_dev_from_median"])),
                        j(as<int>(bead_dist["j"])),
                        x_median(as<double>(bead_dist["x_median"])),
                        sum_abs_dev_median(as<double>(bead_dist["sum_abs_dev_median"]))
{
    mu = x_median;
    precision = x_sorted.size() / sum_abs_dev_median;
    sum_abs_dev_cur = sum_abs_dev_median;
    mcmc_scale_mult = 3.0 / sqrt(2 * x_sorted.size() * precision);
}

double BeadDist::computeSumAbsDev(const double x) {
    const double m_x(fabs(x_median - x));
    const int n(x_sorted.size());
    if (n == 2) {
        return fabs(x_sorted(0) - x) + fabs(x_sorted(1) - x);
    }
    if (n == 1) {
        return fabs(x_sorted(0) - x);
    }
    int k = j - 1;
    double deduct = 0.0;
    const int k_increment = (x >= x_median) ? 1 : -1;
    while (1) {
        if ((x - x_sorted(k + k_increment)) * k_increment <= 0.0) {
            break;
        }
        k += k_increment;
        if ((k + k_increment < 0) | (k + k_increment == n)) {
            break;
        }
    }
    for (int i = (x >= x_median) ? j : j - 1;
            i != k + k_increment;
            i += k_increment) {
        deduct -= 2.0 * abs_dev_from_median(i);
    }
    return sum_abs_dev_median +
            deduct + k_increment * m_x * (2 * (k + 1) - n + (k_increment - 1));
}

double BeadDist::computeLaplaceLikelihood(const double sum_abs_dev_x) {
    return -sum_abs_dev_x * precision + x_sorted.size() * log(precision);
}

void BeadDist::updateLaplacePrecision(RngStream rng,
        const double precision_prior_shape, const double precision_prior_rate) {
    // beads distributed Laplace(mu, prec^{-1})
    // prec ~ Gamma(prec_prior_shape, prec_prior_rate)
    // full conditional is then Gamma(n + prec_prior_shape, prec_prior_rate + sum_d)
    const double full_conditional_rate = sum_abs_dev_cur + precision_prior_rate;
    const double full_conditional_shape = x_sorted.size() + precision_prior_shape;
    precision = RngStream_GA1(full_conditional_shape, rng) / full_conditional_rate;
    return;
}

void BeadDist::updateLaplaceMu(RngStream rng, const double mu_prior_mean,
        const double mu_prior_prec) {
    MHValues mhv;
    const double prop_mu = RngStream_N01(rng) * mcmc_scale_mult + mu;
    const double prop_smd = computeSumAbsDev(prop_mu);

    mhv.setPropLik(computeLaplaceLikelihood(prop_smd));
    mhv.setPropPrior(-.5 * pow(prop_mu - mu_prior_mean, 2) * mu_prior_prec);

    mhv.setCurLik(computeLaplaceLikelihood(sum_abs_dev_cur));
    mhv.setCurPrior(-.5 * pow(mu - mu_prior_mean, 2) * mu_prior_prec);
    // symmetric proposal so set these to zero (log 1).
    mhv.setPropToCur(0.0);
    mhv.setCurToProp(0.0);

    if (log(RngStream_RandU01(rng)) <= mhv.logA()) {
        mu = prop_mu;
        sum_abs_dev_cur = prop_smd;
    }
    return;
}

CovarianceTemplate::CovarianceTemplate(int offset_val_, int p_, double rho_,
        int n_burn_) :
    within_theta_offset_val(offset_val_),
    p(p_),
    lower_tri(p_ * (p_ + 1) / 2),
    theta_tuner(n_burn_, p_ == 1 ? 1 : 2),
    rho(rho_),
    prop_rho(rho_),
    sigma(p_),
    prop_sigma(p_),
    D(p_, p_),
    L_cor(p_, p_),
    DL(p_, p_),
    internal_theta(p_ == 1 ? 1 : 2),
    prop_internal_theta(p_ == 1 ? 1 : 2)
{
    sigma.fill(1.0);
    fillCholeskyDecomp();
    internal_theta(0) = 0.0;
    if (p > 1)
        internal_theta(1) = rho2phi(rho, p);
};

void CovarianceTemplate::proposeTheta(RngStream rng, VectorXd & theta,
        MHValues & mhv) {
    if (p == 1) {
        const double cur_sig = sigma(0);
        const double log_cur_sig = internal_theta(0);
        const double log_prop_sig = log_cur_sig + .5 * (RngStream_RandU01(rng) - .5);
        const double prop_sig = exp(prop_internal_theta(0));

        // temporarily fill the internal template state with proposal values
        sigma.fill(prop_sig);
        fillCholeskyDecomp();

        // set internal proposal values... this saves the proposal values
        // for later possible acceptance in CovarianeTemplate::acceptLastProposal
        prop_internal_theta(0) = log_prop_sig;
        prop_sigma.fill(prop_sig);

        // fill passed theta vector with proposed theta parameter
        fillTheta(theta);

        // set MH proposal and prior values
        mhv.setCurPrior(-.5 * log_cur_sig + .5 * cur_sig);
        mhv.setPropPrior(-.5 * log_prop_sig + .5 * prop_sig);
        mhv.setCurToProp(-log_prop_sig);
        mhv.setPropToCur(-log_cur_sig);

        // return to original internal state
        sigma.fill(cur_sig);
        fillCholeskyDecomp();
    } else {
        const double cur_sig = sigma(0);
        internal_theta(0) = log(cur_sig);
        internal_theta(1) = rho2phi(rho, p);

        prop_internal_theta(0) = RngStream_RandU01(rng);
        prop_internal_theta(0) = theta_tuner.getScale() * prop_internal_theta(0) + internal_theta(0);
        prop_rho = RngStream_UnifAB(-1.0 / p, 1.0, rng);
        prop_internal_theta(1) = rho2phi(prop_rho, p);

        const double prop_sig = exp(prop_internal_theta(0));
        const double cur_rho = rho;
        sigma.fill(prop_sig);
        prop_sigma.fill(prop_sig);
        rho = prop_rho;
        fillCholeskyDecomp();
        fillTheta(theta);

        // uniform prior on rho, gamma prior on sigma
        mhv.setCurPrior(-.5 * log(cur_sig) + .5 * cur_sig);
        mhv.setPropPrior(-.5 * log(prop_sig) + .5 * prop_sig);
        // on sampling scale, symmetric proposal, so we need only the jacobian of transformation
        // from regular scale to sampling scale.
        // for rho, scaled derivative of inverse hyperbolic tangent
        // for sigma, the jacobian is just the value of log_sigma
        mhv.setCurToProp(/*log(dphi_drho(prop_rho, p))*/ - prop_internal_theta(0));
        mhv.setPropToCur(/*log(dphi_drho(cur_rho, p))*/ - internal_theta(0));

        sigma.fill(cur_sig);
        rho = cur_rho;
        fillCholeskyDecomp();
    }
    return;
}

void CovarianceTemplate::acceptLastProposal(bool accept) {
    if (accept) {
        sigma = prop_sigma;
        rho = prop_rho;
        fillCholeskyDecomp();
        internal_theta = prop_internal_theta;
    };
    theta_tuner.update(accept, internal_theta);
}

void CovarianceTemplate::fillCholeskyDecomp() {
    /* fill diagonals of L_cor and D */
    D.setZero();
    L_cor.setZero();
    int i, j, k;
    double ell;
    for (i = 0; i < p; i++) {
        j = i + 1;
        ell = sqrt(1.0 - (j - 1.0) * rho * rho / (1.0 + (j - 2.0) * rho));
        L_cor(i, i) = ell;
        D(i, i) = sigma(i);
    };

    /* fill off diagonal */
    for (j = 1; j < p; j++) {
        L_cor(j, 0) = rho;
    };
    for (k = 2; k < p; k++) {
        for (j = k + 1; j <= p; j++) {
            L_cor(j - 1, k - 1) = L_cor(k - 1, k - 1) * rho / (1.0 + (k - 1) * rho);
        };
    };
    DL = D * L_cor;

    /* extract the lower diagonal of overall covariance factor*/
    k = 0;
    for (j = 0; j < p; j++) {
        for (i = j; i < p; i++) {
            lower_tri(k) = DL(i, j);
            ++k;
        };
    };
    return;
};






