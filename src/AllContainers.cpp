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
        checkListNames(r_names, *in_iter); // rely on implicit conversion to List.
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

    Mt = latent_term["Zt"]; // implicit conversion
    const int nmu = Mt.rows();
    const int nobs = Mt.cols();
    mu_g = VectorXd::Constant(nmu, 0.0);
    nu_g = VectorXd::Constant(nmu, 100.0);
    gamma = VectorXd::Constant(nmu / 2, 0);
    counts = Mt * VectorXd::Constant(nobs, 1.0);

    yss_w = VectorXd::Constant(nmu, 1.0);
    sum_y_w = VectorXd::Constant(nmu, 0.0);
    sum_yy_w = VectorXd::Constant(nmu, 0.0);
    resid = VectorXd::Constant(nobs, 1.0);
    fitted = Mt.transpose() * mu_g;
    weights = Mt.transpose() * nu_g;

    Rcout << "Initialized latent component... ";
};

void Latent::update(RngStream rng, const VectorXd & data_y,
        const VectorXd & linear_fitted, const Hypers & hypers) {
    updateMu(rng, data_y, linear_fitted, hypers);
    updateNu(rng, data_y, linear_fitted, hypers);
    return;
}

void Latent::updateNu(RngStream rng, const VectorXd & data_y,
        const VectorXd & linear_fitted, const Hypers & hypers) {
    double joint_rate, joint_shape;
    resid.noalias() = data_y - fitted - linear_fitted;
    yss_w.noalias() = Mt * resid.cwiseProduct(resid);
    size_t j, n = nu_g.size();
    for (j = 0; j < n; j++) {
        joint_shape = hypers.mfi_nu_shape + counts(j) / 2.0;
        joint_rate = hypers.mfi_nu_rate + yss_w(j) / 2.0;
        nu_g(j) = RngStream_GA1(joint_shape, rng) / joint_rate;
    }
    weights.noalias() = Mt.transpose() * nu_g;
    return;
}

void Latent::updateMu(RngStream rng, const VectorXd & data_y,
            const VectorXd & linear_fitted, const Hypers & hypers) {
    double c0, c1, c01;
    double l0, l1, l01;
    double mu0, mu1, mu01;
    double null_prob;
    sum_y_w.noalias() = Mt * (data_y - linear_fitted).cwiseProduct(weights);
    sum_yy_w.noalias() = Mt * (data_y - linear_fitted).cwiseProduct(data_y - linear_fitted).cwiseProduct(weights);

    size_t i, j, jp1, n_group = mu_g.size() / 2;
    for (i = 0, j = 0, jp1 = 1;
            i < n_group;
            ++i, j += 2, jp1 += 2) {
        l0 = hypers.gam0 + counts(j) * nu_g(j);
        l1 = hypers.gam1 + counts(jp1) * nu_g(jp1);
        l01 = hypers.gam01 + counts(j) * nu_g(j) + counts(jp1) * nu_g(jp1);

        mu0 = sum_y_w(j) / l0;
        mu1 = sum_y_w(jp1) / l1;
        mu01 = (sum_y_w(j) + sum_y_w(jp1)) / l01;

        c0 = -.5 * sum_yy_w(j);
        c0 += .5 * pow(sum_y_w(j), 2) / l0;
        c0 += .5 * log(hypers.gam0) -.5 * log(l0);

        c1 = -.5 * sum_yy_w(jp1);
        c1 += .5 * pow(sum_y_w(jp1), 2) / l1;
        c1 += .5 * log(hypers.gam1) -.5 * log(l1);

        c01 = -.5 * sum_yy_w(j) -.5 * sum_yy_w(jp1);
        c01 += .5 * pow(sum_y_w(j) + sum_y_w(jp1), 2) / l01;
        c01 += .5 * log(hypers.gam01) -.5 * log(l01);

        null_prob = (1 - hypers.p) / (hypers.p * exp(c0 + c1 - c01) + (1 - hypers.p));
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

Hypers::Hypers(SEXP r_fixed_hypers) :
        names_initialized(false),
        r_names(0),
        fixed_hypers(initializeHypersList(r_fixed_hypers)),
        gam0(as<double>(fixed_hypers["gam0"])),
        gam1(as<double>(fixed_hypers["gam1"])),
        gam01(as<double>(fixed_hypers["gam01"])),
        lambda_a_prior(as<double>(fixed_hypers["lambda_a_prior"])),
        lambda_b_prior(as<double>(fixed_hypers["lambda_b_prior"])),
        bead_precision_rate(as<double>(fixed_hypers["bead_precision_rate"])),
        bead_precision_shape(as<double>(fixed_hypers["bead_precision_shape"])),
        p_a_prior(as<double>(fixed_hypers["p_a_prior"])),
        p_b_prior(as<double>(fixed_hypers["p_b_prior"])),
        p(.05),
        mfi_nu_shape(.5),
        mfi_nu_rate(.5),
        shape_sampler(2.0, .25, lambda_a_prior, lambda_b_prior) { };

Hypers::Hypers() :
        names_initialized(true),
        r_names(0),
        fixed_hypers(0),
        gam0(.1),
        gam1(.1),
        gam01(.1),
        lambda_a_prior(.05),
        lambda_b_prior(.05),
        bead_precision_rate(1.0),
        bead_precision_shape(1.0),
        p_a_prior(.05),
        p_b_prior(.5),
        p(.05),
        mfi_nu_shape(.5),
        mfi_nu_rate(.5),
        shape_sampler(2.0, .25, lambda_a_prior, lambda_b_prior) {};

void ShapeDensity::setSummaryStatistics(const VectorXd & nu_vec) {
    n = nu_vec.rows();
    sum_nu = nu_vec.sum();
    sum_log_nu = 0.0;
    for (int i = 0; i < n; i++) {
        sum_log_nu += log(nu_vec(i));
    }
    return;
};

void Hypers::update(RngStream rng, VectorXd & mfi_precision) {
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

Linear::Linear(SEXP r_linear_terms, int n_burn) {
    initializeListNames();
    List linear_terms(r_linear_terms);
    bool is_ok = checkListNames(r_names, linear_terms);
    if (!is_ok)
        stop("Missing required list elements in Linear component initialization");

    theta = as<VectorXd>(linear_terms["theta"]);
    Lambdat = as<SpMat>(linear_terms["Lambdat"]);
    Zt = as<SpMat>(linear_terms["Zt"]);
    List r_Ztlist = as<List>(linear_terms["Ztlist"]);

    b = VectorXd::Constant(Zt.rows(), 0.0);
    u = VectorXd::Constant(Zt.rows(), 0.0);
    fitted = Zt.transpose() * b;

    I_p.resize(Zt.rows(), Zt.rows());
    I_p.setIdentity();
    LtZt = Lambdat * Zt;
    Omega = LtZt * LtZt.transpose() + I_p;

    helpers.initialize(r_linear_terms, Omega, Lambdat);

    VectorXi component_p = as<VectorXi>(linear_terms["component_p"]);
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

void Linear::update(RngStream rng, const VectorXd & latent_fitted,
        const VectorXd & mfi_obs_weights, const VectorXd & data_y,
        const Hypers & hypers) {
    LtZt = Lambdat * Zt;
    work_y_vec = LtZt * mfi_obs_weights.asDiagonal() * (data_y - latent_fitted);
    Omega = LtZt * mfi_obs_weights.asDiagonal() * LtZt.transpose() + I_p;
    /* sample will be stored in u */
    mvNormSim(rng, helpers.solver, Omega, work_y_vec, u);
    /* update b and fitted */
    setFitted();

    for (size_t i = 0; i < cov_templates.size(); ++i) {
        updateComponent(rng, latent_fitted, mfi_obs_weights, data_y,
                (int) i, Ztlist[i], helpers.block_lambdat[i], cov_templates[i]);
    };
}

void Linear::updateComponent(RngStream rng, const VectorXd & latent_fitted,
        const VectorXd & weights, const VectorXd & data_y,
        const int k,
        const SparseMatrix<double> & zt_block,
        SparseMatrix<double> & lambdat_block,
        CovarianceTemplate & cvt) {
    const int nb = helpers.nB(k);
    const int b_offset = helpers.getOffsetB(k);
    MHValues mhv;

    work_y_vec.noalias() = data_y - latent_fitted - fitted;

    const double cur_lik = thetaLikelihood(work_y_vec, weights);

    work_theta_vec = theta;
    cvt.proposeTheta(rng, work_theta_vec, mhv);

    work_y_vec.noalias() += zt_block.transpose() *
            lambdat_block.transpose() * u.segment(b_offset, nb);

    setLambdaHelperBlock(work_theta_vec, k);
    work_y_vec.noalias() -= zt_block.transpose() *
            lambdat_block.transpose() * u.segment(b_offset, nb);
    const double prop_lik = thetaLikelihood(work_y_vec, weights);

    mhv.setCurLik(cur_lik);
    mhv.setPropLik(prop_lik);

    if (log(RngStream_RandU01(rng)) < mhv.logA()) {
        // accept proposal
        theta.noalias() = work_theta_vec;
        setLambdaBlock(theta, k);
        // update the internal state of the covariance template
        cvt.acceptLastProposal(true);
        // update fitted values with new
        setFitted();
    } else {
        // reject proposal
        // reset the lambda helper
        setLambdaHelperBlock(theta, k);
        cvt.acceptLastProposal(false);
    }
    return;
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

double thetaLikelihood(const VectorXd & delta, const VectorXd & weights) {
    return -.5 * delta.cwiseProduct(weights).dot(delta);
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
    theta_tuner(n_burn_, 2),
    rho(rho_),
    prop_rho(rho_),
    sigma(p_),
    prop_sigma(p_),
    D(p_, p_),
    L_cor(p_, p_),
    DL(p_, p_),
    internal_theta(2),
    prop_internal_theta(2)
{
    sigma.fill(1.0);
    fillCholeskyDecomp();
    internal_theta(0) = 1.0;
    internal_theta(1) = rho2phi(rho, p);
};

void CovarianceTemplate::proposeTheta(RngStream rng, VectorXd & theta,
        MHValues & mhv) {
    if (p == 1) {
        double cur_sig = sigma(0);
        double prop_sig = exp(.5 * (RngStream_RandU01(rng) - .5)) * cur_sig;

        // temporarily fill the internal state with proposal values
        sigma.fill(prop_sig);
        prop_sigma.fill(prop_sig);
        prop_rho = rho;
        fillCholeskyDecomp();

        // input new theta value
        fillTheta(theta);

        // set MH proposal and prior values
        mhv.setCurPrior(-.5 * log(cur_sig) + .5 * cur_sig);
        mhv.setPropPrior(-.5 * log(prop_sig) + .5 * prop_sig);
        mhv.setCurToProp( -log(prop_sig));
        mhv.setPropToCur( -log(cur_sig));

        // return to original internal state
        sigma.fill(cur_sig);
        fillCholeskyDecomp();
    } else {
        const double cur_sig = sigma(0);
        internal_theta(0) = log(cur_sig);
        internal_theta(1) = rho2phi(rho, p);

        prop_internal_theta(0) = RngStream_N01(rng);
        prop_internal_theta(1) = RngStream_N01(rng);
        prop_internal_theta = theta_tuner.par_L * prop_internal_theta;
        prop_internal_theta.noalias() += internal_theta;

        const double prop_sig = exp(prop_internal_theta(0));
        const double cur_rho = rho;
        sigma.fill(prop_sig);
        prop_sigma.fill(prop_sig);
        prop_rho = phi2rho(prop_internal_theta(1), p);
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
        mhv.setCurToProp(log(dphi_drho(prop_rho, p)) - prop_internal_theta(0));
        mhv.setPropToCur(log(dphi_drho(cur_rho, p)) - internal_theta(0));

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






