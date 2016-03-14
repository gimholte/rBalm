#include "Rcpp.h"
#include "RngStream.h"
#include "Rv.h"
#include "beadDist.h"

beadDist::beadDist(Rcpp::List & bead_el, double prior_shape, double prior_rate) :
        median(Rcpp::as<double>(bead_el["median"])),
        sum_d(Rcpp::as<double>(bead_el["sum_d"])),
        prec_prior_shape(prior_shape),
        prec_prior_rate(prior_rate),
        j(Rcpp::as<int>(bead_el["j"]))
{
    SEXP ll = bead_el["sort_x"];
    sort_x = Rcpp::NumericVector(ll);

    ll = bead_el["d"];
    d = Rcpp::NumericVector(ll);

    mu = median;
    n = sort_x.size();
    prec = n / sum_d;
    mcmc_scale_mult = 1.5 / sqrt(2 * n * prec);
}

double beadDist::computeSumAbsDev(const double x) {
    const double m_x(fabs(median - x));
    if (n == 2) {
        return fabs(sort_x(0) - x) + fabs(sort_x(1) - x);
    }
    if (n == 1) {
        return fabs(sort_x(0) - x);
    }
    int k = j - 1;
    double deduct = 0.0;
    const int k_increment = (x >= median) ? 1 : -1;
    while (1) {
        if ((x - sort_x(k + k_increment)) * k_increment <= 0.0) {
            break;
        }
        k += k_increment;
        if ((k + k_increment < 0) | (k + k_increment == n)) {
            break;
        }
    }

    const int istart = (x >= median) ? j : j - 1;
    for (int i = istart; i != k + k_increment; i += k_increment) {
        deduct -= 2.0 * d(i);
    }
    return sum_d + deduct + k_increment * m_x * (2 * (k + 1) - n + (k_increment - 1));
}

double beadDist::computeLaplaceLikelihood(const double sum_abs_dev_x) {
    return -sum_abs_dev_x * prec + n * log(prec);
}

void beadDist::updateLaplacePrecision(RngStream & rng) {
    // beads distributed Laplace(mu, prec^{-1})
    // prec ~ Gamma(prec_prior_shape, prec_prior_rate)
    // full conditional is then Gamma(n + prec_prior_shape, prec_prior_rate + sum_d)
    const double full_conditional_rate = cur_smd + prec_prior_rate;
    const double full_conditional_shape = n + prec_prior_shape;
    prec = RngStream_GA1(full_conditional_shape, rng) / full_conditional_rate;
    return;
}

void beadDist::updateLaplaceMu(RngStream & rng, const double mu_prior_mean,
        const double mu_prior_prec) {
    const double prop_mu = RngStream_N01(rng) * mcmc_scale_mult + mu;
    const double prop_smd = computeSumAbsDev(prop_mu);
    const double prop_lik = computeLaplaceLikelihood(prop_smd);
    const double prop_prior = -.5 * pow(prop_mu - mu_prior_mean, 2) * mu_prior_prec;

    const double cur_lik = computeLaplaceLikelihood(cur_smd);
    const double cur_prior = -.5 * pow(mu - mu_prior_mean, 2) * mu_prior_prec;

    const double joint_prop = prop_lik + prop_prior;
    const double joint_cur = cur_lik + cur_prior;

    const double loga = joint_prop - joint_cur;
    if (log(RngStream_RandU01(rng)) <= loga) {
        mu = prop_mu;
        cur_smd = prop_smd;
    }
    return;
}

void parseBeadDistList(Rcpp::List & bead_list,
        std::vector<beadDist> & bvec,
        Rcpp::List & bead_prior_pars) {
    double prior_shape(1.0), prior_rate(1.0);
    if (bead_prior_pars.containsElementNamed("prior_shape")) {
        prior_shape = Rcpp::as<double>(bead_prior_pars["prior_shape"]);
    }
    if (bead_prior_pars.containsElementNamed("prior_rate")) {
        prior_rate = Rcpp::as<double>(bead_prior_pars["prior_rate"]);
    }

    Rcpp::List::iterator in_iter;
    for (in_iter = bead_list.begin(); in_iter != bead_list.end(); ++in_iter) {
        Rcpp::List ll(*in_iter);
        bvec.push_back(beadDist(ll, prior_shape, prior_rate));
    }
    return;
}





