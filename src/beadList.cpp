#include "Rcpp.h"
#include "RngStream.h"
#include "Rv.h"
#include "beadList.h"

beadList::beadList(Rcpp::List & bead_el, double prior_scale, double prior_rate) :
        median(Rcpp::as<double>(bead_el["median"])),
        sum_d(Rcpp::as<double>(bead_el["sum_d"])),
        scale_prior_shape(prior_scale),
        scale_prior_rate(prior_rate),
        j(Rcpp::as<int>(bead_el["j"]))
{
    SEXP ll = bead_el["sort_x"];
    sort_x = Rcpp::NumericVector(ll);

    ll = bead_el["d"];
    d = Rcpp::NumericVector(ll);

    mu = median;
    mcmc_scale_mult = 3.0;
    n = sort_x.size();
    scale = sum_d / n;
}

double beadList::computeSumAbsDev(double & x) {
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

double beadList::computeLaplaceLikelihood(double & x) {
    return 0.0;
}

void beadList::updateLaplaceScale(RngStream & rng) {
    return;
}
