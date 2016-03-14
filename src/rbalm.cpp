#include "RcppArmadillo.h"
#include "Rcpp.h"
#include "RngStream.h"
#include "Rv.h"
#include "beadDist.h"
#include <vector>
using namespace Rcpp;

RcppExport SEXP rBalmMcmc(SEXP data_input) {
    List data_list(data_input);
    return data_list;
}

/*
 * Evaluate the sum of absolute deviations a list of bead distributions
 * x_bead_list: R list of processed bead distributions
 * x_eval: R numeric vector of values at which to evaluate the SADs
 */
RcppExport SEXP beadListTest(SEXP x_bead_list, SEXP x_eval) {
    List bead_list(x_bead_list);
    List bead_prior = List::create(_["prior_shape"] = 1.0, _["prior_rate"] = 1.0);
    NumericVector eval(x_eval);
    NumericVector out(eval.size());
    std::vector<beadDist> blist;
    parseBeadDistList(bead_list, blist, bead_prior);

    if (eval.size() != bead_list.size())
        return wrap(R_NilValue);
    NumericVector::iterator it;
    int i;
    for (it = eval.begin(), i = 0; it != eval.end(); it++, i++) {
        out(i) = blist[i].computeSumAbsDev(*it);
    }
    return wrap(out);
}

RcppExport SEXP beadListMcmcTest(SEXP r_bead_list, SEXP r_chain_ctl,
        SEXP r_bead_prior_pars) {
    RngStream rng = RngStream_CreateStream("");
    List bead_list(r_bead_list);
    List chain_ctl(r_chain_ctl);
    List bead_prior_pars(r_bead_prior_pars);

    int n_iter = as<int>(chain_ctl["n_iter"]);
    int n_bead_dist = bead_list.size();

    Dimension out_dim(n_bead_dist, n_iter);
    NumericMatrix out_mu(out_dim);
    NumericMatrix out_prec(out_dim);

    std::vector<beadDist> bead_distributions;
    parseBeadDistList(bead_list, bead_distributions, bead_prior_pars);

    for(int i = 0; i < n_iter; i++) {
        for (int j = 0; j < n_bead_dist; j++) {
            bead_distributions[j].updateLaplaceMu(rng, 1.0, 1.0);
            bead_distributions[j].updateLaplacePrecision(rng);
            out_mu(j, i) = bead_distributions[j].getMu();
            out_prec(j, i) = bead_distributions[j].getPrecision();
        }
    }
    return wrap(List::create(_["mu"] = out_mu, _["prec"] = out_prec));
}
