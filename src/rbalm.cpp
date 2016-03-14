#include "RcppArmadillo.h"
#include "Rcpp.h"
#include "RngStream.h"
#include "Rv.h"
#include "beadList.h"
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
    NumericVector eval(x_eval);
    NumericVector out(eval.size());
    std::vector<beadList> blist;

    List::iterator in_iter;
    for (in_iter = bead_list.begin(); in_iter != bead_list.end(); ++in_iter) {
        List ll(*in_iter);
        blist.push_back(beadList(ll, 1.0, 1.0));
    }

    if (eval.size() != bead_list.size())
        return wrap(R_NilValue);
    NumericVector::iterator it;
    int i;
    for (it = eval.begin(), i = 0; it != eval.end(); it++, i++) {
        out(i) = blist[i].computeSumAbsDev(*it);
    }
    return wrap(out);
}


