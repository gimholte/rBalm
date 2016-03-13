#include "RcppArmadillo.h"
#include "Rcpp.h"
#include "Rv.h"

RcppExport SEXP rBalmMcmc(SEXP data_input)
   Rcpp::List data_list(data_input)
{
    return data_list;
}
