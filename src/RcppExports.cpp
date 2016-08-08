// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// bambaMcmc
SEXP bambaMcmc(SEXP r_bead_list, SEXP r_chain_pars, SEXP r_latent_reTrms, SEXP r_linear_reTrms, SEXP r_hyper_list, SEXP r_at_matrix, SEXP r_update_beads);
RcppExport SEXP rBalm_bambaMcmc(SEXP r_bead_listSEXP, SEXP r_chain_parsSEXP, SEXP r_latent_reTrmsSEXP, SEXP r_linear_reTrmsSEXP, SEXP r_hyper_listSEXP, SEXP r_at_matrixSEXP, SEXP r_update_beadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type r_bead_list(r_bead_listSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_chain_pars(r_chain_parsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_latent_reTrms(r_latent_reTrmsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_linear_reTrms(r_linear_reTrmsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_hyper_list(r_hyper_listSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_at_matrix(r_at_matrixSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_update_beads(r_update_beadsSEXP);
    __result = Rcpp::wrap(bambaMcmc(r_bead_list, r_chain_pars, r_latent_reTrms, r_linear_reTrms, r_hyper_list, r_at_matrix, r_update_beads));
    return __result;
END_RCPP
}
// testLinearHelpersInit
Rcpp::List testLinearHelpersInit(SEXP retrms_, Eigen::VectorXd tau);
RcppExport SEXP rBalm_testLinearHelpersInit(SEXP retrms_SEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type retrms_(retrms_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type tau(tauSEXP);
    __result = Rcpp::wrap(testLinearHelpersInit(retrms_, tau));
    return __result;
END_RCPP
}
// testLinearThetaFilling
Rcpp::List testLinearThetaFilling(SEXP retrms_);
RcppExport SEXP rBalm_testLinearThetaFilling(SEXP retrms_SEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type retrms_(retrms_SEXP);
    __result = Rcpp::wrap(testLinearThetaFilling(retrms_));
    return __result;
END_RCPP
}
// testConstructR
Rcpp::List testConstructR(SEXP retrms_, Eigen::VectorXd r_gamma);
RcppExport SEXP rBalm_testConstructR(SEXP retrms_SEXP, SEXP r_gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type retrms_(retrms_SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type r_gamma(r_gammaSEXP);
    __result = Rcpp::wrap(testConstructR(retrms_, r_gamma));
    return __result;
END_RCPP
}
// beadListTest
NumericVector beadListTest(SEXP x_bead_list, NumericVector x);
RcppExport SEXP rBalm_beadListTest(SEXP x_bead_listSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type x_bead_list(x_bead_listSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    __result = Rcpp::wrap(beadListTest(x_bead_list, x));
    return __result;
END_RCPP
}
// beadListMcmcTest
Rcpp::List beadListMcmcTest(SEXP r_bead_list, SEXP r_chain_ctl);
RcppExport SEXP rBalm_beadListMcmcTest(SEXP r_bead_listSEXP, SEXP r_chain_ctlSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type r_bead_list(r_bead_listSEXP);
    Rcpp::traits::input_parameter< SEXP >::type r_chain_ctl(r_chain_ctlSEXP);
    __result = Rcpp::wrap(beadListMcmcTest(r_bead_list, r_chain_ctl));
    return __result;
END_RCPP
}
// testSolver
Rcpp::List testSolver(SEXP r_linear_terms);
RcppExport SEXP rBalm_testSolver(SEXP r_linear_termsSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< SEXP >::type r_linear_terms(r_linear_termsSEXP);
    __result = Rcpp::wrap(testSolver(r_linear_terms));
    return __result;
END_RCPP
}
// testMvNormSim
Eigen::MatrixXd testMvNormSim(const Eigen::SparseMatrix<double> omega, Eigen::VectorXd tau, int n);
RcppExport SEXP rBalm_testMvNormSim(SEXP omegaSEXP, SEXP tauSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double> >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    __result = Rcpp::wrap(testMvNormSim(omega, tau, n));
    return __result;
END_RCPP
}
// testCholeskyFill
SEXP testCholeskyFill();
RcppExport SEXP rBalm_testCholeskyFill() {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    __result = Rcpp::wrap(testCholeskyFill());
    return __result;
END_RCPP
}
// logDeterminant
double logDeterminant(const Eigen::MatrixXd& S);
RcppExport SEXP rBalm_logDeterminant(SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type S(SSEXP);
    __result = Rcpp::wrap(logDeterminant(S));
    return __result;
END_RCPP
}
