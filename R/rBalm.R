#' @title Robust Bayesian analysis of Luminex bead-based assays
#' @description todo description
#' @details todo details
#' @param data todo
#' @param n_iter todo 
#' @param n_burn todo
#' 
#' @return some output
#' @export 
#' @useDynLib rBalm

rBalm <- function(data, n_iter, n_burn) {
    d_list <- list(x = rnorm(100))
    .Call("rBalmMcmc", d_list, PACKAGE = "rBalm")
}