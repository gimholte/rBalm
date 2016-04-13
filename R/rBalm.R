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


#' @title Helpers for parameterizing Lambda 
#' @description todo
#' @details mapping indicies to parameters
#' @param lform_output
#' @return list
#' @export 

createMappingHelpers <- function(lform_output) {
    np <- lengths(lform_output$cnms)
    ntheta <- choose(np + 1, 2)
    thoff <- unname(c(0, cumsum(ntheta)))
    list(component_p = np, offset_theta = thoff)
}
