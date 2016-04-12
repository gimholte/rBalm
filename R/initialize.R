#' @title Initialize data structures for rBalm algorithm
#' @description todo
#' @details internal
#' @param input A data.table with bead data
#' @param control A list with MCMC parameters
#' 
#' 
#' 
#' @return List with required rBalm structures 
#' @export 


rBalmInit <- function(input, control) {
    return(NULL)
}

#' @title Initialize bead list element for MCMC
#'  
#' @description Given a list of bead distributions, process each list element
#' for efficient evaluation of the Laplace likelihood 
#' 
#' @details doto
#' 
#' @param bead_dists List of atomic double vectors containing bead measurements
#' @return List of bead distributions, in the same order, with summary information
#' 
#' @export 

beadListInit <- function(bead_dists) {
    out <- lapply(bead_dists, processBeadVector)
    return(out)
}

#' @title Process a bead distribution
#' 
#' @description Summary statistics and various quantities 
#' needed for Laplace density evaluation
#' 
#' @param x bead values
#' @details The list is named 
#' 
#' @return Return a named list with elements
#' @export

processBeadVector <- function(x) {
    sort_x <- sort(x)    
    med <- median(sort_x)
    j <- max(which(sort_x <= med))
    d <- abs(sort_x - med)
    sum_d <- sum(d)

    out <- list(sort_x, med, j, d, sum_d)
    names(out) <- c("x_sorted", "x_median", "j", "abs_dev_from_median", "sum_abs_dev_median")
    return(out)
}
