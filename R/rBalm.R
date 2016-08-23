#' @title Robust Bayesian analysis of Luminex bead-based assays
#' @description todo description
#' @details todo details
#' @param latent effect formula
#' @param technical effects random effect formula
#' @param mfi_by by statement for reducing data table
#' @param data data frame
#' @param n_iter number of samples to collect
#' @param n_thin thinning interval between samples
#' @param n_burn burn in samples
#' @param update_beads should bead-level parameters be updated?
#' @param prior_control list of named hyperparameters and values
#' 
#' @return some output
#' @export 
#' @import lme4
#' @import Matrix
#' @import data.table
#' @useDynLib rBalm

rBalm <- function(latent, tech = NULL, mfi_by, data,
        n_sample, n_thin, n_burn, update_beads = TRUE,
        prior_control = NULL, offset = 0) {
    chain <- list(n_iter = n_sample, n_thin = n_thin, n_burn = n_burn)
    chain[["update_beads"]] = TRUE
    chain[["update_nu_prior"]] = TRUE
    setkeyv(data, strsplit(mfi_by, ",")[[1]])
    mfi_simp_dat <- data[, list(mfi = as.double(median(log1p(fl)) - offset),
                    n = .N),
            by = eval(mfi_by)]    
    i <- mfi_simp_dat[data, which = TRUE]
    bead_dist_list <- beadListInit(split(log1p(data$fl) - offset, f = i))
    lform_latent = lFormula(latent, data = mfi_simp_dat)$reTrms 
    at_matrix <- makeAnalyteFrame(lform_latent)
    lform_tech = NULL
    if (!is.null(tech)) {
        lform_tech = lFormula(tech, data = mfi_simp_dat)$reTrms
        helpers <- createMappingHelpers(lform_tech)    
        lform_tech = c(lform_tech, helpers)
    }
    hyper_list <- mergeHyperLists(prior_control)
    lform_latent$var_prior = hyper_list$var_prior
    Dt <- makeDeltaModelMatrix(lform_latent$Zt)
    lform_latent$Dt <- Dt
    # C call
    output <- bambaMcmc(bead_dist_list, chain, lform_latent, lform_tech, 
            hyper_list, at_matrix, update_beads)    
    
    group_id <- rownames(lform_latent$Zt)[seq(1, nrow(lform_latent$Zt), by = 2)]
    names(output$ppr) <- group_id
    return(list(trace = output, mfi_dat = mfi_simp_dat, 
                    lform_tech = lform_tech, 
                    lform_latent = lform_latent,
                    ppr = output$ppr))    
}

rBalm_cInput <- function(latent, tech = NULL, mfi_by, data,
        n_sample, n_thin, n_burn, update_beads = TRUE,
        prior_control = NULL, offset = 0) {
    chain <- list(n_iter = n_sample, n_thin = n_thin, n_burn = n_burn)
    chain[["update_beads"]] = TRUE
    chain[["update_nu_prior"]] = TRUE
    setkeyv(data, strsplit(mfi_by, ",")[[1]])
    mfi_simp_dat <- data[, list(mfi = as.double(median(log1p(fl)) - offset),
                    n = .N),
            by = eval(mfi_by)]    
    i <- mfi_simp_dat[data, which = TRUE]
    bead_dist_list <- beadListInit(split(log1p(data$fl) - offset, f = i))
    lform_latent = lFormula(latent, data = mfi_simp_dat)$reTrms 
    at_matrix <- makeAnalyteFrame(lform_latent)
    lform_tech = NULL
    if (!is.null(tech)) {
        lform_tech = lFormula(tech, data = mfi_simp_dat)$reTrms
        helpers <- createMappingHelpers(lform_tech)    
        lform_tech = c(lform_tech, helpers)
    }
    hyper_list <- mergeHyperLists(prior_control)
    lform_latent$var_prior = hyper_list$var_prior
    Dt <- makeDeltaModelMatrix(lform_latent$Zt)
    lform_latent$Dt <- Dt
    return(list(lform_tech = lform_tech, lform_latent = lform_latent,
                    bead_dist_list = bead_dist_list, hyper_list = hyper_list))    
}

#' @title Helpers for parameterizing Lambda 
#' @description todo
#' @details mapping indicies to parameters
#' @param lform_output
#' @return list
createMappingHelpers <- function(lform_output) {
    np <- lengths(lform_output$cnms)
    ntheta <- choose(np + 1, 2)
    thoff <- unname(c(0, cumsum(ntheta)))
    list(component_p = np, offset_theta = thoff)
}

makeAnalyteFrame <- function(lf_lat) {
    cnms <- names(lf_lat$cnms)
    k_analyte <- which(tolower(strsplit(cnms, split = ":")[[1]]) == "analyte")
    analyte_id_all <- sapply(strsplit(rownames(lf_lat$Zt), split = ":"), `[[`, k_analyte)
    analyte <- as.factor(analyte_id_all[seq(1, nrow(lf_lat$Zt), by = 2)])
    t(as(model.matrix(~ 0 + analyte), "dgCMatrix"))
}

makeDeltaModelMatrix <- function(Mt) {
    npair <- nrow(Mt) / 2
    Dt <- Mt
    for (i in 1:npair) {
        j <- 2 * i - 1
        jp1 <- 2 * i
        Dt[j,] <- Dt[j,] + Mt[jp1,]
        Dt[jp1,] <- .5 * Dt[jp1,]
        Dt[jp1,] <- Dt[jp1,] - .5 * Mt[j,]
    }
    return(Dt)
}

mergeHyperLists <- function(user_input = NULL) {
    # defaultHyperList supplies a named list with default initial values for each
    # hyperparameter in the model
    hyp_merge <- defaultHyperList()
    if (is.null(user_input)) {
        return(hyp_merge)
    }
    # some basic input checking
    if (is.null(names(user_input)) | 
            (!is.vector(user_input) & !is.list(user_input))) {
        stop("hyperparameter input must be a named vector or list")
    }
    user_input_list <- as.list(user_input)
    # check that all entries are numeric
#    if (!all(sapply(user_input_list, is.numeric))) {
#        stop("hyperparameter inputs must be numeric")
#    }
#    
    hyp_names <- names(hyp_merge)
    user_input_names <- names(user_input_list)
    # check that all names are valid
    if (!all(user_input_names %in% hyp_names)) {
        stop("invalid parameter names in supplied hyperparameter input")
    }
    # 
    idx <- user_input_names %in% hyp_names
    names_to_replace <- user_input_names[idx]
    hyp_merge[names_to_replace] <- user_input_list[names_to_replace]
    return(hyp_merge)
}

defaultHyperList <- function() {
    hyp <- list(   # prior precision of mu_g
            tau_prior_shape = 4.0,
            tau_prior_rate = 64.0,
            tau = 1.0 / 16.0,
            
            # prior mean of the mean of mu_g
            mu_overall_bar = 0.0,
         
            # prior weight of mu_g
            n_0 = .05,
            
            # prior shapes of beta distribution parameters on p_a
            p_alpha = .05,
            p_beta = .5,
            
            # rates for exponential priors on a, b
            lambda_a_prior = .0005,
            lambda_b_prior = .0005,
            folded_t_scale = .05,
            folded_t_location = .1,
            folded_t_df = 5.0,
            
            # bead precision scale parameter priors
            bead_precision_shape = .5,
            bead_precision_rate = .5,
            
            # shape and rate for theta
            theta_shape = .5,
            theta_rate = .5,
            
            # shape and rate for precision parameter of U vector
            var_prior = "gamma",
            
            prec_mean_prior_mean = 50,
            prec_mean_prior_sd = 100, 
            prec_var_prior_mean = 50 * 50, 
            prec_var_prior_sd = 2500,
    
            sig_delta = 1,
            sig_delta_scale = 4.0
        )
    return(hyp)
}




