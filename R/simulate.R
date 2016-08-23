simulateFromPrior <- function(n_group, n_rep_per_group,
        n_bead, n_analyte, prior_control) {
    require(data.table)
    hyp <- mergeHyperLists(prior_control)
    
    overall_prec <- with(hyp, {
                rgamma(1, tau_prior_shape, rate = tau_prior_rate)
            })
    
    overall_mu <- with(hyp, {
                rnorm(1, mu_overall_bar, sd = 1 / sqrt(tau))
            })
    
    delta_sig <- with(hyp, {
                sig_delta_scale * abs(rt(1, df = 1))
            })
        
    group <- paste0("Group_", 1:n_group)
    analyte <- paste0("Analyte_", LETTERS[1:n_analyte])
    trt <- paste0("Trt_", c("A", "B"))    
    well <- paste0("Well_", 1:(n_rep_per_group * 2))
    
    latent_p <- with(hyp, {
                p <- rbeta(n_analyte, shape1 = p_alpha,
                        shape2 = p_beta)
            })
    names(latent_p) <- analyte
    
    latent_ind <- sapply(latent_p, function(resp_prob_by_analyte) {
                rbinom(n_group, size = 1, prob = resp_prob_by_analyte)
            })
    latent_ind <- as.vector(latent_ind)
    names(latent_ind) <- outer(group, analyte, paste, sep = ":")
    
    group_sd <- with(hyp, {
                sig <- folded_t_scale * rt(n_group * n_analyte, df = folded_t_df) +
                        folded_t_location
                abs(sig)
        })

    group_mean <- with(hyp, {
                m0 <- rnorm(n_group * n_analyte, mean = overall_mu, sd = 1 / sqrt(overall_prec))
                #m0 <- rep(0, n_group * n_analyte)
                delta <- rnorm(n_group * n_analyte, mean = 0, sd = delta_sig)
                m <- rep(m0, each = 2)
                m <- m + c(-.5, .5) * rep(delta, each = 2) * rep(latent_ind, each = 2)
                mu_g <- m
                mu_g[seq(1, length(mu_g), by = 2)] <- m0
                mu_g[seq(2, length(mu_g), by = 2)] <- delta * latent_ind
                list(m = m, mu_g = mu_g, delta = delta)
            })

    mfi <- mapply(FUN = function(mean, s) {
                rnorm(n_rep_per_group, mean = mean, sd = s)
            }, group_mean$m, group_sd)

    mfi_frame <- data.table(mfi = as.vector(mfi), group = rep(rep(group, each = n_rep_per_group * 2), n_analyte),
            analyte = rep(analyte, each = n_group * n_rep_per_group * 2), 
            trt = rep(rep(trt, each = n_rep_per_group), n_analyte * n_group),
            well = rep(well, n_analyte * n_group),
            latent = rep(latent_ind, each = 2 * n_rep_per_group))
    mfi_frame$laplace_scale_b <- rep(.25, nrow(mfi_frame))

    bead_frame <- mfi_frame[, .(fl = {
                e1 <- rexp(n_bead)
                e2 <- rexp(n_bead)
                exp(mfi + laplace_scale_b * (e1 - e2)) - 1
            }), by = "group,analyte,trt,well,latent"]
    
    delta <- group_mean$delta * as.vector(latent_ind)
    names(delta) <- names(latent_ind)
    
    return(list(mfi_frame = mfi_frame, bead_frame = bead_frame, latent_p = latent_p,
                    latent_ind = latent_ind, overall_prec = overall_prec,
                    group_mean = data.table(m = group_mean$m, mu_g = group_mean$mu_g,
                            group_sd,
                            analyte = rep(analyte, each = 2 * n_group),
                            group = rep(rep(group, each = 2), n_analyte),
                            trt = rep(trt, n_analyte * n_group)),
                    delta = delta, delta_sig = delta_sig,
                    overall_mean = overall_mu))
}


