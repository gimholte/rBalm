library(rBalm)
context("Bead sampler construction and usage")

test_that("bead distributions have proper named attributes", {
            set.seed(123)
            n <- c(50, 51, 100, 200, 505)
            beads <- lapply(n, rnorm, mean = 0, sd = 1)
            processed_beads <- beadListInit(beads)
            
            expect_true(all(names(processed_beads[[1]]) %in%
                                    c("x_sorted", "sum_abs_dev_median", "abs_dev_from_median",
                                            "j", "x_median")))
        })

test_that("sum of absolute deviations are correct", {
            set.seed(123)
            n <- c(50, 51, 100, 200, 505)
            beads <- lapply(n, rnorm, mean = 0, sd = 1)
            eval_at <- c(-100, .5, 0, .5, 100)            

            processed_beads <- beadListInit(beads)                        
            sum_abs_devs <- mapply(function(vec, eval) sum(abs(vec - eval)), beads, eval_at)
            sum_abs_devs_cpp <- beadListTest(processed_beads, eval_at)
            expect_equivalent(sum_abs_devs_cpp, sum_abs_devs)
        })

test_that("sum of absolute deviations are correct (with ties)", {
            set.seed(123)
            n <- c(50, 51, 100, 200, 505)
            eval_at <- c(-1, 0, 1, 3, 9)
            beads <- lapply(n, rpois, lambda = 2)
            
            processed_beads <- beadListInit(beads)
            sum_abs_devs <- mapply(function(vec, eval) sum(abs(vec - eval)), beads, eval_at)
            sum_abs_devs_cpp <- beadListTest(processed_beads, eval_at)
            expect_equivalent(sum_abs_devs_cpp, sum_abs_devs)
        })

test_that("markov chain monte carlo successfully executes", {
            set.seed(123)
            n_dists <- 200
            n <- rpois(n_dists, lambda = 200)
            n_iter <- 1000
            beads <- lapply(n, rnorm, mean = 0, sd = 1)
            processed_beads <- beadListInit(beads);
            chain_ctl <- list("n_iter" = n_iter,
                    "n_thin" = 1,
                    "n_burn" = 100)            
            out <- beadListMcmcTest(processed_beads, chain_ctl)
            expect_equivalent(dim(out[[1]]), c(n_iter, n_dists))
            expect_equivalent(dim(out[[2]]), c(n_iter, n_dists))
        })

