library(rBalm)

set.seed(123)
n <- c(50, 51, 100, 200, 505)
beads <- lapply(n, rnorm, mean = 0, sd = 1)
processed_beads <- beadListInit(beads)

test_that("bead distributions have proper named attributes", {
            expect_true(all(names(processed_beads[[1]]) %in%
                                    c("sort_x", "sum_d", "d", "j", "median")))
        })

beadListTest <- function(beads, at) {
    out <- .Call("beadListTest", beads, at, PACKAGE = "rBalm")
    return(out)
}

eval_at <- c(-100, .5, 0, .5, 100)
sum_abs_devs <- mapply(function(vec, eval) sum(abs(vec - eval)), beads, eval_at)
sum_abs_devs_cpp <- beadListTest(processed_beads, eval_at)

test_that("sum of absolute deviations are correct", {
    expect_equal(sum_abs_devs_cpp[1], sum_abs_devs[1])
    expect_equal(sum_abs_devs_cpp[2], sum_abs_devs[2])
    expect_equal(sum_abs_devs_cpp[3], sum_abs_devs[3])
    expect_equal(sum_abs_devs_cpp[4], sum_abs_devs[4])
    expect_equal(sum_abs_devs_cpp[5], sum_abs_devs[5])
})


beads <- lapply(n, rpois, lambda = 2)
processed_beads <- beadListInit(beads)

eval_at <- c(-1, 0, 1, 3, 9)
sum_abs_devs <- mapply(function(vec, eval) sum(abs(vec - eval)), beads, eval_at)
sum_abs_devs_cpp <- beadListTest(processed_beads, eval_at)

test_that("sum of absolute deviations are correct (with ties)", {
            expect_equal(sum_abs_devs_cpp[1], sum_abs_devs[1])
            expect_equal(sum_abs_devs_cpp[2], sum_abs_devs[2])
            expect_equal(sum_abs_devs_cpp[3], sum_abs_devs[3])
            expect_equal(sum_abs_devs_cpp[4], sum_abs_devs[4])
            expect_equal(sum_abs_devs_cpp[5], sum_abs_devs[5])
        })

