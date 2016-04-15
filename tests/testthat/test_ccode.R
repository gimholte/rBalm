library("testthat")
library("rBalm")
context("C++ methods for linear model component")

test_that("LinearHelpers member functions initialize", {
            small_test_path <- system.file("extdata", "small_test.RData", package = "rBalm")
            load(small_test_path)
            lform <- lFormula(mfi ~ (0 + plate | subject) + (1 |treatment:subject),
                    data = small_mfi)$reTrms
            helpers <- createMappingHelpers(lform)
            lform <- c(lform, helpers)
            tau <- rnorm(nrow(lform$Lambdat))
            out <- tryCatch(testLinearHelpersInit(lform, tau), error = print)
            Zt <- lform$Zt
            Lambdat <- lform$Lambdat
            Omega <- Lambdat %*% Zt %*% t(Zt) %*% Matrix::t(Lambdat) + diag(1, nrow(Zt))
            expect_equivalent(out$tau_solve, drop(solve(Omega, tau)))
            expect_equivalent(object = as(.bdiag(out$helper_blocks), "dgCMatrix"),
                    expected = as(lform$Lambdat, "dgCMatrix"))
            expect_equivalent(object = c(0, cumsum(out$block_nonZeros)),
                    expected = out$offset_lambda)
            expect_equivalent(lform$theta[lform$Lind], .bdiag(out$helper_blocks)@x)
        })

test_that("Theta values update properly", {
            small_test_path <- system.file("extdata", "small_test.RData", package = "rBalm")      
            load(small_test_path)
            lform <- lFormula(mfi ~ (0 + plate | subject) + (1 |treatment:subject),
                    data = small_mfi)$reTrms
            helpers <- createMappingHelpers(lform)
            lform <- c(lform, helpers)
            out <- testLinearThetaFilling(lform);
            with(out, expect_equivalent(theta, c(template_L0[1], 
                            template_L1[lower.tri(template_L1, TRUE)])))
            expect_equivalent(as(.bdiag(list(out$Lambdat_block0, out$Lambdat_block1)), "dgCMatrix"),
                            as(out$Lambdat, "dgCMatrix"))
            L1 = diag(9, nrow = 2)
            L1[1, 2] = L1[2, 1] = 4.5;
            expect_equivalent(L1, tcrossprod(out$template_L1))
            expect_equivalent(out$template_L1, as.matrix(t(out$Lambdat_block1[1:2, 1:2])))
        })