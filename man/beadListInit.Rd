% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/initialize.R
\name{beadListInit}
\alias{beadListInit}
\title{Initialize bead list element for MCMC}
\usage{
beadListInit(bead_dists)
}
\arguments{
\item{bead_dists}{List of atomic double vectors containing bead measurements}
}
\value{
List of bead distributions, in the same order, with summary information
}
\description{
Given a list of bead distributions, process each list element
for efficient evaluation of the Laplace likelihood
}
\details{
doto
}

