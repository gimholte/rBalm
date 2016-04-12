#ifndef ALLCONTAINTERSFORWARD_H
#define ALLCONTAINTERSFORWARD_H

class BeadDist;
class ModelData;
class Linear;
class Latent;
class Hypers;
class CovarianceTemplate;

namespace Rcpp {
    template<> BeadDist as(SEXP) ;
};

#endif
