#include <RcppEigen.h>
#include <Rcpp.h>
#include "RngStream.h"
#include "Rv.h"
#include "beadDist.h"
#include <vector>

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MappedSparseMatrix;

typedef Eigen::SparseMatrix< double > SpMat;
typedef Eigen::Map< Eigen::MatrixXd > MapMatd;


class SimpleBalm {
public:
    // data
    VectorXd y;         // vector of MFI values
    VectorXd weight_y; // vector filled with precision for corresponding y value
    VectorXd mean_y;    // vector filled with overall mean for corresp. y value

    const MappedSparseMatrix<double> Zt;    // sparse design matrix giving subject/treatment combinations
    SpMat Z;   // transpose of Z
    int n_subject;
    int n_mfi;
    VectorXd counts;
    std::vector<beadDist> blist;     /* vector of bead distributions
                                        with elements corresponding to y */
    // model parameters
    VectorXd mu_y;  // means of mfi values for subject/trt combinations
    VectorXd nu_y;  // precisions of mfi values for subject/trt combinations
    double a0, b0, a1, b1; // means, variances of gamma prior on precisions
    double gam01, gam0, gam1; // prior precision on means of mfi values
    double p;       // proportion of responses

    // helpers
    RngStream rng;
    VectorXd unit;
    VectorXd sum_y;
    VectorXd sum_y_w;
    VectorXd sum_yy_w;
    VectorXd yss;

    // trace
    MatrixXd trace_bead_mu;
    MatrixXd trace_bead_prec;
    MatrixXd trace_mu_y;
    MatrixXd trace_nu_y;

    // mcmc parameters
    int n_iter;
    int n_thin;
    int n_burn;

    // member functions
    SimpleBalm(SEXP r_bead_list, SEXP r_Zt, SEXP r_chain_pars);

    inline void update_y() {
        for (int i = 0; i < n_mfi; i++) {
            y(i) = blist[i].getMu();
        }
    }

    inline void fill_mean_y() {
        mean_y.noalias() = Z * mu_y;
    }

    inline void fill_weight_y() {
        weight_y.noalias() = Z * nu_y;
    }

    inline void fill_sum_y_w() {
        sum_y_w.noalias() = Zt * (y.cwiseProduct(weight_y));
        sum_yy_w.noalias() = Zt * (y.cwiseProduct(y).cwiseProduct(weight_y));
    }
    inline void fill_yss() {
        yss.noalias() = Zt * ((y - mean_y).cwiseProduct(y - mean_y));
    }

    void iterate();
    void updateBeadDists();
    void updateMu();
    void updateNu();
    void gatherTrace(const int i);
};

SimpleBalm::SimpleBalm(SEXP r_bead_list, SEXP r_Zt, SEXP r_chain_pars) :
    Zt(as<MappedSparseMatrix<double> >(r_Zt))
{
    List bead_list(r_bead_list);
    List bead_prior = List::create(_["prior_shape"] = 1.0, _["prior_rate"] = 1.0);
    parseBeadDistList(bead_list, blist, bead_prior);



    n_mfi = bead_list.size();
    Z = Zt.adjoint();
    y = VectorXd::Constant(n_mfi, 0);
    weight_y = VectorXd::Constant(n_mfi, 0);
    mean_y = VectorXd::Constant(n_mfi, 0);
    n_subject = Zt.rows() / 2;
    mu_y = VectorXd::Constant(n_subject * 2, 0);
    nu_y = VectorXd::Constant(n_subject * 2, 0);

    a0 = 05;
    b0 = .05;
    a1 = .05;
    b1 = .05;
    gam01 = .05;
    gam0 = .05;
    gam1 = .05;
    p = .05;

    // helpers
    rng = RngStream_CreateStream("");
    unit = VectorXd::Constant(n_mfi, 1);
    counts = Zt * unit;
    sum_y_w = VectorXd::Constant(2 * n_subject, 0);
    sum_yy_w = VectorXd::Constant(2 * n_subject, 0);
    this->update_y();

    // mcmc stuff
    List chain_pars(r_chain_pars);
    n_iter = chain_pars["n_iter"];
    n_thin = chain_pars["n_thin"];
    n_burn = chain_pars["n_burn"];

    trace_bead_mu.resize(n_mfi, n_iter);
    trace_bead_prec.resizeLike(trace_bead_mu);
    trace_mu_y.resize(2 * n_subject, n_iter);
    trace_nu_y.resizeLike(trace_mu_y);
}

void SimpleBalm::iterate() {
    this->updateBeadDists();
    this->updateMu();
    this->updateNu();
    return;
}

void SimpleBalm::updateBeadDists() {
    this->fill_mean_y();
    this->fill_weight_y();
    for (int j = 0; j < n_mfi; j++) {
        blist[j].updateLaplaceMu(rng, mean_y(j), weight_y(j));
        blist[j].updateLaplacePrecision(rng);
    }
    this->update_y();
    return;
}

void SimpleBalm::updateMu() {
    int i, j, jp1;
    double c0, c1, c01;
    double l0, l1, l01;
    double mu0, mu1, mu01;
    double null_prob;
    this->fill_weight_y();
    this->fill_sum_y_w();
    for (i = 0; i < n_subject; i++) {
        j = 2 * i;
        jp1 = j + 1;
        l0 = gam0 + counts(j) * nu_y(j);
        l1 = gam1 + counts(jp1) * nu_y(jp1);
        l01 = gam01 + counts(j) * nu_y(j) + counts(jp1) * nu_y(jp1);

        mu0 = sum_y_w(j) / l0;
        mu1 = sum_y_w(jp1) / l1;
        mu01 = (sum_y_w(j) + sum_y_w(jp1)) / l01;

        c0 = -.5 * nu_y(j) * sum_yy_w(j);
        c0 += .5 * pow(sum_y_w(j), 2) / l0;
        c0 += .5 * log(gam0) -.5 * log(l0);

        c1 = -.5 * nu_y(jp1) * sum_yy_w(jp1);
        c1 += .5 * pow(sum_y_w(jp1), 2) / l1;
        c1 += .5 * log(gam1) -.5 * log(l1);

        c01 = -.5 * nu_y(j) * sum_yy_w(j);
        c01 += -.5 * nu_y(jp1) * sum_yy_w(jp1);
        c01 += .5 * pow(sum_y_w(j) + sum_y_w(jp1), 2) / l01;
        c01 += .5 * log(gam01) -.5 * log(l01);

        null_prob = p / (p * exp(c0 + c1 - c01) + (1 - p));
        if (RngStream_RandU01(rng) < null_prob) {
            mu01 = RngStream_N01(rng) / sqrt(l01) + mu01;
            mu_y(j) = mu01;
            mu_y(jp1) = mu01;
        } else {
            mu_y(j) = RngStream_N01(rng) / sqrt(l0) + mu0;
            mu_y(jp1) = RngStream_N01(rng) / sqrt(l1) + mu1;
        }
    }
    return;
}

void SimpleBalm::updateNu() {
    int i, j;
    double joint_rate;
    double joint_shape;
    // update the mean vector for y
    this->fill_mean_y();
    this->fill_yss();
    for (i = 0; i < n_subject; i++) {
        j = 2 * i;
        joint_shape = a0 + counts(j);
        joint_rate = b0 + yss(j);
        nu_y(j) = RngStream_GA1(joint_shape, rng) / joint_rate;

        j = 2 * i + 1;
        joint_shape = a1 + counts(j);
        joint_rate = b1 + yss(j);
        nu_y(j) = RngStream_GA1(joint_shape, rng) / joint_rate;
    }
    return;
}

void SimpleBalm::gatherTrace(const int i) {
    trace_bead_mu.col(i) = y;
    for (int j = 0; j < n_mfi; j++) {
        trace_bead_prec(j, i) = blist[j].getPrecision();
    }
    trace_mu_y.col(i) = mu_y;
    trace_nu_y.col(i) = nu_y;
    return;
}

// [[Rcpp::depends(RcppEigen)]]
RcppExport SEXP rBalmMcmc(SEXP r_bead_list, SEXP r_Zt, SEXP r_chain_pars) {
    SimpleBalm model(r_bead_list, r_Zt, r_chain_pars);
    for (int i = 0; i < model.n_burn; i++) {
        model.iterate();
    }
    int k = 0;
    for (int j = 1; j <= model.n_thin * model.n_iter; j++) {
        model.iterate();
        if (j % model.n_thin == 0) {
            model.gatherTrace(k);
            k++;
        }
    }
    return List::create(_["mu_y"] = wrap(model.trace_mu_y),
            _["nu_y"] = wrap(model.trace_nu_y),
            _["mu_bead"] = wrap(model.trace_bead_mu),
            _["prec_bead"] = wrap(model.trace_bead_prec));
}

// [[Rcpp::depends(RcppEigen)]]
RcppExport SEXP rBalmMSP(SEXP r_Zt) {
    const MappedSparseMatrix<double> Zt(as<MappedSparseMatrix<double> >(r_Zt));
    return wrap(Zt);
}

/*
 * Evaluate the sum of absolute deviations a list of bead distributions
 * x_bead_list: R list of processed bead distributions
 * x_eval: R numeric vector of values at which to evaluate the SADs
 */
RcppExport SEXP beadListTest(SEXP x_bead_list, SEXP x_eval) {
    List bead_list(x_bead_list);
    List bead_prior = List::create(_["prior_shape"] = 1.0, _["prior_rate"] = 1.0);
    NumericVector eval(x_eval);
    NumericVector out(eval.size());
    std::vector<beadDist> blist;
    parseBeadDistList(bead_list, blist, bead_prior);

    if (eval.size() != bead_list.size())
        return wrap(R_NilValue);
    NumericVector::iterator it;
    int i;
    for (it = eval.begin(), i = 0; it != eval.end(); it++, i++) {
        out(i) = blist[i].computeSumAbsDev(*it);
    }
    return wrap(out);
}

RcppExport SEXP beadListMcmcTest(SEXP r_bead_list, SEXP r_chain_ctl,
        SEXP r_bead_prior_pars) {
    RngStream rng = RngStream_CreateStream("");
    List bead_list(r_bead_list);
    List chain_ctl(r_chain_ctl);
    List bead_prior_pars(r_bead_prior_pars);

    int n_iter = as<int>(chain_ctl["n_iter"]);
    int n_bead_dist = bead_list.size();

    Dimension out_dim(n_bead_dist, n_iter);
    NumericMatrix out_mu(out_dim);
    NumericMatrix out_prec(out_dim);

    std::vector<beadDist> bead_distributions;
    parseBeadDistList(bead_list, bead_distributions, bead_prior_pars);

    for(int i = 0; i < n_iter; i++) {
        for (int j = 0; j < n_bead_dist; j++) {
            bead_distributions[j].updateLaplaceMu(rng, 1.0, 1.0);
            bead_distributions[j].updateLaplacePrecision(rng);
            out_mu(j, i) = bead_distributions[j].getMu();
            out_prec(j, i) = bead_distributions[j].getPrecision();
        }
    }
    return wrap(List::create(_["mu"] = out_mu, _["prec"] = out_prec));
}
