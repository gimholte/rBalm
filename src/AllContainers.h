#ifndef ALLCONTAINTERS_H
#define ALLCONTAINTERS_H

class BeadDist {
    Rcpp::NumericVector x_sorted;
    Rcpp::NumericVector abs_dev_from_median;

    // computation quantities
    int j;
    double x_median;
    double sum_abs_dev_median;

    //mcmc related quantities
    double mu;
    double precision;
    double sum_abs_dev_cur;
    double mcmc_scale_mult;
public:
    BeadDist() :
        x_sorted(0),
        abs_dev_from_median(0),
        j(0),
        x_median(0.0),
        sum_abs_dev_median(0.0),
        mu(0.0),
        precision(1.0),
        sum_abs_dev_cur(0.0),
        mcmc_scale_mult(1.0) {};

    BeadDist(const Rcpp::List & bead_dist);

    double computeLaplaceLikelihood(const double sum_abs_dev_x);

    double computeSumAbsDev(const double x);

    void updateLaplacePrecision(RngStream rng,
            const double prior_precision_shape, const double prior_precision_rate);

    void updateLaplaceMu(RngStream rng,
            const double mu_prior_mean, const double mu_prior_prec);

    inline double getMu() {
        return mu;
    }

    inline double getPrecision() {
        return precision;
    }
};

class ModelData {
    //R input checking stuff
    std::vector<std::string> r_names;
    void initializeListNames() {
        r_names.push_back("j");
        r_names.push_back("sum_abs_dev_median");
        r_names.push_back("x_median");
        r_names.push_back("x_sorted");
        r_names.push_back("abs_dev_from_median");
        return;
    };

    void fillY() {
        std::vector<BeadDist>::iterator it_bd;
        int i;
        for (i = 0, it_bd = bead_vec.begin();
                it_bd != bead_vec.end();
                ++i, ++it_bd) {
            y(i) = it_bd->getMu();
        }
    };

public:
    /* vector of bead distributions with elements corresponding to y */
    std::vector<BeadDist> bead_vec;
    int n_mfi;
    Eigen::VectorXd y;

    ModelData(SEXP r_model_data);
    void update(RngStream rng, Eigen::VectorXd & mu_prior_mean,
            Eigen::VectorXd & mu_prior_precision,
            const double precision_prior_shape,
            const double precision_prior_rate);
};

class ChainParameters {
    std::map<std::string, int> iter_pars;
    std::vector<std::string> r_names;
    void initializeListNames() {
        r_names.push_back("n_thin");
        r_names.push_back("n_iter");
        r_names.push_back("n_burn");
    };
public:
    ChainParameters(SEXP r_chain_parameters);

    int nIter() {
        return iter_pars["n_iter"];
    }

    int nThin() {
        return iter_pars["n_thin"];
    }

    int nBurn() {
        return iter_pars["n_burn"];
    }

    int totalIterations() {
        return iter_pars["n_burn"] + iter_pars["n_iter"] * iter_pars["n_thin"];
    }
};

class Latent {
    std::vector<std::string> r_names;

    // private helpers
    Eigen::VectorXd sum_y_w, sum_yy_w, yss_w, resid;

    void initializeListNames() {
        r_names.push_back("Zt");
        r_names.push_back("cnms");
    };

public:
    Eigen::SparseMatrix<double> Mt;
    Eigen::VectorXd gamma;
    Eigen::VectorXd mu_g;
    Eigen::VectorXd nu_g;
    Eigen::VectorXd counts;
    // public helpers (can depend on them being in sync with parameters)
    Eigen::VectorXd weights, fitted;

    Latent(SEXP r_latent_terms);
    void update(RngStream rng,
            const Eigen::VectorXd & data_y,
            const Eigen::VectorXd & linear_fitted,
            const Hypers & hypers);

    void updateMu(RngStream rng,
            const Eigen::VectorXd & data_y,
            const Eigen::VectorXd & linear_fitted,
            const Hypers & hypers);

    void updateNu(RngStream rng,
            const Eigen::VectorXd & data_y,
            const Eigen::VectorXd & linear_fitted,
            const Hypers & hypers);
};

class MHValues {
public:
    MHValues() :
        prop_to_cur(0.0),
        cur_to_prop(0.0),
        prop_prior(0.0),
        cur_prior(0.0),
        prop_lik(0.0),
        cur_lik(0.0) {
    }
        double logA() {
        return (prop_lik + prop_prior + prop_to_cur) -
                (cur_lik + cur_prior + cur_to_prop);
    }

    void setCurLik(double curLik) {
        cur_lik = curLik;
    }

    void setCurPrior(double curPrior) {
        cur_prior = curPrior;
    }

    void setCurToProp(double curToProp) {
        cur_to_prop = curToProp;
    }

    void setPropLik(double propLik) {
        prop_lik = propLik;
    }

    void setPropPrior(double propPrior) {
        prop_prior = propPrior;
    }

    void setPropToCur(double propToCur) {
        prop_to_cur = propToCur;
    }

    double getCurLik() const {
        return cur_lik;
    }

    double getCurPrior() const {
        return cur_prior;
    }

    double getCurToProp() const {
        return cur_to_prop;
    }

    double getPropLik() const {
        return prop_lik;
    }

    double getPropPrior() const {
        return prop_prior;
    }

    double getPropToCur() const {
        return prop_to_cur;
    }

private:
    double prop_to_cur;
    double cur_to_prop;
    double prop_prior;
    double cur_prior;
    double prop_lik;
    double cur_lik;
};

class CovarianceTemplate {
public:
    int within_theta_offset_val;
    int p;
    Eigen::VectorXd lower_tri;

    CovarianceTemplate(int offset_val_, int p_, double rho_ = 0.0);

    void proposeTheta(RngStream rng, Eigen::VectorXd & theta,
            MHValues & mhv);

    void acceptLastProposal();

    void fillTheta(Eigen::VectorXd & theta) {
        int t_ind, n = lower_tri.rows();
        for (int i = 0; i < n; i++) {
            t_ind = getThetaOffset() + i;
            theta[t_ind] = lower_tri[i];
        };
        return;
    };

    int getThetaOffset() const {
        return within_theta_offset_val;
    };

    void setThetaOffset(int k) {
        within_theta_offset_val = k;
    };

    Eigen::MatrixXd getDL() const {
        Eigen::MatrixXd newDL = DL;
        return DL;
    }

    void setSigma(Eigen::VectorXd & new_sig) {
        if (new_sig.rows() != sigma.rows()) {
            Rcpp::stop("invalid length of new sigma values");
        }
        sigma = new_sig;
        fillCholeskyDecomp();
    }

private:
    double rho, prop_rho;
    Eigen::VectorXd sigma, prop_sigma;
    Eigen::MatrixXd D;
    Eigen::MatrixXd L_cor;
    Eigen::MatrixXd DL;

    void fillCholeskyDecomp();
};


class LinearHelpers {
private:
    Eigen::VectorXi offset_b;
    Eigen::VectorXi offset_theta;
    Eigen::VectorXi offset_lambda;
    Eigen::VectorXi Lind;

    void setOffsetB(SEXP Gp_) {
        offset_b = Rcpp::as<Eigen::VectorXi>(Gp_);
    }

    void setOffsetTheta(SEXP ot_) {
        offset_theta = Rcpp::as<Eigen::VectorXi>(ot_);
    }

    void setOffsetLambda(const Eigen::SparseMatrix<double> & l_init,
            const Eigen::VectorXi & offset_b__) {
        int i, osb, nb, n;
        Eigen::SparseMatrix<double> tmp;
        for (i = 0, offset_lambda(0) = 0, n = offset_b__.rows() - 1; i < n; ++i) {
            osb = offset_b__(i);
            nb = offset_b__(i + 1) - offset_b__(i);
            tmp = l_init.block(osb, osb, nb, nb);
            offset_lambda(i + 1) = tmp.nonZeros() + offset_lambda(i);
        }
    }

    void setLambdaBlocks(const Eigen::SparseMatrix<double> l_init,
            const Eigen::VectorXi & offset_b__) {
        int nblock = offset_b__.rows() - 1;
        if (nblock < 1)
            return;
        int nb, osb;
        Eigen::SparseMatrix<double> tmp;
        block_lambdat.erase(block_lambdat.begin(), block_lambdat.end());
        for (int i = 0; i < nblock; ++i) {
            nb = offset_b__(i + 1) - offset_b__(i);
            osb = offset_b__(i);
            tmp = l_init.block(osb, osb, nb, nb);
            block_lambdat.push_back(tmp);
        }
        return;
    }

public:
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver;
    std::vector<Eigen::SparseMatrix<double> > block_lambdat;

    void initialize(SEXP linear_terms_, const Eigen::SparseMatrix<double> & solver_init,
            const Eigen::SparseMatrix<double> & lambda_init) {
        Rcpp::List linear_terms(linear_terms_);
        setOffsetB(linear_terms["Gp"]);
        setOffsetTheta(linear_terms["offset_theta"]);
        setOffsetLambda(lambda_init, offset_b);
        setLind(linear_terms["Lind"]);
        setLambdaBlocks(lambda_init, offset_b);
        solver.analyzePattern(solver_init);
    }


    int nB(int k) const {
        return offset_b(k + 1) - offset_b(k);
    }

    int getOffsetB(int k) const {
        return offset_b(k);
    }

    int nTheta(int k) const {
        return offset_theta(k + 1) - offset_theta(k);
    }

    int getOffsetTheta(int k) const {
        return offset_theta(k);
    }

    void setLind(SEXP Lind_) {
        Lind = Rcpp::as<Eigen::VectorXi>(Lind_);
        Lind.array() -= 1;
    }

    int getLind(int k) const {
        return Lind(k);
    }

    int numBlocks() const {
        return (int) block_lambdat.size();
    }

    double* getBlockValuePtr(int block_idx) {
        return block_lambdat[block_idx].valuePtr();
    }

    int numBlockNonZeros(int block_idx) const {
        return block_lambdat[block_idx].nonZeros();
    }

    int getOffsetLambda(int k) const {
        return offset_lambda(k);
    }

    int nLambda(int k) const {
        return offset_lambda(k + 1) - offset_lambda(k);
    }
};

class Linear {
    std::vector<std::string> r_names;
    void initializeListNames() {
        r_names.push_back("Zt");
        r_names.push_back("cnms");
        r_names.push_back("Lambdat");
        r_names.push_back("Gp");
        r_names.push_back("component_p");
        r_names.push_back("Lind");
        return;
    };

public:
    // parameters
    Eigen::VectorXd theta;
    Eigen::VectorXd b, u;
    std::vector<CovarianceTemplate> cov_templates;

    // fundamental matrices
    std::vector<Eigen::SparseMatrix<double> > Ztlist;
    Eigen::SparseMatrix<double> Lambdat, Zt, Omega, I_p, LtZt;

    // helpers for computation and avoiding ubiquitous temporaries
    LinearHelpers helpers;
    Eigen::VectorXd work_y_vec, work_theta_vec;

    // convenient things
    Eigen::VectorXd fitted;

    Linear(SEXP r_linear_terms);
    void update(RngStream rng,
            const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & mfi_obs_weights,
            const Eigen::VectorXd & data_y,
            const Hypers & hypers);

    void updateComponent(RngStream rng, const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & weights, const Eigen::VectorXd & data_y,
            const int k, const Eigen::SparseMatrix<double> & zt_component,
            Eigen::SparseMatrix<double> & lambdat_block,
            CovarianceTemplate & cvt);

    void setFitted() {
        b = Lambdat.transpose() * u;
        fitted = Zt.transpose() * b;
    }

    inline bool checkBlocks(const std::vector<Eigen::SparseMatrix<double> > blocks,
            const Eigen::SparseMatrix<double> L) {
        int nzL = L.nonZeros();
        int nzB(0);

        int cdimB(0);
        int rdimB(0);

        for(size_t i = 0; i < blocks.size(); ++i) {
            nzB += blocks[i].nonZeros();
            rdimB += blocks[i].rows();
            cdimB += blocks[i].cols();
        };

        if (nzB != nzL)
            return false;

        if (rdimB != L.rows())
            return false;

        if (cdimB != L.cols())
            return false;

        return true;
    }

    void setLambdaHelperBlock(Eigen::VectorXd & new_theta, int block_idx);
    void setLambdaBlock(Eigen::VectorXd & new_theta, int block_idx);
};

class Hypers {
    // vector of required list elements from R.
    std::vector<std::string> r_names;
    void initializeListNames() {
        r_names.push_back("gam0");
        r_names.push_back("gam1");
        r_names.push_back("gam01");
        r_names.push_back("lambda_a_prior");
        r_names.push_back("lambda_b_prior");
        r_names.push_back("bead_precision_rate");
        r_names.push_back("bead_precision_shape");
        return;
    };
public:
    // estimated
    double mfi_nu_shape;
    double mfi_nu_rate;
    double p;

    //fixed
    double gam0;
    double gam1;
    double gam01;
    double lambda_a_prior;
    double lambda_b_prior;
    double bead_precision_rate;
    double bead_precision_shape;

    Hypers(SEXP r_fixed_hyper);
    Hypers();
    void update(RngStream rng, Eigen::VectorXd & mfi_precision);

    inline double integratedPrecisionConditional(double x, int n,
            double sum_nu, double sum_log_nu) {
        if (x <= 0.0) {
            return -INFINITY;
        }
        return R::lgammafn(n * x + 1.0) - n * x * log(sum_nu + lambda_b_prior) +
                x * (sum_log_nu - lambda_a_prior) - n * R::lgammafn(x);
    };
};

void mvNormSim(RngStream rng, Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & solver,
        const Eigen::SparseMatrix<double> & omega, const Eigen::VectorXd & tau,
        Eigen::VectorXd & sample);

double thetaLikelihood(const Eigen::VectorXd & delta, const Eigen::VectorXd & weights);

#endif
