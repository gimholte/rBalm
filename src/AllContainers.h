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
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > Omega_bar_solver;
    Eigen::SparseMatrix<double> Rt;
    Eigen::VectorXd G_diag, tau, mu_compressed, prior_tau;
    Eigen::SparseMatrix<double> Omega_bar, V, G;

    Eigen::SparseMatrix<double> Mt;
    Eigen::VectorXd gamma;
    Eigen::VectorXd mu_g;
    Eigen::VectorXd nu_g;
    Eigen::VectorXd counts;
    // public helpers (can depend on them being in sync with parameters)
    Eigen::VectorXd weights, fitted;
    std::string var_prior;

    Latent(SEXP r_latent_terms);

    void updateMu(RngStream rng,
            const Eigen::VectorXd & y_tilde,
            const Hypers & hypers);

    void updateNu(RngStream rng,
            const Eigen::VectorXd & y_tilde,
            const Hypers & hypers);

    void updateMuMarginal(RngStream rng,
            Linear & linear,
            const Eigen::VectorXd & data_y,
            const Hypers & hyp);

    void constructRG(double g01, double g0, double g1,
            double m_c, double m_0, double m_1);
    void expandMu();
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

class MHtune {
private:
    int n_burn, n_check, num_iter, num_accept, total_accept, p;
    double running_stat_count, mcmc_scale, mcmc_ar_target;
    Eigen::MatrixXd par_cov, par_tmp;
    Eigen::VectorXd par_mean, mean_tmp1;

public:
    Eigen::MatrixXd par_L;
    MHtune(int n_burn_, int par_dim, double ar_target = .25) :
        n_burn(n_burn_),
        n_check(50),
        num_iter(0),
        num_accept(0),
        total_accept(0),
        p(par_dim),
        running_stat_count(0.0),
        mcmc_scale(1.0),
        mcmc_ar_target(ar_target),
        par_cov(p, p),
        par_tmp(p, p),
        par_L(p, p),
        par_mean(p),
        mean_tmp1(p)
    { };

    void updateRunningStats(const Eigen::VectorXd & sample) {
        double n = running_stat_count;
        double np1 = n + 1.0;
        if (n == 0) {
            par_cov.setZero();
            par_L.setZero();
            par_mean = sample;
            running_stat_count += 1.0;
            return;
        }
        mean_tmp1.noalias() = par_mean + (sample - par_mean) / np1;
        par_cov *= n / (double) np1;
        par_cov.noalias() += (sample - mean_tmp1) * (sample - par_mean).transpose() / np1;
        par_mean = mean_tmp1;
        running_stat_count += 1.0;
        return;
    }

    void updateMcmcScale() {
        const double a_prop = ((double) num_accept) / n_check;
        //Rcpp::Rcout << mcmc_scale << std::endl;
         if (a_prop < fmax(mcmc_ar_target - .05, .05)) {
             mcmc_scale *= 1.0 / 1.1;
         } else if (a_prop > fmin(mcmc_ar_target + .05, .95)) {
             mcmc_scale *= 1.1;
         }
    }

    void incrementNumIter() {
        num_iter++;
    }

    void incrementNumAccept() {
        num_accept++;
        total_accept++;
    }

    void resetNumAccept() {
        num_accept = 0;
    }

    void update(bool accept, const Eigen::VectorXd & cur_theta) {
        incrementNumIter();
        if (num_iter < n_burn) {
            updateRunningStats(cur_theta);
            if (accept)
                incrementNumAccept();
            if ((num_iter % n_check) == 0) {
                updateMcmcScale();
                resetNumAccept();
            }
            setParL();
        }
    }

    void setParL() {
        if (total_accept < 2 * p) {
            par_tmp.setIdentity();
            par_tmp *= mcmc_scale;
            par_tmp.noalias() += par_cov;
            par_L = par_tmp.llt().matrixL();
            return;
        }
        par_L = par_cov.llt().matrixL();
    }

    Eigen::MatrixXd getCov() {
        return par_cov;
    }

    double getScale() {
        return mcmc_scale;
    }
};

class CovarianceTemplate {
public:
    int within_theta_offset_val;
    int p;
    Eigen::VectorXd lower_tri;
    MHtune theta_tuner;

    CovarianceTemplate(int offset_val_, int p_, double rho_,
            int n_burn_);

    void proposeTheta(RngStream rng, Eigen::VectorXd & theta,
            MHValues & mhv);

    void acceptLastProposal(bool accept);

    void fillTheta(Eigen::VectorXd & theta) const {
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
        return DL;
    }

    void setSigma(const Eigen::VectorXd & new_sig) {
        if (new_sig.size() != sigma.size()) {
            Rcpp::stop("invalid length of new sigma values");
        }
        sigma = new_sig;
        fillCholeskyDecomp();
    }

    Eigen::VectorXd getSigma() const {
        return sigma;
    }

    void setRho(const double new_rho) {
        rho = new_rho;
        fillCholeskyDecomp();
    }

    double getRho() const {
        return rho;
    }

private:
    double rho, prop_rho;
    Eigen::VectorXd sigma, prop_sigma;
    Eigen::MatrixXd D;
    Eigen::MatrixXd L_cor;
    Eigen::MatrixXd DL;
    Eigen::VectorXd internal_theta, prop_internal_theta;

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
        offset_lambda.resizeLike(offset_b__);
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
        block_lambdat.clear();
        block_lambdat.reserve(nblock);
        for (int k = 0; k < nblock; ++k) {
            nb = offset_b__(k + 1) - offset_b__(k);
            osb = offset_b__(k);
            Eigen::SparseMatrix<double> tmp = l_init.block(osb, osb, nb, nb);
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

    Eigen::VectorXi getOffsetLambda() const {
        return offset_lambda;
    }

    Eigen::SparseMatrix<double> getLambdaBlock(int block_idx) {
        return block_lambdat[block_idx];
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
    Eigen::SparseMatrix<double> Zt, Omega, I_p, LtZt, Lambdat;

    // helpers for computation and avoiding ubiquitous temporaries
    LinearHelpers helpers;
    Eigen::VectorXd work_y_vec, work_theta_vec;

    // convenient things
    Eigen::VectorXd fitted;
    bool is_null;

    Linear(SEXP r_linear_terms, int n_burn);
    Linear(SEXP r_linear_terms, int n_burn, const Hypers & hyp);

    void updateTheta(RngStream rng, const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & mfi_obs_weights, const Eigen::VectorXd & data_y,
            const Hypers & hypers);

    void updateU(RngStream rng, const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & mfi_obs_weights, const Eigen::VectorXd & data_y,
            const Hypers & hypers);

    void updateComponent(RngStream rng, const Eigen::VectorXd & latent_fitted,
            const Eigen::VectorXd & weights, const Eigen::VectorXd & data_y,
            const int k,  const Eigen::SparseMatrix<double> & zt_block,
            Eigen::SparseMatrix<double> & lambdat_block, CovarianceTemplate & cvt,
            const Hypers & hyp);

    void setFitted() {
        b = Lambdat.transpose() * u;
        fitted = Zt.transpose() * b;
    }

    bool isNull() const {
        return is_null;
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

    void setLambdaHelperBlock(const Eigen::VectorXd & new_theta, int block_idx);
    void setLambdaBlock(const Eigen::VectorXd & new_theta, int block_idx);
};

/*
 * Functor class for evaluating the marginal density of shape parameter among
 * n gamma(shape, rate) distributed variables, where shape/rate have exponential
 * priors
 */

class ShapeDensity {
private:
    // summary statistics
    double sum_nu;
    double sum_log_nu;
    int n;

    // slice sampler stuff;
    double w;
    const double exponential_weight;
    const double lower;
    const double upper;

    // prior
    double lambda_a_prior;
    double lambda_b_prior;

public:
    ShapeDensity(const double w_init, const double weight, const double la, const double lb) :
        sum_nu(1.0),
        sum_log_nu(0.0),
        n(1),
        w(w_init),
        exponential_weight(weight),
        lower(0.0),
        upper(INFINITY),
        lambda_a_prior(la),
        lambda_b_prior(lb) { };

    double operator() (double x) const {
        if (x <= 0.0) {
            return -INFINITY;
        }
        return R::lgammafn(n * x + 1.0) - n * x * log(sum_nu + lambda_b_prior) +
                        x * (sum_log_nu - lambda_a_prior) - n * R::lgammafn(x);
    }

    void setSummaryStatistics(const Eigen::VectorXd & nu_vec) {
        n = nu_vec.size();
        sum_nu = nu_vec.sum();
        sum_log_nu = 0.0;
        for (int i = 0; i < n; i++) {
            sum_log_nu += log(nu_vec(i));
        }
        return;
    };

    void setPrior(const double la, const double lb) {
        lambda_a_prior = la;
        lambda_b_prior = lb;
    }

    void updateW(const double latest_w) {
        w = w * (1 - exponential_weight) + exponential_weight * latest_w;
    }

    inline double getW() const {
        return w;
    }

    void setW(const double new_w) {
        w = new_w;
    }

    inline double getLower() const {
        return lower;
    }

    inline double getUpper() const {
        return upper;
    }

    inline double getSumNu() const {
        return sum_nu;
    }

    inline double getSumLogNu() const {
        return sum_log_nu;
    }

    inline int getN() const {
        return n;
    }
};

class Hypers {
    // vector of required list elements from R.
    bool names_initialized;
    std::vector<std::string> r_names;
    void initializeListNames() {
        if (names_initialized)
            return;
        r_names.push_back("lambda_a_prior");
        r_names.push_back("lambda_b_prior");
        r_names.push_back("bead_precision_rate");
        r_names.push_back("bead_precision_shape");
        r_names.push_back("p_alpha");
        r_names.push_back("p_beta");
        r_names.push_back("n_0");
        r_names.push_back("tau");
        r_names.push_back("tau_prior_shape");
        r_names.push_back("tau_prior_rate");
        r_names.push_back("mu_overall_bar");
        r_names.push_back("cauchy_sd_scale");
        r_names.push_back("prec_mean_prior_mean");
        r_names.push_back("prec_mean_prior_sd");
        r_names.push_back("prec_var_prior_mean");
        r_names.push_back("prec_var_prior_sd");
        names_initialized = true;
        return;
    };
public:
    Rcpp::List fixed_hypers;

    //fixed
    double lambda_a_prior;
    double lambda_b_prior;
    ShapeDensity shape_sampler;

    double bead_precision_rate;
    double bead_precision_shape;
    double p_alpha;
    double p_beta;
    double tau_prior_shape, tau_prior_rate;
    double mu_overall_bar;
    double n_0;
    double theta_rate, theta_shape;
    double tau;

    // estimated
    double mfi_nu_shape;
    double mfi_nu_rate;
    double cauchy_sd_scale;
    double prec_mean_prior_mean;
    double prec_mean_prior_sd;
    double prec_var_prior_mean;
    double prec_var_prior_sd;

    double mu_overall;

    Eigen::SparseMatrix<double> At;
    Eigen::VectorXd p;
    Eigen::VectorXd a_counts, total_counts;

    Hypers(SEXP r_fixed_hyper, SEXP r_at_matrix);
    Hypers(SEXP r_at_matrix);
    void update(RngStream rng, const Eigen::VectorXd & mfi_precision, const Eigen::VectorXd & gamma,
            const Eigen::VectorXd & mu_vec, const std::string var_prior);
    void updateP(RngStream rng, const Eigen::VectorXd & gamma);
    void updateMuPrior(RngStream rng, const Eigen::VectorXd & gamma,
            const Eigen::VectorXd & mu_vec);
    void updateCauchySdScale(RngStream rng, const Eigen::VectorXd & mfi_precision);
    void updateGammaMeanVarPrior(RngStream rng, const Eigen::VectorXd & mfi_precision);

    inline double integratedPrecisionConditional(double x, int n,
            double sum_nu, double sum_log_nu) {
        if (x <= 0.0) {
            return -INFINITY;
        }
        return R::lgammafn(n * x + 1.0) - n * x * log(sum_nu + lambda_b_prior) +
                x * (sum_log_nu - lambda_a_prior) - n * R::lgammafn(x);
    };

    Rcpp::List initializeHypersList(SEXP r_fixed_hypers) {
        initializeListNames();  //should only be called once.
        Rcpp::List tmp_list(r_fixed_hypers);
        bool is_ok = checkListNames(r_names, tmp_list);
        if (!is_ok) {
            Rcpp::stop("Missing required names for fixed hyper-parameters from R.");
        }
        return tmp_list;
    };
};

double thetaLikelihood(const Eigen::VectorXd & delta, const Eigen::VectorXd & weights);

double thetaLikelihood(const Eigen::VectorXd & weights,
        const Eigen::SparseMatrix<double> omega,
        const Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & solver,
        const Eigen::VectorXd & tau,  Eigen::VectorXd & u);
#endif
