class beadDist {
    Rcpp::NumericVector sort_x;
    Rcpp::NumericVector d;
    double mu;
    double cur_smd;
    double mcmc_scale_mult;
    int n;
    double prec;

    double median;
    double sum_d;
    double prec_prior_shape;
    double prec_prior_rate;
    int j;
public:
    beadDist() :
        median(0.0),
        sum_d(0.0),
        prec_prior_shape(1.0),
        prec_prior_rate(1.0),
        j(1) {};

    beadDist(Rcpp::List & bead_list, double prior_shape = 1.0, double prior_rate = 1.0);
    double computeLaplaceLikelihood(const double sum_abs_dev_x);
    double computeSumAbsDev(const double x);
    void updateLaplacePrecision(RngStream & rng);
    void updateLaplaceMu(RngStream & rng, const double mu_prior_mean, const double mu_prior_prec);

    inline double getMu() {
        return mu;
    }
    inline double getPrecision() {
        return prec;
    }
    inline double getSmd() {
        return cur_smd;
    }
    inline void setSmd(const double smd) {
        cur_smd = smd;
    }
    inline double getMcmcScaleMult() {
        return mcmc_scale_mult;
    }
    inline void setMcmcScaleMult(const double m) {
        mcmc_scale_mult = m;
    }
};

void parseBeadDistList(Rcpp::List & bead_list, std::vector<beadDist> & bvec,
        Rcpp::List & bead_prior_pars);

