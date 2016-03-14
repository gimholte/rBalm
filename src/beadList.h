class beadList {
    Rcpp::NumericVector sort_x;
    Rcpp::NumericVector d;
    double scale;
    double mu;
    double cur_smd;
    double mcmc_scale_mult;
    int n;

    double median;
    double sad;
    double scale_prior_shape;
    double scale_prior_rate;
    int j;
public:
    beadList() :
        median(0.0),
        sad(0.0),
        scale_prior_shape(1.0),
        scale_prior_rate(1.0),
        j(1) {};

    beadList(Rcpp::List & bead_list, double prior_scale, double prior_rate);
    double computeLaplaceLikelihood(double & x);
    double computeSumAbsDev(double & x);
    void updateLaplaceScale(RngStream & rng);
};
