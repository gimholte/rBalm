class SimpleBalm {
public:
    ModelData data;
    ChainParameters chain_pars;
    Latent latent;
    Linear linear;
    Hypers hypers;

    // helpers
    RngStream rng;
    Eigen::VectorXd all_fitted;

    // trace
    Eigen::MatrixXd trace_b;
    Eigen::MatrixXd trace_mu_g;
    Eigen::MatrixXd trace_nu_g;
    Eigen::MatrixXd trace_hypers;
    Eigen::MatrixXd trace_bead_mfi;
    Eigen::MatrixXd trace_bead_prec;
    Eigen::MatrixXd trace_theta;

    SimpleBalm(SEXP r_bead_list, SEXP r_chain_pars,
            SEXP r_latent_terms, SEXP r_linear_terms);

    void iterate();
    void gatherTrace(const int i);
};
